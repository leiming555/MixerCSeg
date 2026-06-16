import math
import torch
import torch.nn as nn


def make_gn(num_channels: int, max_groups: int = 8):
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def logit(x: float):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
        groups=1,
        act=True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            make_gn(out_channels),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class BoundaryRefinementModuleV2(nn.Module):
    """
    BRM-v2: Recall-friendly Boundary Refinement Module.

    改进目的：
        原 BRM 提高 Precision 但压低 Recall。
        BRM-v2 只做温和的正向边界增强，避免抑制真实细裂缝。

    放置位置：
        SRF 后，SegHead 前。

    输入：
        x: [B, C, H, W]

    输出：
        out: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        min_hidden_channels: int = 16,
        max_scale: float = 0.10,
        init_scale: float = 0.02,
    ):
        super().__init__()

        hidden_channels = max(channels // reduction, min_hidden_channels)

        self.max_scale = max_scale

        # alpha 经过 sigmoid 后限制在 0~max_scale
        init_alpha = logit(init_scale / max_scale)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        # 1x1 降维
        self.reduce = ConvGNAct(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            act=True,
        )

        # 局部边界分支
        self.local_edge = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            act=True,
        )

        # 空洞边界分支：帮助连接弱裂缝和小断裂
        self.dilated_edge = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=hidden_channels,
            act=True,
        )

        # 融合边界特征
        self.edge_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=False),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            make_gn(channels),
        )

        # 生成边界注意力，只用于控制增强强度，不直接压制原特征
        self.attn = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        identity = x

        feat = self.reduce(x)

        edge_local = self.local_edge(feat)
        edge_dilated = self.dilated_edge(feat)

        edge_feat = torch.cat([edge_local, edge_dilated], dim=1)
        edge_feat = self.edge_fuse(edge_feat)

        attn = self.attn(feat)

        # 正数、小幅度、有上限的增强系数
        scale = self.max_scale * torch.sigmoid(self.alpha)

        # 重点：只做正向残差增强，不做抑制
        out = identity + scale * attn * edge_feat

        return out