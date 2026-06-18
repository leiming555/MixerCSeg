import math
import torch
import torch.nn as nn


def make_gn(num_channels: int, max_groups: int = 8):
    """
    GroupNorm is more stable when batch size is small.
    """
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


class EdgeDetailRecoveryModule(nn.Module):
    """
    EDRM: Edge Detail Recovery Module

    作用：
        用于恢复浅层特征中的细裂缝边缘和弱纹理细节。

    推荐位置：
        DEGConv 后，CCEM 前。

    推荐使用：
        只加在 F1，最多加在 F1/F2。

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
        init_alpha = logit(init_scale / max_scale)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        # 1x1 降维，降低计算量
        self.reduce = ConvGNAct(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            act=True,
        )

        # 分支 1：3x3 深度卷积，提取局部边缘细节
        self.branch_3x3 = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            act=True,
        )

        # 分支 2：5x5 深度卷积，捕获稍大范围的细裂缝纹理
        self.branch_5x5 = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=5,
            padding=2,
            groups=hidden_channels,
            act=True,
        )

        # 分支 3：简单残差边缘分支，增强弱响应
        self.branch_dilated = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=hidden_channels,
            act=True,
        )

        # 融合三个分支
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1, bias=False),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            make_gn(channels),
        )

    def forward(self, x):
        identity = x

        feat = self.reduce(x)

        f3 = self.branch_3x3(feat)
        f5 = self.branch_5x5(feat)
        fd = self.branch_dilated(feat)

        edge_feat = torch.cat([f3, f5, fd], dim=1)
        edge_feat = self.fuse(edge_feat)

        # 只做小幅正向残差增强，避免像 BRM 一样压低 Recall
        scale = self.max_scale * torch.sigmoid(self.alpha)
        out = identity + scale * edge_feat

        return out