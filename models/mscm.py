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


class MultiScaleContextModule(nn.Module):
    """
    MSCM: Multi-Scale Context Module

    作用：
        增强深层语义特征中的多尺度上下文信息，
        帮助模型恢复断裂裂缝、低对比度裂缝和长距离裂缝结构。

    推荐放置：
        DEGConv/CCEM 后，SRF 前。
        更推荐加在 F3/F4 深层特征上，不建议加在最终输出前。

    输入:
        x: [B, C, H, W]

    输出:
        out: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        min_hidden_channels: int = 16,
        max_scale: float = 0.15,
        init_scale: float = 0.03,
    ):
        super().__init__()

        hidden_channels = max(channels // reduction, min_hidden_channels)

        self.max_scale = max_scale
        init_alpha = logit(init_scale / max_scale)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        self.reduce = ConvGNAct(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            act=True,
        )

        # 小感受野：保留局部结构
        self.branch_d1 = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            groups=hidden_channels,
            act=True,
        )

        # 中等感受野：连接小断裂
        self.branch_d2 = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=hidden_channels,
            act=True,
        )

        # 大感受野：增强长距离上下文
        self.branch_d3 = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=hidden_channels,
            act=True,
        )

        # 全局上下文分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels, kernel_size=1, bias=False),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            make_gn(channels),
        )

    def forward(self, x):
        identity = x

        feat = self.reduce(x)

        f1 = self.branch_d1(feat)
        f2 = self.branch_d2(feat)
        f3 = self.branch_d3(feat)

        fg = self.global_branch(feat)
        fg = fg.expand_as(feat)

        fused = torch.cat([f1, f2, f3, fg], dim=1)
        fused = self.fuse(fused)

        # 只做正向残差增强，不直接抑制原特征，避免 Recall 大幅下降
        scale = self.max_scale * torch.sigmoid(self.alpha)
        out = identity + scale * fused

        return out
