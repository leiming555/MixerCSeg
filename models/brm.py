import torch
import torch.nn as nn


def make_gn(num_channels: int, max_groups: int = 8):
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class BoundaryRefinementModule(nn.Module):
    """
    BRM: Boundary Refinement Module

    放置位置：
        SRF 后，SegHead 前。

    作用：
        细化裂缝边界，减少预测 mask 粗糙、边界偏移和背景误检。
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        hidden_channels = max(channels // reduction, 16)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.edge_extract = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.attn = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        feat = self.reduce(x)
        edge_feat = self.edge_extract(feat)
        attn = self.attn(edge_feat)

        out = x + self.gamma * attn * x

        return out