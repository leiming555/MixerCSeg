import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(num_channels: int, max_groups: int = 8):
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class SkeletonGuidanceHead(nn.Module):
    """
    SGH: Skeleton Guidance Head

    用途：
        训练期辅助预测裂缝骨架，增强细裂缝连续性。
        推理期可以不用该分支，不增加推理开销。

    输入：
        SRF 输出特征 x: [B, C, H, W]

    输出：
        skeleton_logit: [B, 1, H, W]
    """

    def __init__(self, channels: int, hidden_channels: int = 64):
        super().__init__()

        hidden_channels = min(hidden_channels, channels)

        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            make_gn(hidden_channels),
            nn.SiLU(inplace=True),

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

            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.head(x)