import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    GroupNorm is more stable than BatchNorm when batch size is small.
    The original paper uses batch size = 1, so GroupNorm is preferred.
    """
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        act=True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = make_gn(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CrackContinuityEnhancementModule(nn.Module):
    """
    CCEM: Crack Continuity Enhancement Module.

    Purpose:
        Enhance thin and elongated crack structures.
        Reduce discontinuous predictions in crack segmentation.

    Structure:
        Input feature F
            ↓
        1x1 channel reduction
            ↓
        Three continuity-aware branches:
            1. local branch: 3x3 depthwise conv
            2. strip branch: 1xk and kx1 depthwise conv
            3. dilation branch: dilated 3x3 depthwise conv
            ↓
        concat + 1x1 fusion
            ↓
        spatial-channel gate
            ↓
        residual output

    Input:
        x: Tensor[B, C, H, W]

    Output:
        Tensor[B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        strip_kernel: int = 7,
        dilation: int = 2,
        min_hidden_channels: int = 16,
    ):
        super().__init__()

        hidden_channels = max(channels // reduction, min_hidden_channels)

        assert strip_kernel % 2 == 1, "strip_kernel should be odd, e.g. 5, 7, 9."

        # 1x1 projection: reduce computation
        self.reduce = ConvGNAct(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            act=True,
        )

        # Branch 1: local edge/detail branch
        self.local_branch = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            act=True,
        )

        # Branch 2: strip convolution branch for elongated cracks
        self.strip_h = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(1, strip_kernel),
            padding=(0, strip_kernel // 2),
            groups=hidden_channels,
            act=True,
        )

        self.strip_v = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(strip_kernel, 1),
            padding=(strip_kernel // 2, 0),
            groups=hidden_channels,
            act=True,
        )

        # Branch 3: dilated branch for bridging small crack gaps
        self.dilation_branch = ConvGNAct(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=hidden_channels,
            act=True,
        )

        # Fuse all branches
        self.fuse = ConvGNAct(
            in_channels=hidden_channels * 4,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            act=False,
        )

        # Gate: decide where crack-continuity features should be enhanced
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Learnable residual scale.
        # Initialized as 0 to avoid destroying pretrained/original features at the beginning.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

        x_reduced = self.reduce(x)

        local_feat = self.local_branch(x_reduced)

        strip_h_feat = self.strip_h(x_reduced)
        strip_v_feat = self.strip_v(x_reduced)

        dilation_feat = self.dilation_branch(x_reduced)

        fused = torch.cat(
            [local_feat, strip_h_feat, strip_v_feat, dilation_feat],
            dim=1,
        )

        fused = self.fuse(fused)

        gate = self.gate(fused)

        out = identity + self.gamma * gate * fused

        return out