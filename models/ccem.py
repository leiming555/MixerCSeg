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


class WindowSelfAttention2d(nn.Module):
    def __init__(self, channels: int, window_size: int = 8):
        super().__init__()
        num_heads = 1
        for heads in (4, 2, 1):
            if channels % heads == 0:
                num_heads = heads
                break
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.norm = make_gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = ConvGNAct(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            act=False,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        x = self.norm(x)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        hp, wp = x.shape[-2:]
        num_h = hp // ws
        num_w = wp // ws
        qkv = self.qkv(x).reshape(
            b, 3, self.num_heads, self.head_dim, num_h, ws, num_w, ws
        )
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3).reshape(
            3, b * num_h * num_w, self.num_heads, ws * ws, self.head_dim
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, num_h, num_w, self.num_heads, ws, ws, self.head_dim)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(b, c, hp, wp)
        out = out[:, :, :h, :w]
        return self.proj(out)


class ConvTransformerFFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden_channels = channels * expansion
        self.net = nn.Sequential(
            ConvGNAct(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
                act=True,
            ),
            ConvGNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                act=True,
            ),
            ConvGNAct(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=1,
                padding=0,
                act=False,
            ),
        )

    def forward(self, x):
        return self.net(x)


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
        mode: str = "full",
        gate_mode: str = "original",
        branch_weight: bool = False,
    ):
        super().__init__()

        valid_modes = {"full", "enhanced", "transformer", "no_local", "no_strip", "no_dilation", "no_gate"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported CCEM mode: {mode}. Expected one of {sorted(valid_modes)}.")
        valid_gate_modes = {"original", "leaky"}
        if gate_mode not in valid_gate_modes:
            raise ValueError(f"Unsupported CCEM gate_mode: {gate_mode}. Expected one of {sorted(valid_gate_modes)}.")
        self.mode = mode
        self.is_enhanced = mode == "enhanced"
        self.is_transformer = mode == "transformer"
        self.use_advanced_refine = self.is_enhanced or self.is_transformer
        self.gate_mode = gate_mode
        self.branch_weight = branch_weight
        self.use_local = mode != "no_local"
        self.use_strip = mode != "no_strip"
        self.use_dilation = mode != "no_dilation"
        self.use_gate = mode != "no_gate"

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

        if self.use_local:
            # Branch 1: local edge/detail branch
            self.local_branch = ConvGNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                act=True,
            )

        if self.use_strip:
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

        if self.use_dilation:
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

        if self.use_advanced_refine:
            # Wider dilated context helps connect slightly larger crack gaps.
            enhanced_dilation = dilation + 1
            self.dilation_large_branch = ConvGNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=enhanced_dilation,
                dilation=enhanced_dilation,
                groups=hidden_channels,
                act=True,
            )
            # Low-frequency context suppresses isolated texture responses.
            self.context_branch = ConvGNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
                act=True,
            )

        if self.is_transformer:
            self.window_attn_branch = WindowSelfAttention2d(
                channels=hidden_channels,
                window_size=8,
            )
            self.transformer_ffn_branch = ConvTransformerFFN(
                channels=hidden_channels,
                expansion=2,
            )

        if self.branch_weight:
            if self.use_local:
                self.local_weight = nn.Parameter(torch.ones(1))
            if self.use_strip:
                self.strip_weight = nn.Parameter(torch.ones(1))
            if self.use_dilation:
                self.dilation_weight = nn.Parameter(torch.ones(1))

        # Fuse all branches
        branch_count = int(self.use_local) + int(self.use_strip) * 2 + int(self.use_dilation)
        if self.is_enhanced:
            branch_count += 2
            self.enhanced_branch_logits = nn.Parameter(torch.zeros(branch_count))
        if self.is_transformer:
            branch_count += 4
            self.transformer_branch_logits = nn.Parameter(torch.zeros(branch_count))
        self.fuse = ConvGNAct(
            in_channels=hidden_channels * branch_count,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            act=False,
        )

        if self.use_advanced_refine:
            attn_channels = max(channels // 8, 4)
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, attn_channels, kernel_size=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(attn_channels, channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.spatial_attn = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

        if self.use_gate:
            # Gate: decide where crack-continuity features should be enhanced
            gate_in_channels = channels * 2 if self.use_advanced_refine else channels
            self.gate = nn.Conv2d(gate_in_channels, channels, kernel_size=1, bias=True)

        # Learnable residual scale.
        # Initialized as 0 to avoid destroying pretrained/original features at the beginning.
        gamma_init = 0.1 if self.use_advanced_refine else 0.0
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x):
        identity = x

        x_reduced = self.reduce(x)

        branch_feats = []

        if self.use_local:
            local_feat = self.local_branch(x_reduced)
            if self.branch_weight:
                local_feat = self.local_weight * local_feat
            branch_feats.append(local_feat)

        if self.use_strip:
            strip_h_feat = self.strip_h(x_reduced)
            strip_v_feat = self.strip_v(x_reduced)
            if self.branch_weight:
                strip_h_feat = self.strip_weight * strip_h_feat
                strip_v_feat = self.strip_weight * strip_v_feat
            branch_feats.append(strip_h_feat)
            branch_feats.append(strip_v_feat)

        if self.use_dilation:
            dilation_feat = self.dilation_branch(x_reduced)
            if self.branch_weight:
                dilation_feat = self.dilation_weight * dilation_feat
            branch_feats.append(dilation_feat)

        if self.use_advanced_refine:
            dilation_large_feat = self.dilation_large_branch(x_reduced)
            context_feat = self.context_branch(
                F.avg_pool2d(x_reduced, kernel_size=3, stride=1, padding=1)
            )
            branch_feats.append(dilation_large_feat)
            branch_feats.append(context_feat)

        if self.is_transformer:
            attn_feat = self.window_attn_branch(x_reduced)
            ffn_feat = self.transformer_ffn_branch(x_reduced + attn_feat)
            branch_feats.append(attn_feat)
            branch_feats.append(ffn_feat)

        if self.use_advanced_refine:
            branch_logits = (
                self.transformer_branch_logits
                if self.is_transformer
                else self.enhanced_branch_logits
            )
            branch_weights = F.softmax(branch_logits, dim=0)
            branch_scale = float(len(branch_feats))
            branch_feats = [
                feat * branch_weights[idx] * branch_scale
                for idx, feat in enumerate(branch_feats)
            ]

        fused = torch.cat(branch_feats, dim=1)

        fused = self.fuse(fused)

        if self.use_advanced_refine:
            channel_attn = self.channel_attn(fused)
            spatial_pool = torch.cat(
                [fused.mean(dim=1, keepdim=True), fused.amax(dim=1, keepdim=True)],
                dim=1,
            )
            spatial_attn = torch.sigmoid(self.spatial_attn(spatial_pool))
            fused = fused * channel_attn * spatial_attn

        if self.use_gate:
            gate_input = torch.cat([identity, fused], dim=1) if self.use_advanced_refine else fused
            gate = torch.sigmoid(self.gate(gate_input))
            if self.gate_mode == "leaky":
                gate = 0.5 + 0.5 * gate
            fused = gate * fused

        out = identity + self.gamma * fused

        return out
