import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENTAL_MODULE_MODES = [
    "win_attn_strip_gate",
    "axial_attn_context",
    "freq_lap_mixer",
    "dynamic_kernel_gate",
    "lowrank_global_fuse",
    "pyramid_context_gate",
    "shifted_window_mixer",
    "direction_selective_strip",
    "edge_memory_gate",
    "criss_cross_lite",
    "wavelet_hf_context",
    "multi_order_dwconv",
    "graph_line_propagation",
    "token_affinity_refine",
    "boundary_confidence_gate",
    "snake_strip_mixer",
    "contextual_mlp_gate",
    "dilated_attn_hybrid",
    "scale_adaptive_fusion",
    "contour_transform_mixer",
]


MODE_BRANCHES = {
    "win_attn_strip_gate": ("window", "strip", "local"),
    "axial_attn_context": ("axial", "lowrank", "local"),
    "freq_lap_mixer": ("frequency", "pyramid", "local"),
    "dynamic_kernel_gate": ("dynamic", "strip", "local"),
    "lowrank_global_fuse": ("lowrank", "mlp", "local"),
    "pyramid_context_gate": ("pyramid", "scale", "local"),
    "shifted_window_mixer": ("shift_window", "strip", "mlp"),
    "direction_selective_strip": ("strip", "snake", "dynamic"),
    "edge_memory_gate": ("contour", "frequency", "mlp"),
    "criss_cross_lite": ("axial", "line", "local"),
    "wavelet_hf_context": ("frequency", "lowrank", "pyramid"),
    "multi_order_dwconv": ("multi_order", "local", "pyramid"),
    "graph_line_propagation": ("line", "strip", "lowrank"),
    "token_affinity_refine": ("affinity", "window", "local"),
    "boundary_confidence_gate": ("contour", "dynamic", "strip"),
    "snake_strip_mixer": ("snake", "line", "strip"),
    "contextual_mlp_gate": ("mlp", "lowrank", "dynamic"),
    "dilated_attn_hybrid": ("pyramid", "window", "dynamic"),
    "scale_adaptive_fusion": ("scale", "lowrank", "pyramid"),
    "contour_transform_mixer": ("contour", "window", "frequency"),
}


def make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    for groups in (max_groups, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
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
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = make_gn(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class LocalBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        return self.net(x)


class StripBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        self.h = ConvGNAct(channels, channels, kernel_size=(1, kernel_size), padding=(0, pad), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=channels)
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.h(x), self.v(x)], dim=1))


class SnakeStripBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.h1 = ConvGNAct(channels, channels, kernel_size=(1, kernel_size), padding=(0, pad), groups=channels)
        self.v1 = ConvGNAct(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=channels)
        self.v2 = ConvGNAct(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=channels)
        self.h2 = ConvGNAct(channels, channels, kernel_size=(1, kernel_size), padding=(0, pad), groups=channels)
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        hv = self.v2(self.h1(x))
        vh = self.h2(self.v1(x))
        return self.fuse(torch.cat([hv, vh], dim=1))


class PyramidBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.d1 = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.d2 = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.d3 = ConvGNAct(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.d1(x), self.d2(x), self.d3(x)], dim=1))


class WindowAttentionBranch(nn.Module):
    def __init__(self, channels: int, window_size: int = 8, shifted: bool = False):
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
        self.shifted = shifted
        self.norm = make_gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        shift = ws // 2 if self.shifted else 0
        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(-2, -1))

        x = self.norm(x)
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        hp, wp = x.shape[-2:]
        num_h, num_w = hp // ws, wp // ws
        qkv = self.qkv(x).reshape(
            b, 3, self.num_heads, self.head_dim, num_h, ws, num_w, ws
        )
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3).reshape(
            3, b * num_h * num_w, self.num_heads, ws * ws, self.head_dim
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, num_h, num_w, self.num_heads, ws, ws, self.head_dim)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(b, c, hp, wp)
        out = out[:, :, :h, :w]
        if shift:
            out = torch.roll(out, shifts=(shift, shift), dims=(-2, -1))
        return self.proj(out)


class AxialContextBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        row = x.mean(dim=-1, keepdim=True).expand_as(x)
        col = x.mean(dim=-2, keepdim=True).expand_as(x)
        return self.fuse(torch.cat([x, row, col], dim=1))


class FrequencyBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        return self.fuse(torch.cat([high, torch.abs(high)], dim=1))


class ContourBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", kernel)
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        kernel = self.laplace_kernel.to(dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        edge = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        return self.fuse(torch.cat([edge, torch.abs(edge)], dim=1))


class DynamicKernelBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.k3 = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.k5 = ConvGNAct(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.k7 = ConvGNAct(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True),
        )
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        feats = torch.stack([self.k3(x), self.k5(x), self.k7(x)], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((feats * weights).sum(dim=1))


class LowRankGlobalBranch(nn.Module):
    def __init__(self, channels: int, pool_size: int = 4):
        super().__init__()
        self.pool_size = pool_size
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        context = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, context], dim=1))


class LinePropagationBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        h_norm = torch.arange(1, x.shape[-2] + 1, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
        w_norm = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        row_flow = torch.cumsum(x, dim=-2) / h_norm
        col_flow = torch.cumsum(x, dim=-1) / w_norm
        return self.fuse(torch.cat([x, row_flow, col_flow], dim=1))


class MultiOrderBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        x1 = self.dw(x)
        x2 = self.dw(x1)
        return self.fuse(torch.cat([x, x1, x2], dim=1))


class AffinityBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        q = self.q(x)
        k = F.avg_pool2d(self.k(x), kernel_size=3, stride=1, padding=1)
        gate = torch.sigmoid(q * k)
        return self.proj(self.v(x) * gate)


class MLPBranch(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden_channels = channels * expansion
        self.net = nn.Sequential(
            ConvGNAct(channels, hidden_channels, kernel_size=1, padding=0),
            ConvGNAct(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            ConvGNAct(hidden_channels, channels, kernel_size=1, padding=0, act=False),
        )

    def forward(self, x):
        return self.net(x)


class ScaleAdaptiveBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        size = x.shape[-2:]
        p2 = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        p4 = F.avg_pool2d(x, kernel_size=4, stride=4, ceil_mode=True)
        p2 = F.interpolate(p2, size=size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, p2, p4], dim=1))


def make_branch(name: str, channels: int) -> nn.Module:
    if name == "local":
        return LocalBranch(channels)
    if name == "strip":
        return StripBranch(channels)
    if name == "snake":
        return SnakeStripBranch(channels)
    if name == "pyramid":
        return PyramidBranch(channels)
    if name == "window":
        return WindowAttentionBranch(channels, shifted=False)
    if name == "shift_window":
        return WindowAttentionBranch(channels, shifted=True)
    if name == "axial":
        return AxialContextBranch(channels)
    if name == "frequency":
        return FrequencyBranch(channels)
    if name == "contour":
        return ContourBranch(channels)
    if name == "dynamic":
        return DynamicKernelBranch(channels)
    if name == "lowrank":
        return LowRankGlobalBranch(channels)
    if name == "line":
        return LinePropagationBranch(channels)
    if name == "multi_order":
        return MultiOrderBranch(channels)
    if name == "affinity":
        return AffinityBranch(channels)
    if name == "mlp":
        return MLPBranch(channels)
    if name == "scale":
        return ScaleAdaptiveBranch(channels)
    raise ValueError(f"Unsupported experimental branch: {name}")


class ExperimentalEnhancementModule(nn.Module):
    def __init__(
        self,
        channels: int,
        mode: str,
        reduction: int = 4,
        min_hidden_channels: int = 16,
    ):
        super().__init__()
        if mode not in MODE_BRANCHES:
            raise ValueError(
                f"Unsupported exp_module_mode: {mode}. Expected one of {EXPERIMENTAL_MODULE_MODES}."
            )
        self.mode = mode
        hidden_channels = max(channels // reduction, min_hidden_channels)
        branch_names = MODE_BRANCHES[mode]
        self.reduce = ConvGNAct(channels, hidden_channels, kernel_size=1, padding=0)
        self.branches = nn.ModuleList(
            [make_branch(branch_name, hidden_channels) for branch_name in branch_names]
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, len(branch_names), kernel_size=1, bias=True),
        )
        self.fuse = ConvGNAct(hidden_channels * len(branch_names), channels, kernel_size=1, padding=0, act=False)
        attn_channels = max(channels // 8, 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, attn_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(attn_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x
        reduced = self.reduce(x)
        branch_feats = [branch(reduced) for branch in self.branches]
        weights = self.router(reduced).softmax(dim=1)
        weighted_feats = [
            feat * weights[:, idx:idx + 1] * float(len(branch_feats))
            for idx, feat in enumerate(branch_feats)
        ]
        fused = self.fuse(torch.cat(weighted_feats, dim=1))
        channel_attn = self.channel_attn(fused)
        spatial_pool = torch.cat(
            [fused.mean(dim=1, keepdim=True), fused.amax(dim=1, keepdim=True)],
            dim=1,
        )
        fused = fused * channel_attn * torch.sigmoid(self.spatial_attn(spatial_pool))
        gate = torch.sigmoid(self.gate(torch.cat([identity, fused], dim=1)))
        return identity + self.gamma * gate * fused
