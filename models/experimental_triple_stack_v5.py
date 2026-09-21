import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.layers import HoGEdgeGateConv


TRIPLE_STACK_V5_MODES = [
    "mwr_eoc_upb",
    "mwr_eoc_tcb",
    "mwr_rtc_upb",
    "mwr_rtc_tcb",
    "aca_eoc_upb",
    "aca_eoc_tcb",
    "aca_rtc_upb",
    "aca_rtc_tcb",
    "fap_eoc_upb",
    "fap_eoc_tcb",
    "fap_rtc_upb",
    "fap_rtc_tcb",
    "sat_eoc_upb",
    "sat_eoc_tcb",
    "sat_rtc_upb",
    "sat_rtc_tcb",
    "ads_eoc_upb",
    "ads_eoc_tcb",
    "ads_rtc_upb",
    "ads_rtc_tcb",
]


def _depthwise_filter(x, kernel, padding=1):
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    return F.conv2d(
        x,
        kernel.repeat(x.shape[1], 1, 1, 1),
        padding=padding,
        groups=x.shape[1],
    )


class DifferentialKernelsMixin:
    def _init_diff_kernels(self):
        dxx = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 1.0],
            [0.0, 0.0, 0.0],
        ]).view(1, 1, 3, 3)
        dyy = dxx.transpose(-1, -2)
        dxy = 0.25 * torch.tensor([
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]).view(1, 1, 3, 3)
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer("dxx", dxx)
        self.register_buffer("dyy", dyy)
        self.register_buffer("dxy", dxy)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _hessian(self, x):
        return (
            _depthwise_filter(x, self.dxx),
            _depthwise_filter(x, self.dyy),
            _depthwise_filter(x, self.dxy),
        )

    def _sobel(self, x):
        gx = _depthwise_filter(x, self.sobel_x)
        gy = _depthwise_filter(x, self.sobel_y)
        magnitude = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        return gx, gy, magnitude


class MultiScaleWidthRouter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.h3 = ConvGNAct(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.v3 = ConvGNAct(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        self.h7 = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.v7 = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.h11 = ConvGNAct(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.v11 = ConvGNAct(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.router = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        widths = []
        for kernel_size in (3, 7, 11):
            local_max = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            widths.append(torch.abs(local_max - local_mean))
        width_stack = torch.stack(widths, dim=1)
        width_maps = torch.stack([item.mean(dim=1) for item in widths], dim=1)
        weights = self.router(width_maps).softmax(dim=1).unsqueeze(2)
        branches = torch.stack([
            self.h3(x) + self.v3(x),
            self.h7(x) + self.v7(x),
            self.h11(x) + self.v11(x),
        ], dim=1)
        mixed = (branches * weights).sum(dim=1)
        width_prior = (width_stack * weights).sum(dim=1)
        return self.fuse(torch.cat([x, mixed, width_prior], dim=1))


class AxialCurvatureAligner(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.horizontal = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.vertical = ConvGNAct(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels)
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.anti_diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        gx, gy, magnitude = self._sobel(x)
        jxx = F.avg_pool2d(gx.pow(2), kernel_size=3, stride=1, padding=1)
        jyy = F.avg_pool2d(gy.pow(2), kernel_size=3, stride=1, padding=1)
        jxy = F.avg_pool2d(gx * gy, kernel_size=3, stride=1, padding=1)
        discriminant = torch.sqrt((jxx - jyy).pow(2) + 4.0 * jxy.pow(2) + 1e-6)
        coherence = discriminant / (jxx + jyy + 1e-6)
        cos2 = (jxx - jyy) / (discriminant + 1e-6)
        sin2 = (2.0 * jxy) / (discriminant + 1e-6)
        weights = torch.stack([cos2, -cos2, sin2, -sin2], dim=1).softmax(dim=1)
        branches = torch.stack([
            self.horizontal(x),
            self.vertical(x),
            self.diagonal(x),
            self.anti_diagonal(x),
        ], dim=1)
        aligned = (branches * weights).sum(dim=1)
        return self.fuse(torch.cat([x, aligned, coherence * magnitude], dim=1))


class FrequencyAxialPyramid(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.small = ConvGNAct(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.medium = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.large = ConvGNAct(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 3, kernel_size=1))
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        low3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        low7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        low11 = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
        bands = torch.stack([x - low3, low3 - low7, low7 - low11], dim=1)
        branches = torch.stack([
            self.small(bands[:, 0]),
            self.medium(bands[:, 1]),
            self.large(bands[:, 2]),
        ], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        mixed = (branches * weights).sum(dim=1)
        high_frequency = bands.abs().mean(dim=1)
        return self.fuse(torch.cat([x, mixed, high_frequency, low11], dim=1))


class SparseAxialTokenConditioner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.vertical = ConvGNAct(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels)
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.anti_diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.memory = nn.Parameter(torch.randn(1, 4, channels) * 0.02)
        self.token_norm = nn.LayerNorm(channels)
        self.token_attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.token_score = nn.Linear(channels, 1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        branches = torch.stack([
            self.horizontal(x),
            self.vertical(x),
            self.diagonal(x),
            self.anti_diagonal(x),
        ], dim=1)
        queries = self.token_norm(branches.mean(dim=(-2, -1)))
        memory = self.memory.expand(x.shape[0], -1, -1)
        tokens, _ = self.token_attn(queries, memory, memory, need_weights=False)
        weights = self.token_score(tokens).softmax(dim=1).unsqueeze(-1).unsqueeze(-1)
        mixed = (branches * weights).sum(dim=1)
        context = tokens.mean(dim=1)[:, :, None, None].expand_as(x)
        gap = F.relu(F.max_pool2d(mixed, kernel_size=5, stride=1, padding=2) - mixed)
        return self.fuse(torch.cat([x, mixed, context, gap], dim=1))


class AnisotropicDynamicShiftConditioner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvGNAct(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels),
            ConvGNAct(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels),
            ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels),
            ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels),
        ])
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 4, kernel_size=1))
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        shifted = [
            0.5 * (torch.roll(x, 1, -1) + torch.roll(x, -1, -1)),
            0.5 * (torch.roll(x, 1, -2) + torch.roll(x, -1, -2)),
            0.5 * (torch.roll(x, (1, 1), (-2, -1)) + torch.roll(x, (-1, -1), (-2, -1))),
            0.5 * (torch.roll(x, (2, -2), (-2, -1)) + torch.roll(x, (-2, 2), (-2, -1))),
        ]
        branches = torch.stack([module(item) for module, item in zip(self.branches, shifted)], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        mixed = (branches * weights).sum(dim=1)
        horizontal_vertical = shifted[0] + shifted[1]
        diagonals = shifted[2] + shifted[3]
        return self.fuse(torch.cat([x, mixed, horizontal_vertical, diagonals], dim=1))


class EigenOrientationConsensus(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.horizontal = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.vertical = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.anti_diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        dxx, dyy, dxy = self._hessian(x)
        gx, gy, magnitude = self._sobel(x)
        trace = dxx + dyy
        discriminant = torch.sqrt((dxx - dyy).pow(2) + 4.0 * dxy.pow(2) + 1e-6)
        eigen_high = 0.5 * (trace + discriminant)
        eigen_low = 0.5 * (trace - discriminant)
        ridge = torch.abs(torch.abs(eigen_high) - torch.abs(eigen_low))
        curvature = torch.abs(trace) + torch.abs(dxy)
        hessian_cos = (dxx - dyy) / (discriminant + 1e-6)
        hessian_sin = (2.0 * dxy) / (discriminant + 1e-6)
        gradient_den = gx.pow(2) + gy.pow(2) + 1e-6
        gradient_cos = (gx.pow(2) - gy.pow(2)) / gradient_den
        gradient_sin = (2.0 * gx * gy) / gradient_den
        cos2 = 0.5 * (hessian_cos + gradient_cos)
        sin2 = 0.5 * (hessian_sin + gradient_sin)
        weights = torch.stack([cos2, -cos2, sin2, -sin2], dim=1).softmax(dim=1)
        branches = torch.stack([
            self.horizontal(x),
            self.vertical(x),
            self.diagonal(x),
            self.anti_diagonal(x),
        ], dim=1)
        oriented = (branches * weights).sum(dim=1)
        consensus = torch.sigmoid(ridge + magnitude - curvature) * oriented
        return self.fuse(torch.cat([x, ridge, curvature, oriented, consensus], dim=1))


class RidgeTopologyCrossAttention(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.token_norm = nn.LayerNorm(channels)
        self.token_attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.token_score = nn.Linear(channels, 1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        dxx, dyy, dxy = self._hessian(x)
        ridge = torch.sqrt((dxx - dyy).pow(2) + 4.0 * dxy.pow(2) + 1e-6)
        plus = 0.25 * (
            torch.roll(x, 1, -1)
            + torch.roll(x, -1, -1)
            + torch.roll(x, 1, -2)
            + torch.roll(x, -1, -2)
        )
        diagonal = 0.25 * (
            torch.roll(x, (1, 1), (-2, -1))
            + torch.roll(x, (-1, -1), (-2, -1))
            + torch.roll(x, (1, -1), (-2, -1))
            + torch.roll(x, (-1, 1), (-2, -1))
        )
        topology = torch.sigmoid(plus + diagonal - torch.abs(plus - diagonal)) * ridge
        gap = F.relu(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x)
        maps = torch.stack([x, ridge, topology, gap], dim=1)
        tokens = self.token_norm(maps.mean(dim=(-2, -1)))
        tokens, _ = self.token_attn(tokens, tokens, tokens, need_weights=False)
        weights = self.token_score(tokens).softmax(dim=1).unsqueeze(-1).unsqueeze(-1)
        mixed = (maps * weights).sum(dim=1)
        context = tokens.mean(dim=1)[:, :, None, None].expand_as(x)
        return self.fuse(torch.cat([x, mixed, context, topology * torch.sigmoid(gap)], dim=1))


class UncertaintyPrecisionBridge(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(4, 3, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        local_mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_max = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_min = -F.max_pool2d(-x, kernel_size=5, stride=1, padding=2)
        gap = F.relu(local_max - x)
        boundary = torch.abs(x - local_mean)
        contrast = local_max - local_min
        p = torch.sigmoid(x)
        uncertainty = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        evidence = torch.cat([
            gap.amax(dim=1, keepdim=True),
            boundary.mean(dim=1, keepdim=True),
            contrast.amax(dim=1, keepdim=True),
            uncertainty.mean(dim=1, keepdim=True),
        ], dim=1)
        gates = self.spatial(evidence).softmax(dim=1)
        fill = x + torch.sigmoid(boundary) * gap
        refined = gates[:, :1] * x + gates[:, 1:2] * fill + gates[:, 2:] * local_mean
        return self.fuse(torch.cat([x, gap, boundary, uncertainty, refined], dim=1))


class TopologyConfidenceBoundary(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(4, 2, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        plus = 0.25 * (
            torch.roll(x, 1, -1)
            + torch.roll(x, -1, -1)
            + torch.roll(x, 1, -2)
            + torch.roll(x, -1, -2)
        )
        diagonal = 0.25 * (
            torch.roll(x, (1, 1), (-2, -1))
            + torch.roll(x, (-1, -1), (-2, -1))
            + torch.roll(x, (1, -1), (-2, -1))
            + torch.roll(x, (-1, 1), (-2, -1))
        )
        endpoint = F.relu(x - plus)
        horizontal = 0.5 * (torch.roll(endpoint, 2, -1) + torch.roll(endpoint, -2, -1))
        vertical = 0.5 * (torch.roll(endpoint, 2, -2) + torch.roll(endpoint, -2, -2))
        diagonal_bridge = 0.5 * (
            torch.roll(endpoint, (2, 2), (-2, -1))
            + torch.roll(endpoint, (-2, -2), (-2, -1))
        )
        bridge = horizontal + vertical + diagonal_bridge
        connectivity = torch.sigmoid(plus + diagonal - torch.abs(plus - diagonal))
        gap = F.relu(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x)
        evidence = torch.cat([
            endpoint.amax(dim=1, keepdim=True),
            bridge.amax(dim=1, keepdim=True),
            connectivity.mean(dim=1, keepdim=True),
            gap.mean(dim=1, keepdim=True),
        ], dim=1)
        keep_fill = self.spatial(evidence).softmax(dim=1)
        candidate = x + connectivity * torch.sigmoid(bridge) * gap
        refined = keep_fill[:, :1] * x + keep_fill[:, 1:] * candidate
        return self.fuse(torch.cat([x, endpoint, bridge, connectivity, refined], dim=1))


def make_point1_module(name: str, channels: int) -> nn.Module:
    if name == "mwr":
        return MultiScaleWidthRouter(channels)
    if name == "aca":
        return AxialCurvatureAligner(channels)
    if name == "fap":
        return FrequencyAxialPyramid(channels)
    if name == "sat":
        return SparseAxialTokenConditioner(channels)
    if name == "ads":
        return AnisotropicDynamicShiftConditioner(channels)
    raise ValueError(f"Unsupported TripleStack-v5 point1: {name}")


def make_point2_module(name: str, channels: int) -> nn.Module:
    if name == "eoc":
        return EigenOrientationConsensus(channels)
    if name == "rtc":
        return RidgeTopologyCrossAttention(channels)
    raise ValueError(f"Unsupported TripleStack-v5 point2: {name}")


def make_point3_module(name: str, channels: int) -> nn.Module:
    if name == "upb":
        return UncertaintyPrecisionBridge(channels)
    if name == "tcb":
        return TopologyConfidenceBoundary(channels)
    raise ValueError(f"Unsupported TripleStack-v5 point3: {name}")


class TripleStackV5Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V5_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v5_mode: {mode}. Expected one of {TRIPLE_STACK_V5_MODES}."
            )
        point1_name, point2_name, point3_name = mode.split("_")
        self.mode = mode
        self.point1 = make_point1_module(point1_name, channels)
        self.deg = HoGEdgeGateConv(in_dim=channels, nbins=nbins)
        self.point2 = make_point2_module(point2_name, channels)
        self.point3 = make_point3_module(point3_name, channels)
        self.point1_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.point2_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.point3_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.final_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.final_norm = make_gn(channels)
        self.point1_gamma = nn.Parameter(torch.tensor(0.05))
        self.point2_gamma = nn.Parameter(torch.tensor(0.05))
        self.point3_gamma = nn.Parameter(torch.tensor(0.05))
        self.final_gamma = nn.Parameter(torch.tensor(1.0))

    def _residual_gate(self, x, refined, gate, gamma):
        weight = torch.sigmoid(gate(torch.cat([x, refined], dim=1)))
        return x + gamma * weight * (refined - x)

    def forward(self, x):
        identity = x
        point1 = self._residual_gate(x, self.point1(x), self.point1_gate, self.point1_gamma)
        deg = self.deg(point1)
        point2 = self._residual_gate(deg, self.point2(deg), self.point2_gate, self.point2_gamma)
        point3 = self._residual_gate(point2, self.point3(point2), self.point3_gate, self.point3_gamma)
        point3 = self.final_norm(point3)
        return self._residual_gate(identity, point3, self.final_gate, self.final_gamma)
