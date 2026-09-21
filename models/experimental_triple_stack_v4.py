import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.layers import HoGEdgeGateConv


TRIPLE_STACK_V4_MODES = [
    "amc_eov_tgc",
    "amc_eov_pgs",
    "amc_dct_tgc",
    "amc_dct_pgs",
    "spc_eov_tgc",
    "spc_eov_pgs",
    "spc_dct_tgc",
    "spc_dct_pgs",
    "adc_eov_tgc",
    "adc_eov_pgs",
    "adc_dct_tgc",
    "adc_dct_pgs",
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
        lap = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]).view(1, 1, 3, 3)
        self.register_buffer("dxx", dxx)
        self.register_buffer("dyy", dyy)
        self.register_buffer("dxy", dxy)
        self.register_buffer("lap", lap)

    def _hessian(self, x):
        dxx = _depthwise_filter(x, self.dxx)
        dyy = _depthwise_filter(x, self.dyy)
        dxy = _depthwise_filter(x, self.dxy)
        return dxx, dyy, dxy


class AdaptiveMultiScaleCrossAxisConditioner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.h3 = ConvGNAct(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.v3 = ConvGNAct(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        self.h7 = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.v7 = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.hd = ConvGNAct(
            channels,
            channels,
            kernel_size=(1, 5),
            padding=(0, 4),
            dilation=(1, 2),
            groups=channels,
        )
        self.vd = ConvGNAct(
            channels,
            channels,
            kernel_size=(5, 1),
            padding=(4, 0),
            dilation=(2, 1),
            groups=channels,
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 3, kernel_size=1),
        )
        self.cross_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        branches = torch.stack([
            self.h3(x) + self.v3(x),
            self.h7(x) + self.v7(x),
            self.hd(x) + self.vd(x),
        ], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        scale = (branches * weights).sum(dim=1)
        row = x.mean(dim=-1, keepdim=True).expand_as(x)
        col = x.mean(dim=-2, keepdim=True).expand_as(x)
        cross = torch.sigmoid(self.cross_gate(torch.cat([row, col], dim=1)))
        cross = cross * (row + col)
        return self.fuse(torch.cat([x, scale, cross], dim=1))


class SpectralPyramidConditioner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 3, kernel_size=1),
        )
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        low3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        low7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        low11 = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
        bands = torch.stack([x - low3, low3 - low7, low7 - low11], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        spectral = (bands * weights).sum(dim=1)
        contrast = bands.abs().mean(dim=1)
        return self.fuse(torch.cat([x, spectral, contrast, low11], dim=1))


class AnisotropicDynamicConditioner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = ConvGNAct(
            channels,
            channels,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=channels,
        )
        self.vertical = ConvGNAct(
            channels,
            channels,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=channels,
        )
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.dilated = ConvGNAct(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 4, kernel_size=1),
        )
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        shifted_h = 0.5 * (torch.roll(x, 1, -1) + torch.roll(x, -1, -1))
        shifted_v = 0.5 * (torch.roll(x, 1, -2) + torch.roll(x, -1, -2))
        branches = torch.stack([
            self.horizontal(x),
            self.vertical(x),
            self.diagonal(x),
            self.dilated(x),
        ], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        mixed = (branches * weights).sum(dim=1)
        return self.fuse(torch.cat([x, mixed, shifted_h, shifted_v], dim=1))


class HessianEigenOrientationVoting(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.horizontal = ConvGNAct(
            channels,
            channels,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=channels,
        )
        self.vertical = ConvGNAct(
            channels,
            channels,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=channels,
        )
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.anti_diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        dxx, dyy, dxy = self._hessian(x)
        trace = dxx + dyy
        discriminant = torch.sqrt((dxx - dyy).pow(2) + 4.0 * dxy.pow(2) + 1e-6)
        eigen_high = 0.5 * (trace + discriminant)
        eigen_low = 0.5 * (trace - discriminant)
        ridge = torch.abs(eigen_high) - torch.abs(eigen_low)
        ridge = torch.abs(ridge)
        cos2 = (dxx - dyy) / (discriminant + 1e-6)
        sin2 = (2.0 * dxy) / (discriminant + 1e-6)
        orientation_weights = torch.stack([cos2, -cos2, sin2, -sin2], dim=1).softmax(dim=1)
        branches = torch.stack([
            self.horizontal(x),
            self.vertical(x),
            self.diagonal(x),
            self.anti_diagonal(x),
        ], dim=1)
        oriented = (branches * orientation_weights).sum(dim=1)
        vesselness = torch.sigmoid(ridge - torch.abs(trace))
        return self.fuse(torch.cat([x, ridge, vesselness, oriented, torch.abs(trace)], dim=1))


class DirectionalCurvatureTokenMixer(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.horizontal = ConvGNAct(
            channels,
            channels,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=channels,
        )
        self.vertical = ConvGNAct(
            channels,
            channels,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=channels,
        )
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.anti_diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
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
        tokens = branches.mean(dim=(-2, -1))
        normalized = self.token_norm(tokens)
        tokens, _ = self.token_attn(normalized, normalized, normalized, need_weights=False)
        weights = self.token_score(tokens).softmax(dim=1).unsqueeze(-1).unsqueeze(-1)
        mixed = (branches * weights).sum(dim=1)
        context = tokens.mean(dim=1)[:, :, None, None].expand_as(x)
        curvature = torch.abs(_depthwise_filter(x, self.lap))
        return self.fuse(torch.cat([x, mixed, context, curvature], dim=1))


class TopologyGapCalibrator(nn.Module):
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
        endpoint = F.relu(x - plus)
        horizontal = 0.5 * (torch.roll(endpoint, 2, -1) + torch.roll(endpoint, -2, -1))
        vertical = 0.5 * (torch.roll(endpoint, 2, -2) + torch.roll(endpoint, -2, -2))
        diagonal = 0.25 * (
            torch.roll(endpoint, (2, 2), (-2, -1))
            + torch.roll(endpoint, (-2, -2), (-2, -1))
            + torch.roll(endpoint, (2, -2), (-2, -1))
            + torch.roll(endpoint, (-2, 2), (-2, -1))
        )
        bridge = horizontal + vertical + diagonal
        gap = F.relu(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x)
        p = torch.sigmoid(x)
        uncertainty = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        evidence = torch.cat([
            endpoint.amax(dim=1, keepdim=True),
            bridge.amax(dim=1, keepdim=True),
            gap.mean(dim=1, keepdim=True),
            uncertainty.mean(dim=1, keepdim=True),
        ], dim=1)
        keep_fill = self.spatial(evidence).softmax(dim=1)
        candidate = x + torch.sigmoid(bridge) * gap
        refined = keep_fill[:, :1] * x + keep_fill[:, 1:] * candidate
        return self.fuse(torch.cat([x, endpoint, bridge, uncertainty, refined], dim=1))


class PrecisionGuidedSuppressor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(4, 3, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_max = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_min = -F.max_pool2d(-x, kernel_size=5, stride=1, padding=2)
        contrast = local_max - local_min
        p = torch.sigmoid(x)
        uncertainty = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        positive = F.relu(x - mean)
        evidence = torch.cat([
            contrast.amax(dim=1, keepdim=True),
            uncertainty.mean(dim=1, keepdim=True),
            positive.mean(dim=1, keepdim=True),
            torch.abs(x - mean).mean(dim=1, keepdim=True),
        ], dim=1)
        gates = self.spatial(evidence).softmax(dim=1)
        refined = gates[:, :1] * x + gates[:, 1:2] * local_max + gates[:, 2:] * mean
        return self.fuse(torch.cat([x, contrast, uncertainty, positive, refined], dim=1))


def make_point1_module(name: str, channels: int) -> nn.Module:
    if name == "amc":
        return AdaptiveMultiScaleCrossAxisConditioner(channels)
    if name == "spc":
        return SpectralPyramidConditioner(channels)
    if name == "adc":
        return AnisotropicDynamicConditioner(channels)
    raise ValueError(f"Unsupported TripleStack-v4 point1: {name}")


def make_point2_module(name: str, channels: int) -> nn.Module:
    if name == "eov":
        return HessianEigenOrientationVoting(channels)
    if name == "dct":
        return DirectionalCurvatureTokenMixer(channels)
    raise ValueError(f"Unsupported TripleStack-v4 point2: {name}")


def make_point3_module(name: str, channels: int) -> nn.Module:
    if name == "tgc":
        return TopologyGapCalibrator(channels)
    if name == "pgs":
        return PrecisionGuidedSuppressor(channels)
    raise ValueError(f"Unsupported TripleStack-v4 point3: {name}")


class TripleStackV4Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V4_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v4_mode: {mode}. Expected one of {TRIPLE_STACK_V4_MODES}."
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
