import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.experimental_triple_stack_v3 import (
    DifferentialKernelsMixin,
    GapBoundaryConfidence,
    MultiScaleAxialConditioner,
    OrientationCurvatureVoting,
    _depthwise_filter,
)
from models.layers import HoGEdgeGateConv


TRIPLE_STACK_V7_SEARCH_MODES = [
    "rma_mev_dgb",
    "rma_mev_pbc",
    "rma_ctv_dgb",
    "rma_ctv_pbc",
    "wma_mev_dgb",
    "wma_mev_pbc",
    "wma_ctv_dgb",
    "wma_ctv_pbc",
    "fma_mev_dgb",
    "fma_mev_pbc",
    "fma_ctv_dgb",
    "fma_ctv_pbc",
    "tma_mev_dgb",
    "tma_mev_pbc",
    "tma_ctv_dgb",
    "tma_ctv_pbc",
    "cma_mev_dgb",
    "cma_mev_pbc",
    "cma_ctv_dgb",
    "cma_ctv_pbc",
]

TRIPLE_STACK_V7_ABLATION_MODES = [
    "wma_ocv_gbc",
    "msa_ctv_gbc",
    "msa_ocv_dgb",
    "wma_ctv_gbc",
    "wma_ocv_dgb",
    "msa_ctv_dgb",
]

TRIPLE_STACK_V7_REMOVAL_MODES = [
    "id_ctv_dgb", "wma_id_dgb", "wma_ctv_id", "id_id_id",
]

TRIPLE_STACK_V7_MODES = (
    TRIPLE_STACK_V7_SEARCH_MODES + TRIPLE_STACK_V7_ABLATION_MODES + TRIPLE_STACK_V7_REMOVAL_MODES
)


class BaseAnchoredRefinement(nn.Module):
    def _init_anchor(self, channels: int):
        self.anchor_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.anchor_gamma = nn.Parameter(torch.tensor(0.05))

    def _anchor(self, base, refined):
        weight = torch.sigmoid(self.anchor_gate(torch.cat([base, refined], dim=1)))
        return base + self.anchor_gamma * weight * (refined - base)


class ReliabilityRoutedMSA(BaseAnchoredRefinement):
    def __init__(self, channels: int):
        super().__init__()
        self.base = MultiScaleAxialConditioner(channels)
        self.short_h = ConvGNAct(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels)
        self.short_v = ConvGNAct(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels)
        self.long_h = ConvGNAct(
            channels, channels, kernel_size=(1, 5), padding=(0, 4), dilation=(1, 2), groups=channels
        )
        self.long_v = ConvGNAct(
            channels, channels, kernel_size=(5, 1), padding=(4, 0), dilation=(2, 1), groups=channels
        )
        self.router = nn.Conv2d(3, 2, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self._init_anchor(channels)

    def forward(self, x):
        base = self.base(x)
        local_mean = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        variance = torch.clamp(
            F.avg_pool2d(base.pow(2), kernel_size=3, stride=1, padding=1) - local_mean.pow(2), min=0.0
        )
        short = self.short_h(base) + self.short_v(base)
        long = self.long_h(base) + self.long_v(base)
        reliability = torch.cat([
            variance.mean(dim=1, keepdim=True),
            torch.abs(base - local_mean).mean(dim=1, keepdim=True),
            torch.abs(short - long).mean(dim=1, keepdim=True),
        ], dim=1)
        weights = self.router(reliability).softmax(dim=1)
        routed = weights[:, :1] * short + weights[:, 1:] * long
        refined = self.fuse(torch.cat([base, routed, variance, base * weights[:, :1]], dim=1))
        return self._anchor(base, refined)


class WidthAdaptiveMSA(BaseAnchoredRefinement):
    def __init__(self, channels: int):
        super().__init__()
        self.base = MultiScaleAxialConditioner(channels)
        self.narrow_h = ConvGNAct(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.narrow_v = ConvGNAct(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        self.medium_h = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.medium_v = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.wide_h = ConvGNAct(
            channels, channels, kernel_size=(1, 5), padding=(0, 4), dilation=(1, 2), groups=channels
        )
        self.wide_v = ConvGNAct(
            channels, channels, kernel_size=(5, 1), padding=(4, 0), dilation=(2, 1), groups=channels
        )
        self.router = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self._init_anchor(channels)

    def forward(self, x):
        base = self.base(x)
        narrow = self.narrow_h(base) + self.narrow_v(base)
        medium = self.medium_h(base) + self.medium_v(base)
        wide = self.wide_h(base) + self.wide_v(base)
        max3 = F.max_pool2d(base, kernel_size=3, stride=1, padding=1)
        max7 = F.max_pool2d(base, kernel_size=7, stride=1, padding=3)
        width_cues = torch.cat([
            torch.clamp(max3 - base, min=0.0).mean(dim=1, keepdim=True),
            torch.clamp(max7 - max3, min=0.0).mean(dim=1, keepdim=True),
            torch.abs(max7 - base).mean(dim=1, keepdim=True),
        ], dim=1)
        weights = self.router(width_cues).softmax(dim=1)
        routed = weights[:, :1] * narrow + weights[:, 1:2] * medium + weights[:, 2:] * wide
        refined = self.fuse(torch.cat([base, routed, max3 - base, max7 - max3], dim=1))
        return self._anchor(base, refined)


class FrequencyCalibratedMSA(BaseAnchoredRefinement):
    def __init__(self, channels: int):
        super().__init__()
        self.base = MultiScaleAxialConditioner(channels)
        self.frequency_gate = nn.Conv2d(3, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self._init_anchor(channels)

    def forward(self, x):
        base = self.base(x)
        low3 = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        low7 = F.avg_pool2d(base, kernel_size=7, stride=1, padding=3)
        high3 = base - low3
        high7 = base - low7
        frequency_cues = torch.cat([
            high3.abs().mean(dim=1, keepdim=True),
            high7.abs().mean(dim=1, keepdim=True),
            torch.abs(high3 - high7).mean(dim=1, keepdim=True),
        ], dim=1)
        keep = torch.sigmoid(self.frequency_gate(frequency_cues))
        calibrated = keep * high3 + (1.0 - keep) * high7
        refined = self.fuse(torch.cat([base, calibrated, low7, base * keep], dim=1))
        return self._anchor(base, refined)


class TopologyTokenMSA(BaseAnchoredRefinement):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels // 4)
        self.base = MultiScaleAxialConditioner(channels)
        self.query = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self.scale = hidden ** -0.5
        self._init_anchor(channels)

    def forward(self, x):
        base = self.base(x)
        tokens = F.adaptive_avg_pool2d(base, output_size=(4, 4))
        query = self.query(base).flatten(2).transpose(1, 2)
        key = self.key(tokens).flatten(2)
        attention = torch.matmul(query, key).mul(self.scale).softmax(dim=-1)
        value = self.value(tokens).flatten(2).transpose(1, 2)
        context = torch.matmul(attention, value).transpose(1, 2).reshape_as(base)
        row = base.mean(dim=-1, keepdim=True).expand_as(base)
        col = base.mean(dim=-2, keepdim=True).expand_as(base)
        refined = self.fuse(torch.cat([base, context, row, col], dim=1))
        return self._anchor(base, refined)


class CrossAxisConsensusMSA(BaseAnchoredRefinement):
    def __init__(self, channels: int):
        super().__init__()
        self.base = MultiScaleAxialConditioner(channels)
        self.horizontal = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.vertical = ConvGNAct(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels)
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.consensus_gate = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self._init_anchor(channels)

    def forward(self, x):
        base = self.base(x)
        horizontal = self.horizontal(base)
        vertical = self.vertical(base)
        diagonal = self.diagonal(0.5 * (
            torch.roll(base, (1, 1), (-2, -1)) + torch.roll(base, (-1, -1), (-2, -1))
        ))
        cues = torch.cat([
            torch.abs(horizontal - vertical).mean(dim=1, keepdim=True),
            torch.abs(horizontal - diagonal).mean(dim=1, keepdim=True),
            torch.abs(vertical - diagonal).mean(dim=1, keepdim=True),
        ], dim=1)
        agreement = torch.sigmoid(self.consensus_gate(cues))
        axial = 0.5 * (horizontal + vertical)
        consensus = agreement * axial + (1.0 - agreement) * diagonal
        refined = self.fuse(torch.cat([base, consensus, axial * diagonal, base * agreement], dim=1))
        return self._anchor(base, refined)


class MultiScaleEigenVoting(BaseAnchoredRefinement, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.base = OrientationCurvatureVoting(channels)
        self.horizontal = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.vertical = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)
        self._init_anchor(channels)

    def _eigen_response(self, x):
        dxx, dyy, dxy, _ = self._hessian_ridge(x)
        delta = torch.sqrt((dxx - dyy).pow(2) + 4.0 * dxy.pow(2) + 1e-6)
        trace = dxx + dyy
        lambda_high = 0.5 * (trace + delta)
        lambda_low = 0.5 * (trace - delta)
        ridge = torch.maximum(lambda_high.abs(), lambda_low.abs())
        horizontal = 0.5 * (1.0 + (dxx - dyy) / (delta + 1e-6))
        vertical = 1.0 - horizontal
        diagonal = (2.0 * dxy / (delta + 1e-6)).abs()
        return ridge, horizontal, vertical, diagonal

    def forward(self, x):
        base = self.base(x)
        ridge, horizontal_weight, vertical_weight, diagonal_weight = self._eigen_response(base)
        coarse = F.avg_pool2d(base, kernel_size=5, stride=1, padding=2)
        coarse_ridge, _, _, _ = self._eigen_response(coarse)
        vote = (
            horizontal_weight * self.horizontal(base)
            + vertical_weight * self.vertical(base)
            + diagonal_weight * self.diagonal(base)
        )
        agreement = torch.exp(-torch.abs(ridge - coarse_ridge))
        refined = self.fuse(torch.cat([base, ridge, coarse_ridge, vote, agreement * base], dim=1))
        return self._anchor(base, refined)


class CurvatureTopologyVoting(BaseAnchoredRefinement, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.base = OrientationCurvatureVoting(channels)
        self.horizontal = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.vertical = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.anti_diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.router = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)
        self._init_anchor(channels)

    def forward(self, x):
        base = self.base(x)
        dxx, dyy, dxy, ridge = self._hessian_ridge(base)
        curvature = torch.sqrt((dxx + dyy).pow(2) + dxy.pow(2) + 1e-6)
        plus = 0.25 * (
            torch.roll(base, 1, -1) + torch.roll(base, -1, -1)
            + torch.roll(base, 1, -2) + torch.roll(base, -1, -2)
        )
        diagonal_context = 0.25 * (
            torch.roll(base, (1, 1), (-2, -1)) + torch.roll(base, (-1, -1), (-2, -1))
            + torch.roll(base, (1, -1), (-2, -1)) + torch.roll(base, (-1, 1), (-2, -1))
        )
        endpoint = F.relu(base - plus)
        junction = F.relu(plus + diagonal_context - base)
        cues = torch.cat([
            ridge.mean(dim=1, keepdim=True),
            curvature.mean(dim=1, keepdim=True),
            endpoint.mean(dim=1, keepdim=True),
            junction.mean(dim=1, keepdim=True),
        ], dim=1)
        weights = self.router(cues).softmax(dim=1)
        directional = (
            weights[:, :1] * self.horizontal(base)
            + weights[:, 1:2] * self.vertical(base)
            + weights[:, 2:3] * self.diagonal(base)
            + weights[:, 3:] * self.anti_diagonal(base)
        )
        topology = torch.sigmoid(plus + diagonal_context - torch.abs(plus - diagonal_context))
        refined = self.fuse(torch.cat([base, ridge, curvature, directional, topology * base], dim=1))
        return self._anchor(base, refined)


class DualGatedBoundaryBridge(BaseAnchoredRefinement):
    def __init__(self, channels: int):
        super().__init__()
        self.base = GapBoundaryConfidence(channels)
        self.fill_gate = nn.Conv2d(3, 1, kernel_size=5, padding=2)
        self.suppress_gate = nn.Conv2d(3, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self._init_anchor(channels)

    def forward(self, x):
        base = self.base(x)
        local_mean = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(base, kernel_size=5, stride=1, padding=2)
        gap = torch.clamp(local_max - base, min=0.0)
        contrast = torch.abs(base - local_mean)
        probability = torch.sigmoid(base)
        entropy = -(
            probability * torch.log(probability + 1e-6)
            + (1.0 - probability) * torch.log(1.0 - probability + 1e-6)
        )
        cues = torch.cat([
            gap.amax(dim=1, keepdim=True),
            contrast.mean(dim=1, keepdim=True),
            entropy.mean(dim=1, keepdim=True),
        ], dim=1)
        fill_weight = torch.sigmoid(self.fill_gate(cues))
        suppress_weight = torch.sigmoid(self.suppress_gate(cues))
        fill = gap * fill_weight
        suppress = base * (1.0 - suppress_weight) + local_mean * suppress_weight
        refined = self.fuse(torch.cat([base, fill, suppress, contrast], dim=1))
        return self._anchor(base, refined)


def _shift_zero(x, dy: int, dx: int):
    height, width = x.shape[-2:]
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    start_y = max(-dy, 0)
    start_x = max(-dx, 0)
    return padded[..., start_y:start_y + height, start_x:start_x + width]


class ProgressiveBridgeCalibrator(BaseAnchoredRefinement):
    def __init__(self, channels: int):
        super().__init__()
        self.base = GapBoundaryConfidence(channels)
        self.bridge_gate = nn.Conv2d(3, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)
        self._init_anchor(channels)

    def forward(self, x):
        base = self.base(x)
        neighbor = 0.25 * (
            _shift_zero(base, 0, 1) + _shift_zero(base, 0, -1)
            + _shift_zero(base, 1, 0) + _shift_zero(base, -1, 0)
        )
        endpoint = F.relu(base - neighbor)
        horizontal = torch.minimum(_shift_zero(endpoint, 0, 2), _shift_zero(endpoint, 0, -2))
        vertical = torch.minimum(_shift_zero(endpoint, 2, 0), _shift_zero(endpoint, -2, 0))
        diagonal = torch.minimum(_shift_zero(endpoint, 2, 2), _shift_zero(endpoint, -2, -2))
        anti_diagonal = torch.minimum(_shift_zero(endpoint, 2, -2), _shift_zero(endpoint, -2, 2))
        bridge = torch.maximum(torch.maximum(horizontal, vertical), torch.maximum(diagonal, anti_diagonal))
        local_mean = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        noise = torch.abs(base - local_mean)
        cues = torch.cat([
            endpoint.mean(dim=1, keepdim=True),
            bridge.amax(dim=1, keepdim=True),
            noise.mean(dim=1, keepdim=True),
        ], dim=1)
        confidence = torch.sigmoid(self.bridge_gate(cues))
        calibrated = base + confidence * bridge - (1.0 - confidence) * noise
        refined = self.fuse(torch.cat([base, endpoint, bridge, calibrated, confidence * base], dim=1))
        return self._anchor(base, refined)


def make_point1_module(name: str, channels: int) -> nn.Module:
    if name == "id":
        return nn.Identity()
    modules = {
        "msa": MultiScaleAxialConditioner,
        "rma": ReliabilityRoutedMSA,
        "wma": WidthAdaptiveMSA,
        "fma": FrequencyCalibratedMSA,
        "tma": TopologyTokenMSA,
        "cma": CrossAxisConsensusMSA,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v7 point1: {name}")
    return modules[name](channels)


def make_point2_module(name: str, channels: int) -> nn.Module:
    if name == "id":
        return nn.Identity()
    modules = {
        "ocv": OrientationCurvatureVoting,
        "mev": MultiScaleEigenVoting,
        "ctv": CurvatureTopologyVoting,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v7 point2: {name}")
    return modules[name](channels)


def make_point3_module(name: str, channels: int) -> nn.Module:
    if name == "id":
        return nn.Identity()
    modules = {
        "gbc": GapBoundaryConfidence,
        "dgb": DualGatedBoundaryBridge,
        "pbc": ProgressiveBridgeCalibrator,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v7 point3: {name}")
    return modules[name](channels)


class TripleStackV7Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V7_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v7_mode: {mode}. Expected one of {TRIPLE_STACK_V7_MODES}."
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
        self.point1_gamma = nn.Parameter(torch.tensor(0.08))
        self.point2_gamma = nn.Parameter(torch.tensor(0.08))
        self.point3_gamma = nn.Parameter(torch.tensor(0.08))
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
