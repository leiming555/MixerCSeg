import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.layers import HoGEdgeGateConv


DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)

TRIPLE_STACK_V8_MODES = [
    "cwc_lsf_obp",
    "cwc_lsf_trc",
    "cwc_cot_obp",
    "cwc_cot_trc",
    "swa_lsf_obp",
    "swa_lsf_trc",
    "swa_cot_obp",
    "swa_cot_trc",
    "awc_lsf_obp",
    "awc_lsf_trc",
    "awc_cot_obp",
    "awc_cot_trc",
]


class DirectionField(NamedTuple):
    features: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor


def _shift_zero(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    start_y = max(-dy, 0)
    start_x = max(-dx, 0)
    return padded[..., start_y:start_y + height, start_x:start_x + width]


def _directional_neighbors(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([_shift_zero(x, dy, dx) for dy, dx in DIRECTIONS], dim=1)


def _field_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-6))).sum(dim=1, keepdim=True)
    return (1.0 - entropy / math.log(len(DIRECTIONS))).clamp(0.0, 1.0)


class AxialScaleBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        radius = dilation * (kernel_size // 2)
        self.horizontal = ConvGNAct(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, radius),
            dilation=(1, dilation),
            groups=channels,
        )
        self.vertical = ConvGNAct(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(radius, 0),
            dilation=(dilation, 1),
            groups=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.horizontal(x) + self.vertical(x))


class ContinuousWidthConditioner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            AxialScaleBranch(channels, kernel_size=3),
            AxialScaleBranch(channels, kernel_size=5),
            AxialScaleBranch(channels, kernel_size=5, dilation=2),
        ])
        self.width_router = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local3 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        local5 = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        local9 = F.max_pool2d(x, kernel_size=9, stride=1, padding=4)
        cues = torch.cat([
            (local3 - x).abs().mean(dim=1, keepdim=True),
            (local5 - local3).abs().mean(dim=1, keepdim=True),
            (local9 - local5).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        weights = self.width_router(cues).softmax(dim=1)
        branch_features = torch.stack([branch(x) for branch in self.branches], dim=1)
        routed = (branch_features * weights.unsqueeze(2)).sum(dim=1)

        scales = x.new_tensor([0.25, 0.5, 1.0]).view(1, 3, 1, 1)
        expected_width = (weights * scales).sum(dim=1, keepdim=True)
        width_variance = (weights * (scales - expected_width).pow(2)).sum(dim=1, keepdim=True)
        return self.fuse(torch.cat([
            x,
            routed,
            x * expected_width,
            x * width_variance,
        ], dim=1))


class SpectralWidthAligner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            AxialScaleBranch(channels, kernel_size=3),
            AxialScaleBranch(channels, kernel_size=7),
            AxialScaleBranch(channels, kernel_size=5, dilation=2),
        ])
        self.spectral_router = nn.Conv2d(4, 3, kernel_size=3, padding=1)
        self.texture_gate = nn.Conv2d(3, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        low7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        high3 = x - low3
        high7 = x - low7
        spectrum = torch.cat([
            high3.abs().mean(dim=1, keepdim=True),
            high7.abs().mean(dim=1, keepdim=True),
            (high3 - high7).abs().mean(dim=1, keepdim=True),
            low7.abs().mean(dim=1, keepdim=True),
        ], dim=1)
        weights = self.spectral_router(spectrum).softmax(dim=1)
        branch_features = torch.stack([branch(x) for branch in self.branches], dim=1)
        routed = (branch_features * weights.unsqueeze(2)).sum(dim=1)
        texture_keep = torch.sigmoid(self.texture_gate(spectrum[:, :3]))
        aligned_high = texture_keep * high3 + (1.0 - texture_keep) * high7
        return self.fuse(torch.cat([x, routed, aligned_high, low7], dim=1))


class AnisotropicWidthConsensus(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = ConvGNAct(
            channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels
        )
        self.vertical = ConvGNAct(
            channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels
        )
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.anti_diagonal = ConvGNAct(
            channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels
        )
        self.direction_router = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.width_router = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.narrow = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.wide = ConvGNAct(channels, channels, kernel_size=5, padding=4, dilation=2, groups=channels)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        directional = [
            self.horizontal(x),
            self.vertical(x),
            self.diagonal(0.5 * (_shift_zero(x, 1, 1) + _shift_zero(x, -1, -1))),
            self.anti_diagonal(0.5 * (_shift_zero(x, 1, -1) + _shift_zero(x, -1, 1))),
        ]
        direction_cues = torch.cat([
            feature.abs().mean(dim=1, keepdim=True) for feature in directional
        ], dim=1)
        direction_weights = self.direction_router(direction_cues).softmax(dim=1)
        consensus = (
            torch.stack(directional, dim=1) * direction_weights.unsqueeze(2)
        ).sum(dim=1)

        local3 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        local7 = F.max_pool2d(x, kernel_size=7, stride=1, padding=3)
        width_cues = torch.cat([
            (local3 - x).abs().mean(dim=1, keepdim=True),
            (local7 - local3).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        width_weights = self.width_router(width_cues).softmax(dim=1)
        width_mix = width_weights[:, :1] * self.narrow(x) + width_weights[:, 1:] * self.wide(x)
        agreement = direction_weights.amax(dim=1, keepdim=True)
        return self.fuse(torch.cat([x, consensus, width_mix, consensus * agreement], dim=1))


class LearnableSteerableRidgeField(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.directional = nn.Conv2d(
            channels,
            channels * len(DIRECTIONS),
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.field_head = nn.Conv2d(channels, len(DIRECTIONS), kernel_size=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self._init_steerable_kernels(channels)

    def _init_steerable_kernels(self, channels: int) -> None:
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]) / 4.0
        sobel_y = sobel_x.t()
        kernels = []
        for index in range(len(DIRECTIONS)):
            theta = 2.0 * math.pi * index / len(DIRECTIONS)
            kernel = math.cos(theta) * sobel_x + math.sin(theta) * sobel_y
            kernels.append(kernel / kernel.abs().sum().clamp_min(1e-6))
        with torch.no_grad():
            for channel in range(channels):
                for direction, kernel in enumerate(kernels):
                    self.directional.weight[channel * len(DIRECTIONS) + direction, 0].copy_(kernel)

    def forward(self, x: torch.Tensor) -> DirectionField:
        batch, channels, height, width = x.shape
        responses = self.directional(x).view(
            batch, channels, len(DIRECTIONS), height, width
        )
        evidence = responses.mean(dim=1)
        probabilities = (self.field_head(x) + evidence).softmax(dim=1)
        confidence = _field_confidence(probabilities)
        weighted = (responses * probabilities.unsqueeze(1)).sum(dim=2)
        ridge = (responses.abs() * probabilities.unsqueeze(1)).sum(dim=2)
        features = self.fuse(torch.cat([x, weighted, ridge, x * confidence], dim=1))
        return DirectionField(features, probabilities, confidence)


class CurvatureOrientationTokenField(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels // 4)
        self.field_head = nn.Conv2d(channels, len(DIRECTIONS), kernel_size=1)
        self.query = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.key = nn.Linear(channels, hidden, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self.scale = hidden ** -0.5

    def forward(self, x: torch.Tensor) -> DirectionField:
        batch, channels, height, width = x.shape
        neighbors = _directional_neighbors(x)
        opposite = torch.roll(neighbors, shifts=4, dims=1)
        curvature = (neighbors + opposite - 2.0 * x.unsqueeze(1)).abs()
        curvature_evidence = curvature.mean(dim=2)
        initial_probabilities = (
            self.field_head(x) + curvature_evidence
        ).softmax(dim=1)

        flat_x = x.flatten(2)
        flat_probabilities = initial_probabilities.flatten(2)
        normalizer = flat_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        tokens = torch.einsum("bdn,bcn->bdc", flat_probabilities, flat_x) / normalizer
        query = self.query(x).flatten(2).transpose(1, 2)
        attention = torch.matmul(query, self.key(tokens).transpose(1, 2))
        attention = attention.mul(self.scale).softmax(dim=-1)
        context = torch.matmul(attention, self.value(tokens))
        context = context.transpose(1, 2).reshape(batch, channels, height, width)

        token_probabilities = attention.transpose(1, 2).reshape(
            batch, len(DIRECTIONS), height, width
        )
        probabilities = 0.5 * (initial_probabilities + token_probabilities)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
        confidence = _field_confidence(probabilities)
        weighted_curvature = (
            curvature * probabilities.unsqueeze(2)
        ).sum(dim=1)
        features = self.fuse(torch.cat([
            x,
            context,
            weighted_curvature,
            x * confidence,
        ], dim=1))
        return DirectionField(features, probabilities, confidence)


class OrientationBidirectionalPropagator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fill_gate = nn.Conv2d(3, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, field: DirectionField) -> torch.Tensor:
        x = field.features
        forward = _directional_neighbors(x)
        backward = torch.roll(forward, shifts=4, dims=1)
        agreement = torch.exp(-(forward - backward).abs().mean(dim=2))
        weights = field.probabilities * agreement
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        message = (forward * weights.unsqueeze(2)).sum(dim=1)
        bridge = (
            torch.minimum(F.relu(forward), F.relu(backward)) * weights.unsqueeze(2)
        ).sum(dim=1)
        gap = F.relu(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x)
        cues = torch.cat([
            field.confidence,
            (weights * agreement).sum(dim=1, keepdim=True),
            gap.mean(dim=1, keepdim=True),
        ], dim=1)
        fill = torch.sigmoid(self.fill_gate(cues))
        return self.fuse(torch.cat([x, message, bridge * fill, gap * fill], dim=1))


class TopologyReliabilityCalibrator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.propagation_gate = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.suppress_gate = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def _propagate(self, x: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        neighbors = _directional_neighbors(x)
        return (neighbors * probabilities.unsqueeze(2)).sum(dim=1)

    def forward(self, field: DirectionField) -> torch.Tensor:
        x = field.features
        first_message = self._propagate(x, field.probabilities)
        first_consistency = torch.exp(-(x - first_message).abs().mean(dim=1, keepdim=True))
        first_cues = torch.cat([
            field.confidence,
            first_consistency,
            (x - first_message).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        first_gate = torch.sigmoid(self.propagation_gate(first_cues))
        first = x + first_gate * field.confidence * (first_message - x)

        second_message = self._propagate(first, field.probabilities)
        second_consistency = torch.exp(-(first - second_message).abs().mean(dim=1, keepdim=True))
        second = first + first_gate * second_consistency * (second_message - first)
        local_mean = F.avg_pool2d(second, kernel_size=3, stride=1, padding=1)
        isolated = F.relu(second.abs() - local_mean.abs())
        suppress_cues = torch.cat([
            1.0 - field.confidence,
            1.0 - second_consistency,
            isolated.mean(dim=1, keepdim=True),
        ], dim=1)
        suppress = torch.sigmoid(self.suppress_gate(suppress_cues))
        calibrated = second * (1.0 - suppress) + local_mean * suppress
        return self.fuse(torch.cat([x, first, calibrated, isolated], dim=1))


def make_point1_module(name: str, channels: int) -> nn.Module:
    modules = {
        "cwc": ContinuousWidthConditioner,
        "swa": SpectralWidthAligner,
        "awc": AnisotropicWidthConsensus,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v8 point1: {name}")
    return modules[name](channels)


def make_point2_module(name: str, channels: int) -> nn.Module:
    modules = {
        "lsf": LearnableSteerableRidgeField,
        "cot": CurvatureOrientationTokenField,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v8 point2: {name}")
    return modules[name](channels)


def make_point3_module(name: str, channels: int) -> nn.Module:
    modules = {
        "obp": OrientationBidirectionalPropagator,
        "trc": TopologyReliabilityCalibrator,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v8 point3: {name}")
    return modules[name](channels)


class TripleStackV8Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V8_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v8_mode: {mode}. Expected one of {TRIPLE_STACK_V8_MODES}."
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        point1 = self._residual_gate(
            x, self.point1(x), self.point1_gate, self.point1_gamma
        )
        deg = self.deg(point1)
        direction_field = self.point2(deg)
        point2 = self._residual_gate(
            deg,
            direction_field.features,
            self.point2_gate,
            self.point2_gamma,
        )
        direction_field = DirectionField(
            point2,
            direction_field.probabilities,
            direction_field.confidence,
        )
        point3 = self._residual_gate(
            point2,
            self.point3(direction_field),
            self.point3_gate,
            self.point3_gamma,
        )
        point3 = self.final_norm(point3)
        return self._residual_gate(identity, point3, self.final_gate, self.final_gamma)
