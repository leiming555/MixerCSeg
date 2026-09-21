from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.experimental_triple_stack_v3 import (
    DifferentialKernelsMixin,
    MultiScaleAxialConditioner,
)
from models.layers import HoGEdgeGateConv


TRIPLE_STACK_V11_CONDITIONERS = ("ram", "scm", "fpm", "lwm")
TRIPLE_STACK_V11_VOTERS = ("rcv", "pcv")
TRIPLE_STACK_V11_REFINERS = ("cgb", "pbs")
TRIPLE_STACK_V11_MODES = [
    f"{conditioner}_{voter}_{refiner}"
    for conditioner in TRIPLE_STACK_V11_CONDITIONERS
    for voter in TRIPLE_STACK_V11_VOTERS
    for refiner in TRIPLE_STACK_V11_REFINERS
]


PAIR_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class ScaleConditionState(NamedTuple):
    features: torch.Tensor
    reliability: torch.Tensor


class CoupledPriorState(NamedTuple):
    conditioned_features: torch.Tensor
    scale_reliability: torch.Tensor
    deg_features: torch.Tensor
    voted_features: torch.Tensor
    orientation_weights: torch.Tensor
    orientation_reliability: torch.Tensor
    ridge_strength: torch.Tensor
    bridge_confidence: torch.Tensor
    refined_features: torch.Tensor


def _shift_zero(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    padded = F.pad(
        x,
        (
            max(dx, 0),
            max(-dx, 0),
            max(dy, 0),
            max(-dy, 0),
        ),
    )
    start_y = max(-dy, 0)
    start_x = max(-dx, 0)
    return padded[..., start_y:start_y + height, start_x:start_x + width]


def _paired_messages(x: torch.Tensor, distance: int = 1):
    messages = []
    support = []
    for dy, dx in PAIR_DIRECTIONS:
        forward = _shift_zero(x, distance * dy, distance * dx)
        backward = _shift_zero(x, -distance * dy, -distance * dx)
        messages.append(0.5 * (forward + backward))
        support.append(torch.exp(-(forward - backward).abs().mean(dim=1)))
    return torch.stack(messages, dim=1), torch.stack(support, dim=1)


class ReliabilityAwareMSA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.msa = MultiScaleAxialConditioner(channels)
        self.reliability_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> ScaleConditionState:
        msa = self.msa(x)
        local_mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_contrast = (x - local_mean).abs()
        row = x.mean(dim=-1, keepdim=True).expand_as(x)
        column = x.mean(dim=-2, keepdim=True).expand_as(x)
        axial_agreement = torch.exp(-(row - column).abs().mean(dim=1, keepdim=True))
        cues = torch.cat([
            (msa - x).abs().mean(dim=1, keepdim=True),
            local_contrast.mean(dim=1, keepdim=True),
            (row - column).abs().mean(dim=1, keepdim=True),
            axial_agreement,
        ], dim=1)
        reliability = torch.sigmoid(self.reliability_head(cues)) * axial_agreement
        features = self.fuse(torch.cat([x, msa, local_mean, x - local_mean], dim=1))
        return ScaleConditionState(features, reliability.clamp(0.0, 1.0))


class ScaleConsensusMSA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernels = (3, 7, 11)
        self.horizontal = nn.ModuleList([
            ConvGNAct(
                channels,
                channels,
                kernel_size=(1, kernel),
                padding=(0, kernel // 2),
                groups=channels,
            )
            for kernel in kernels
        ])
        self.vertical = nn.ModuleList([
            ConvGNAct(
                channels,
                channels,
                kernel_size=(kernel, 1),
                padding=(kernel // 2, 0),
                groups=channels,
            )
            for kernel in kernels
        ])
        self.router = nn.Conv2d(4, 3, kernel_size=3, padding=1)
        self.reliability_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> ScaleConditionState:
        branches = torch.stack([
            horizontal(x) + vertical(x)
            for horizontal, vertical in zip(self.horizontal, self.vertical)
        ], dim=1)
        summaries = branches.abs().mean(dim=2)
        contrast = (x - F.avg_pool2d(x, 3, 1, 1)).abs().mean(dim=1, keepdim=True)
        weights = self.router(torch.cat([summaries, contrast], dim=1)).softmax(dim=1)
        mixed = (branches * weights.unsqueeze(2)).sum(dim=1)
        mean = branches.mean(dim=1)
        dispersion = (branches - mixed.unsqueeze(1)).pow(2).mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        agreement = torch.exp(-dispersion)
        cues = torch.cat([weights.amax(dim=1, keepdim=True), dispersion, contrast], dim=1)
        reliability = torch.sigmoid(self.reliability_head(cues)) * agreement
        features = self.fuse(torch.cat([x, mixed, mean, mixed - mean], dim=1))
        return ScaleConditionState(features, reliability.clamp(0.0, 1.0))


class FrequencyPreservedMSA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = ConvGNAct(
            channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels
        )
        self.vertical = ConvGNAct(
            channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels
        )
        self.router = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.reliability_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> ScaleConditionState:
        low3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        low7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        low11 = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
        high3, high7, high11 = x - low3, x - low7, x - low11
        axial = self.horizontal(x) + self.vertical(x)
        branches = torch.stack([high3, high7, high11, axial], dim=1)
        energies = branches.abs().mean(dim=2)
        weights = self.router(energies).softmax(dim=1)
        mixed = (branches * weights.unsqueeze(2)).sum(dim=1)
        frequency_agreement = torch.exp(
            -(high3 - high11).abs().mean(dim=1, keepdim=True)
        )
        texture = (high3 - high7).abs().mean(dim=1, keepdim=True)
        cues = torch.cat([
            frequency_agreement,
            texture,
            axial.abs().mean(dim=1, keepdim=True),
        ], dim=1)
        reliability = torch.sigmoid(self.reliability_head(cues)) * frequency_agreement
        features = self.fuse(torch.cat([x, mixed, low7, axial], dim=1))
        return ScaleConditionState(features, reliability.clamp(0.0, 1.0))


class LocalWidthMSA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernels = (5, 9, 13)
        self.horizontal = nn.ModuleList([
            ConvGNAct(
                channels,
                channels,
                kernel_size=(1, kernel),
                padding=(0, kernel // 2),
                groups=channels,
            )
            for kernel in kernels
        ])
        self.vertical = nn.ModuleList([
            ConvGNAct(
                channels,
                channels,
                kernel_size=(kernel, 1),
                padding=(kernel // 2, 0),
                groups=channels,
            )
            for kernel in kernels
        ])
        self.router = nn.Conv2d(4, 3, kernel_size=3, padding=1)
        self.reliability_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> ScaleConditionState:
        gaps = torch.stack([
            F.relu(F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2) - x)
            for kernel in (3, 5, 7)
        ], dim=1)
        branches = torch.stack([
            horizontal(x) + vertical(x)
            for horizontal, vertical in zip(self.horizontal, self.vertical)
        ], dim=1)
        gap_summaries = gaps.mean(dim=2)
        local_contrast = (x - F.avg_pool2d(x, 5, 1, 2)).abs().mean(dim=1, keepdim=True)
        weights = self.router(torch.cat([gap_summaries, local_contrast], dim=1)).softmax(dim=1)
        mixed = (branches * weights.unsqueeze(2)).sum(dim=1)
        width = (gaps * weights.unsqueeze(2)).sum(dim=1)
        dispersion = (branches - mixed.unsqueeze(1)).abs().mean(dim=(1, 2)).unsqueeze(1)
        agreement = torch.exp(-dispersion)
        cues = torch.cat([
            weights.amax(dim=1, keepdim=True),
            agreement,
            width.mean(dim=1, keepdim=True),
        ], dim=1)
        reliability = torch.sigmoid(self.reliability_head(cues)) * agreement
        features = self.fuse(torch.cat([x, mixed, width, F.avg_pool2d(x, 5, 1, 2)], dim=1))
        return ScaleConditionState(features, reliability.clamp(0.0, 1.0))


class ReliabilityCalibratedVoting(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.horizontal = ConvGNAct(
            channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels
        )
        self.vertical = ConvGNAct(
            channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels
        )
        self.orientation_head = nn.Conv2d(5, 4, kernel_size=3, padding=1)
        self.reliability_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(
        self,
        conditioned_features: torch.Tensor,
        scale_reliability: torch.Tensor,
        deg_features: torch.Tensor,
    ) -> CoupledPriorState:
        dxx, dyy, dxy, ridge = self._hessian_ridge(deg_features)
        gx, gy, magnitude = self._sobel(deg_features)
        pair_messages, _ = _paired_messages(deg_features)
        branches = torch.stack([
            self.horizontal(deg_features),
            self.vertical(deg_features),
            pair_messages[:, 2],
            pair_messages[:, 3],
        ], dim=1)
        cues = torch.cat([
            gx.abs().mean(dim=1, keepdim=True),
            gy.abs().mean(dim=1, keepdim=True),
            dxy.abs().mean(dim=1, keepdim=True),
            (dxx - dyy).abs().mean(dim=1, keepdim=True),
            scale_reliability,
        ], dim=1)
        weights = self.orientation_head(cues).softmax(dim=1)
        vote = (branches * weights.unsqueeze(2)).sum(dim=1)
        dispersion = (
            (branches - vote.unsqueeze(1)).abs() * weights.unsqueeze(2)
        ).sum(dim=1).mean(dim=1, keepdim=True)
        curvature = torch.sqrt((dxx + dyy).pow(2) + dxy.pow(2) + 1e-6)
        agreement = torch.exp(-dispersion)
        ridge_strength = torch.sigmoid(ridge.mean(dim=1, keepdim=True))
        reliability_cues = torch.cat([
            scale_reliability,
            ridge_strength,
            torch.exp(-curvature.mean(dim=1, keepdim=True)),
            agreement,
        ], dim=1)
        orientation_reliability = (
            torch.sigmoid(self.reliability_head(reliability_cues))
            * agreement
            * (0.25 + 0.75 * scale_reliability)
        ).clamp(0.0, 1.0)
        voted = self.fuse(torch.cat([
            deg_features,
            vote,
            ridge,
            curvature,
            magnitude,
        ], dim=1))
        zeros = scale_reliability.new_zeros(scale_reliability.shape)
        return CoupledPriorState(
            conditioned_features,
            scale_reliability,
            deg_features,
            voted,
            weights,
            orientation_reliability,
            ridge_strength,
            zeros,
            voted,
        )


class PairwiseCurvatureVoting(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.orientation_head = nn.Conv2d(6, 4, kernel_size=3, padding=1)
        self.reliability_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(
        self,
        conditioned_features: torch.Tensor,
        scale_reliability: torch.Tensor,
        deg_features: torch.Tensor,
    ) -> CoupledPriorState:
        dxx, dyy, dxy, ridge = self._hessian_ridge(deg_features)
        _, _, magnitude = self._sobel(deg_features)
        messages, pair_support = _paired_messages(deg_features)
        ridge_strength = torch.sigmoid(ridge.mean(dim=1, keepdim=True))
        logits = self.orientation_head(torch.cat([
            pair_support,
            scale_reliability,
            ridge_strength,
        ], dim=1)) + torch.log(pair_support.clamp_min(1e-6))
        weights = logits.softmax(dim=1)
        vote = (messages * weights.unsqueeze(2)).sum(dim=1)
        consensus = (pair_support * weights).sum(dim=1, keepdim=True)
        curvature = torch.sqrt((dxx + dyy).pow(2) + dxy.pow(2) + 1e-6)
        reliability_cues = torch.cat([
            scale_reliability,
            ridge_strength,
            consensus,
            torch.exp(-(vote - deg_features).abs().mean(dim=1, keepdim=True)),
        ], dim=1)
        orientation_reliability = (
            torch.sigmoid(self.reliability_head(reliability_cues))
            * consensus
            * (0.25 + 0.75 * scale_reliability)
        ).clamp(0.0, 1.0)
        voted = self.fuse(torch.cat([
            deg_features,
            vote,
            ridge,
            curvature,
            magnitude,
        ], dim=1))
        zeros = scale_reliability.new_zeros(scale_reliability.shape)
        return CoupledPriorState(
            conditioned_features,
            scale_reliability,
            deg_features,
            voted,
            weights,
            orientation_reliability,
            ridge_strength,
            zeros,
            voted,
        )


class ConsensusGuidedBoundaryBridge(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.confidence_head = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, state: CoupledPriorState) -> CoupledPriorState:
        messages, support = _paired_messages(state.voted_features, distance=2)
        routed_support = state.orientation_weights * support
        routed_support = routed_support / routed_support.sum(dim=1, keepdim=True).clamp_min(1e-6)
        bridge = (messages * routed_support.unsqueeze(2)).sum(dim=1)
        local_mean = F.avg_pool2d(state.voted_features, 3, 1, 1)
        local_max = F.max_pool2d(state.voted_features, 5, 1, 2)
        gap = F.relu(local_max - state.voted_features)
        boundary = (state.voted_features - local_mean).abs()
        path_support = (support * state.orientation_weights).sum(dim=1, keepdim=True)
        cues = torch.cat([
            state.scale_reliability,
            state.orientation_reliability,
            state.ridge_strength,
            path_support,
            gap.mean(dim=1, keepdim=True),
        ], dim=1)
        confidence = (
            torch.sigmoid(self.confidence_head(cues))
            * state.orientation_reliability
            * (0.5 + 0.5 * state.scale_reliability)
        ).clamp(0.0, 1.0)
        candidate = self.fuse(torch.cat([
            state.voted_features,
            bridge,
            gap,
            boundary,
            state.deg_features,
        ], dim=1))
        refined = state.voted_features + confidence * (candidate - state.voted_features)
        return state._replace(bridge_confidence=confidence, refined_features=refined)


class PrecisionBudgetSuppressor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.competition_head = nn.Conv2d(6, 2, kernel_size=3, padding=1)
        self.budget_head = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.fill_fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self.delta_fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, state: CoupledPriorState) -> CoupledPriorState:
        messages, support = _paired_messages(state.voted_features, distance=2)
        routed_support = state.orientation_weights * support
        routed_support = routed_support / routed_support.sum(dim=1, keepdim=True).clamp_min(1e-6)
        bridge = (messages * routed_support.unsqueeze(2)).sum(dim=1)
        local_mean = F.avg_pool2d(state.voted_features, 3, 1, 1)
        local_max = F.max_pool2d(state.voted_features, 5, 1, 2)
        gap = F.relu(local_max - state.voted_features)
        noise = state.voted_features - local_mean
        path_support = (support * state.orientation_weights).sum(dim=1, keepdim=True)
        cues = torch.cat([
            state.scale_reliability,
            state.orientation_reliability,
            state.ridge_strength,
            path_support,
            gap.mean(dim=1, keepdim=True),
            noise.abs().mean(dim=1, keepdim=True),
        ], dim=1)
        competition = self.competition_head(cues).softmax(dim=1)
        budget = (
            torch.sigmoid(self.budget_head(cues))
            * state.orientation_reliability
            * (0.25 + 0.75 * state.scale_reliability)
        ).clamp(0.0, 1.0)
        fill = self.fill_fuse(torch.cat([
            state.voted_features,
            bridge,
            gap,
            state.deg_features,
        ], dim=1))
        balanced = competition[:, :1] * fill + competition[:, 1:] * local_mean
        delta = torch.tanh(self.delta_fuse(torch.cat([
            balanced,
            state.voted_features,
            bridge,
            noise,
        ], dim=1)))
        refined = state.voted_features + 0.5 * budget * delta
        return state._replace(bridge_confidence=budget, refined_features=refined)


def make_conditioner(name: str, channels: int) -> nn.Module:
    modules = {
        "ram": ReliabilityAwareMSA,
        "scm": ScaleConsensusMSA,
        "fpm": FrequencyPreservedMSA,
        "lwm": LocalWidthMSA,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v11 conditioner: {name}")
    return modules[name](channels)


def make_voter(name: str, channels: int) -> nn.Module:
    modules = {
        "rcv": ReliabilityCalibratedVoting,
        "pcv": PairwiseCurvatureVoting,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v11 voter: {name}")
    return modules[name](channels)


def make_refiner(name: str, channels: int) -> nn.Module:
    modules = {
        "cgb": ConsensusGuidedBoundaryBridge,
        "pbs": PrecisionBudgetSuppressor,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v11 refiner: {name}")
    return modules[name](channels)


class TripleStackV11Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V11_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v11_mode: {mode}. "
                f"Expected one of {TRIPLE_STACK_V11_MODES}."
            )
        conditioner_name, voter_name, refiner_name = mode.split("_")
        self.mode = mode
        self.conditioner = make_conditioner(conditioner_name, channels)
        self.deg = HoGEdgeGateConv(in_dim=channels, nbins=nbins)
        self.voter = make_voter(voter_name, channels)
        self.refiner = make_refiner(refiner_name, channels)
        self.point1_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.point2_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.point3_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.final_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.final_norm = make_gn(channels)
        self.point1_gamma = nn.Parameter(torch.tensor(0.06))
        self.point2_gamma = nn.Parameter(torch.tensor(0.06))
        self.point3_gamma = nn.Parameter(torch.tensor(0.04))
        self.final_gamma = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _residual_gate(x, refined, gate, gamma):
        weight = torch.sigmoid(gate(torch.cat([x, refined], dim=1)))
        return x + gamma * weight * (refined - x)

    def forward(self, x: torch.Tensor, return_state: bool = False):
        identity = x
        scale_state = self.conditioner(x)
        conditioned = self._residual_gate(
            x,
            scale_state.features,
            self.point1_gate,
            self.point1_gamma,
        )
        deg = self.deg(conditioned)
        state = self.voter(conditioned, scale_state.reliability, deg)
        voted = self._residual_gate(
            deg,
            state.voted_features,
            self.point2_gate,
            self.point2_gamma,
        )
        state = state._replace(voted_features=voted, refined_features=voted)
        state = self.refiner(state)
        refined = self._residual_gate(
            voted,
            state.refined_features,
            self.point3_gate,
            self.point3_gamma,
        )
        output = self._residual_gate(
            identity,
            self.final_norm(refined),
            self.final_gate,
            self.final_gamma,
        )
        if return_state:
            return output, state
        return output
