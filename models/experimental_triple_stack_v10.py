from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.experimental_triple_stack_v7 import (
    CurvatureTopologyVoting,
    DualGatedBoundaryBridge,
    WidthAdaptiveMSA,
)
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

TRIPLE_STACK_V10_MODES = [
    "ard_dbr_bpg",
    "ard_dbr_dmc",
    "ard_pcr_bpg",
    "ard_pcr_dmc",
    "wfd_dbr_bpg",
    "wfd_dbr_dmc",
    "wfd_pcr_bpg",
    "wfd_pcr_dmc",
    "tcd_dbr_bpg",
    "tcd_dbr_dmc",
    "tcd_pcr_bpg",
    "tcd_pcr_dmc",
]


class AnchoredCalibrationState(NamedTuple):
    base_features: torch.Tensor
    weak_evidence: torch.Tensor
    reliability: torch.Tensor
    direction_probabilities: torch.Tensor
    bridge_delta: torch.Tensor
    bridge_confidence: torch.Tensor
    correction_budget: torch.Tensor
    correction: torch.Tensor


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


def _initial_state(
    base_features: torch.Tensor,
    weak_evidence: torch.Tensor,
    reliability: torch.Tensor,
) -> AnchoredCalibrationState:
    batch, _, height, width = base_features.shape
    probabilities = base_features.new_full(
        (batch, len(DIRECTIONS), height, width),
        1.0 / len(DIRECTIONS),
    )
    scalar_zeros = base_features.new_zeros(batch, 1, height, width)
    return AnchoredCalibrationState(
        base_features=base_features,
        weak_evidence=weak_evidence,
        reliability=reliability.clamp(0.0, 1.0),
        direction_probabilities=probabilities,
        bridge_delta=torch.zeros_like(base_features),
        bridge_confidence=scalar_zeros,
        correction_budget=scalar_zeros,
        correction=torch.zeros_like(base_features),
    )


class AgreementReliabilityDecomposer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.reliability_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.weak_fuse = ConvGNAct(
            channels * 4, channels, kernel_size=1, padding=0, act=False
        )

    def forward(self, wma, deg, ctv, base_features):
        disagreement = (deg - ctv).abs()
        dot = (deg * ctv).mean(dim=1, keepdim=True)
        norm = torch.sqrt(
            deg.pow(2).mean(dim=1, keepdim=True)
            * ctv.pow(2).mean(dim=1, keepdim=True)
            + 1e-6
        )
        cosine = (dot / norm).clamp(-1.0, 1.0)
        probability = torch.sigmoid(ctv)
        entropy = -(
            probability * torch.log(probability.clamp_min(1e-6))
            + (1.0 - probability) * torch.log((1.0 - probability).clamp_min(1e-6))
        ).mean(dim=1, keepdim=True) / 0.6931471805599453
        cues = torch.cat([
            disagreement.mean(dim=1, keepdim=True),
            0.5 * (cosine + 1.0),
            entropy,
        ], dim=1)
        reliability = torch.sigmoid(self.reliability_head(cues))
        weak = torch.tanh(self.weak_fuse(torch.cat([
            wma,
            deg,
            ctv,
            disagreement,
        ], dim=1)))
        weak = weak * (1.0 - 0.5 * reliability)
        return _initial_state(base_features, weak, reliability)


class WidthFrequencyDecomposer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.reliability_head = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.weak_fuse = ConvGNAct(
            channels * 5, channels, kernel_size=1, padding=0, act=False
        )

    def forward(self, wma, deg, ctv, base_features):
        low3 = F.avg_pool2d(deg, kernel_size=3, stride=1, padding=1)
        low7 = F.avg_pool2d(deg, kernel_size=7, stride=1, padding=3)
        high3 = deg - low3
        high7 = deg - low7
        width3 = F.relu(F.max_pool2d(wma, kernel_size=3, stride=1, padding=1) - wma)
        width7 = F.relu(F.max_pool2d(wma, kernel_size=7, stride=1, padding=3) - wma)
        cues = torch.cat([
            high3.abs().mean(dim=1, keepdim=True),
            high7.abs().mean(dim=1, keepdim=True),
            width3.mean(dim=1, keepdim=True),
            (deg - ctv).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        reliability = torch.sigmoid(self.reliability_head(cues))
        weak = torch.tanh(self.weak_fuse(torch.cat([
            deg,
            ctv,
            high3,
            high7,
            width7,
        ], dim=1)))
        weak = weak * (0.5 + 0.5 * (1.0 - reliability))
        return _initial_state(base_features, weak, reliability)


class TopologyConsistencyDecomposer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.reliability_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.weak_fuse = ConvGNAct(
            channels * 5, channels, kernel_size=1, padding=0, act=False
        )

    def forward(self, wma, deg, ctv, base_features):
        neighbors = _directional_neighbors(ctv)
        opposite = torch.roll(neighbors, shifts=4, dims=1)
        neighbor_mean = neighbors.mean(dim=1)
        opposite_agreement = torch.exp(
            -(neighbors - opposite).abs().mean(dim=2)
        ).mean(dim=1, keepdim=True)
        endpoint = F.relu(deg - neighbor_mean)
        junction = F.relu(neighbor_mean - ctv)
        path_support = torch.minimum(F.relu(neighbors), F.relu(opposite)).mean(dim=1)
        cues = torch.cat([
            endpoint.mean(dim=1, keepdim=True),
            junction.mean(dim=1, keepdim=True),
            path_support.mean(dim=1, keepdim=True),
            opposite_agreement,
        ], dim=1)
        reliability = torch.sigmoid(self.reliability_head(cues))
        weak = torch.tanh(self.weak_fuse(torch.cat([
            deg,
            ctv,
            endpoint,
            junction,
            path_support,
        ], dim=1)))
        weak = weak * (0.5 + 0.5 * opposite_agreement)
        return _initial_state(base_features, weak, reliability)


class DualPathBridgeRouter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.direction_head = nn.Conv2d(channels * 2, len(DIRECTIONS), kernel_size=1)
        self.route_head = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.confidence_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.delta_fuse = ConvGNAct(
            channels * 4, channels, kernel_size=1, padding=0, act=False
        )

    def forward(self, state: AnchoredCalibrationState) -> AnchoredCalibrationState:
        source = state.base_features + 0.25 * state.weak_evidence
        neighbors = _directional_neighbors(source)
        opposite = torch.roll(neighbors, shifts=4, dims=1)
        agreement = torch.exp(-(neighbors - opposite).abs().mean(dim=2))
        probabilities = (
            self.direction_head(torch.cat([state.base_features, state.weak_evidence], dim=1))
            + agreement
        ).softmax(dim=1)
        endpoint_message = (
            0.5 * (neighbors + opposite) * probabilities.unsqueeze(2)
        ).sum(dim=1)
        first_message = (neighbors * probabilities.unsqueeze(2)).sum(dim=1)
        second_neighbors = _directional_neighbors(first_message)
        second_message = (second_neighbors * probabilities.unsqueeze(2)).sum(dim=1)
        consistency = (agreement * probabilities).sum(dim=1, keepdim=True)
        cues = torch.cat([
            state.reliability,
            consistency,
            state.weak_evidence.abs().mean(dim=1, keepdim=True),
            (second_message - source).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        routes = self.route_head(cues).softmax(dim=1)
        routed = routes[:, :1] * endpoint_message + routes[:, 1:] * second_message
        bridge_delta = torch.tanh(self.delta_fuse(torch.cat([
            source,
            endpoint_message,
            second_message,
            routed - source,
        ], dim=1)))
        bridge_confidence = (
            torch.sigmoid(self.confidence_head(cues))
            * consistency
            * (0.5 + 0.5 * state.reliability)
        ).clamp(0.0, 1.0)
        return state._replace(
            direction_probabilities=probabilities,
            bridge_delta=bridge_delta,
            bridge_confidence=bridge_confidence,
        )


class PathConsensusRouter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.direction_head = nn.Conv2d(channels * 2, len(DIRECTIONS), kernel_size=1)
        self.confidence_head = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.delta_fuse = ConvGNAct(
            channels * 4, channels, kernel_size=1, padding=0, act=False
        )

    def forward(self, state: AnchoredCalibrationState) -> AnchoredCalibrationState:
        source = state.base_features + 0.25 * state.weak_evidence
        neighbors = _directional_neighbors(source)
        opposite = torch.roll(neighbors, shifts=4, dims=1)
        logits = self.direction_head(torch.cat([
            state.base_features,
            state.weak_evidence,
        ], dim=1))
        probabilities = logits.softmax(dim=1)
        opposite_probabilities = torch.roll(probabilities, shifts=4, dims=1)
        bidirectional = torch.minimum(probabilities, opposite_probabilities)
        feature_agreement = torch.exp(-(neighbors - opposite).abs().mean(dim=2))
        consensus = bidirectional * feature_agreement
        consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp_min(1e-6)
        first_message = (
            0.5 * (neighbors + opposite) * consensus.unsqueeze(2)
        ).sum(dim=1)
        second_neighbors = _directional_neighbors(first_message)
        second_message = (second_neighbors * consensus.unsqueeze(2)).sum(dim=1)
        path_consistency = (
            consensus * feature_agreement
        ).sum(dim=1, keepdim=True)
        cues = torch.cat([
            state.reliability,
            path_consistency,
            consensus.amax(dim=1, keepdim=True),
            (second_message - first_message).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        bridge_confidence = (
            torch.sigmoid(self.confidence_head(cues))
            * path_consistency
            * (0.25 + 0.75 * state.reliability)
        ).clamp(0.0, 1.0)
        bridge_delta = torch.tanh(self.delta_fuse(torch.cat([
            source,
            first_message,
            second_message,
            second_message - source,
        ], dim=1)))
        return state._replace(
            direction_probabilities=consensus,
            bridge_delta=bridge_delta,
            bridge_confidence=bridge_confidence,
        )


class BudgetedPrecisionRecallGuard(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.competition_head = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.budget_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.correction_fuse = ConvGNAct(
            channels * 4, channels, kernel_size=1, padding=0, act=False
        )

    def forward(self, state: AnchoredCalibrationState) -> AnchoredCalibrationState:
        base = state.base_features
        local_mean = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(base, kernel_size=5, stride=1, padding=2)
        gap = F.relu(local_max - base)
        noise = base - local_mean
        cues = torch.cat([
            state.bridge_confidence,
            state.reliability,
            state.weak_evidence.abs().mean(dim=1, keepdim=True),
            noise.abs().mean(dim=1, keepdim=True),
        ], dim=1)
        competition = self.competition_head(cues).softmax(dim=1)
        budget = torch.sigmoid(self.budget_head(cues)) * state.bridge_confidence
        fill = torch.tanh(state.bridge_delta + state.weak_evidence + gap)
        suppress = torch.tanh(noise)
        balanced = competition[:, :1] * fill - competition[:, 1:] * suppress
        correction = torch.tanh(self.correction_fuse(torch.cat([
            state.weak_evidence,
            state.bridge_delta,
            balanced,
            gap,
        ], dim=1))) * budget
        return state._replace(correction_budget=budget, correction=correction)


class DistributionMatchedCalibrator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.budget_head = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.correction_fuse = ConvGNAct(
            channels * 4, channels, kernel_size=1, padding=0, act=False
        )

    def forward(self, state: AnchoredCalibrationState) -> AnchoredCalibrationState:
        base = state.base_features
        candidate = state.bridge_delta + state.weak_evidence
        base_mean = F.avg_pool2d(base, kernel_size=5, stride=1, padding=2)
        candidate_mean = F.avg_pool2d(candidate, kernel_size=5, stride=1, padding=2)
        base_variance = F.avg_pool2d(
            (base - base_mean).pow(2), kernel_size=5, stride=1, padding=2
        )
        candidate_variance = F.avg_pool2d(
            (candidate - candidate_mean).pow(2), kernel_size=5, stride=1, padding=2
        )
        matched = (candidate - candidate_mean) * torch.sqrt(
            (base_variance + 1e-6) / (candidate_variance + 1e-6)
        )
        moment_error = (base_variance - candidate_variance).abs().mean(dim=1, keepdim=True)
        cues = torch.cat([
            state.bridge_confidence,
            state.reliability,
            moment_error,
            matched.abs().mean(dim=1, keepdim=True),
        ], dim=1)
        budget = (
            torch.sigmoid(self.budget_head(cues))
            * (0.5 * state.bridge_confidence + 0.5 * state.reliability)
        ).clamp(0.0, 1.0)
        correction = torch.tanh(self.correction_fuse(torch.cat([
            state.weak_evidence,
            state.bridge_delta,
            matched,
            base - base_mean,
        ], dim=1))) * budget
        return state._replace(correction_budget=budget, correction=correction)


def make_decomposer(name: str, channels: int) -> nn.Module:
    modules = {
        "ard": AgreementReliabilityDecomposer,
        "wfd": WidthFrequencyDecomposer,
        "tcd": TopologyConsistencyDecomposer,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v10 decomposer: {name}")
    return modules[name](channels)


def make_router(name: str, channels: int) -> nn.Module:
    modules = {
        "dbr": DualPathBridgeRouter,
        "pcr": PathConsensusRouter,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v10 router: {name}")
    return modules[name](channels)


def make_calibrator(name: str, channels: int) -> nn.Module:
    modules = {
        "bpg": BudgetedPrecisionRecallGuard,
        "dmc": DistributionMatchedCalibrator,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v10 calibrator: {name}")
    return modules[name](channels)


class TripleStackV10Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V10_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v10_mode: {mode}. Expected one of {TRIPLE_STACK_V10_MODES}."
            )
        decomposer_name, router_name, calibrator_name = mode.split("_")
        self.mode = mode

        self.wma = WidthAdaptiveMSA(channels)
        self.deg = HoGEdgeGateConv(in_dim=channels, nbins=nbins)
        self.ctv = CurvatureTopologyVoting(channels)
        self.dgb = DualGatedBoundaryBridge(channels)
        self.wma_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.ctv_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.dgb_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.final_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.final_norm = make_gn(channels)
        self.wma_gamma = nn.Parameter(torch.tensor(0.08))
        self.ctv_gamma = nn.Parameter(torch.tensor(0.08))
        self.dgb_gamma = nn.Parameter(torch.tensor(0.08))
        self.final_gamma = nn.Parameter(torch.tensor(1.0))

        self.decomposer = make_decomposer(decomposer_name, channels)
        self.router = make_router(router_name, channels)
        self.calibrator = make_calibrator(calibrator_name, channels)
        self.sidecar_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sidecar_gamma = nn.Parameter(torch.tensor(0.02))

    @staticmethod
    def _residual_gate(x, refined, gate, gamma):
        weight = torch.sigmoid(gate(torch.cat([x, refined], dim=1)))
        return x + gamma * weight * (refined - x)

    def forward_v7_anchor(self, x: torch.Tensor, return_stages: bool = False):
        identity = x
        wma = self._residual_gate(x, self.wma(x), self.wma_gate, self.wma_gamma)
        deg = self.deg(wma)
        ctv = self._residual_gate(deg, self.ctv(deg), self.ctv_gate, self.ctv_gamma)
        dgb = self._residual_gate(ctv, self.dgb(ctv), self.dgb_gate, self.dgb_gamma)
        anchored = self._residual_gate(
            identity,
            self.final_norm(dgb),
            self.final_gate,
            self.final_gamma,
        )
        if return_stages:
            return anchored, (wma, deg, ctv, dgb)
        return anchored

    def forward_sidecar(self, anchored, stages):
        wma, deg, ctv, _ = stages
        state = self.decomposer(wma, deg, ctv, anchored)
        state = self.router(state)
        state = self.calibrator(state)
        weight = torch.sigmoid(
            self.sidecar_gate(torch.cat([anchored, state.correction], dim=1))
        )
        return anchored + self.sidecar_gamma * weight * state.correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchored, stages = self.forward_v7_anchor(x, return_stages=True)
        return self.forward_sidecar(anchored, stages)
