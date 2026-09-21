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

TRIPLE_STACK_V9_MODES = [
    "eub_oeb_prg",
    "eub_oeb_cfr",
    "eub_gtp_prg",
    "eub_gtp_cfr",
    "dcb_oeb_prg",
    "dcb_oeb_cfr",
    "dcb_gtp_prg",
    "dcb_gtp_cfr",
    "mcb_oeb_prg",
    "mcb_oeb_cfr",
    "mcb_gtp_prg",
    "mcb_gtp_cfr",
]


class RecallCalibrationState(NamedTuple):
    features: torch.Tensor
    uncertainty: torch.Tensor
    direction_consistency: torch.Tensor
    bridge_confidence: torch.Tensor


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


def _empty_bridge(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(reference.shape[0], 1, *reference.shape[-2:])


class EntropyUncertaintyBypass(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.trust_head = nn.Conv2d(channels * 3, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, reference: torch.Tensor, refined: torch.Tensor) -> RecallCalibrationState:
        disagreement = (refined - reference).abs()
        trust = torch.sigmoid(self.trust_head(torch.cat([reference, refined, disagreement], dim=1)))
        uncertainty = -(
            trust * torch.log(trust.clamp_min(1e-6))
            + (1.0 - trust) * torch.log((1.0 - trust).clamp_min(1e-6))
        ) / 0.6931471805599453
        refined_weight = trust * (1.0 - 0.5 * uncertainty)
        mixed = refined_weight * refined + (1.0 - refined_weight) * reference
        consistency = torch.exp(-disagreement.mean(dim=1, keepdim=True))
        features = self.fuse(torch.cat([reference, refined, mixed, disagreement], dim=1))
        return RecallCalibrationState(features, uncertainty, consistency, _empty_bridge(reference))


class DisagreementConfidenceBypass(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.confidence_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, reference: torch.Tensor, refined: torch.Tensor) -> RecallCalibrationState:
        disagreement = (refined - reference).abs()
        dot = (reference * refined).mean(dim=1, keepdim=True)
        norm = torch.sqrt(
            reference.pow(2).mean(dim=1, keepdim=True)
            * refined.pow(2).mean(dim=1, keepdim=True)
            + 1e-6
        )
        cosine = (dot / norm).clamp(-1.0, 1.0)
        local_contrast = (
            reference - F.avg_pool2d(reference, kernel_size=3, stride=1, padding=1)
        ).abs().mean(dim=1, keepdim=True)
        cues = torch.cat([
            disagreement.mean(dim=1, keepdim=True),
            0.5 * (cosine + 1.0),
            local_contrast,
        ], dim=1)
        confidence = torch.sigmoid(self.confidence_head(cues))
        uncertainty = 1.0 - confidence
        mixed = confidence * refined + uncertainty * reference
        features = self.fuse(torch.cat([reference, refined, mixed, disagreement], dim=1))
        return RecallCalibrationState(features, uncertainty, confidence, _empty_bridge(reference))


class MultiScaleConfidenceBypass(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale_router = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.trust_head = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, reference: torch.Tensor, refined: torch.Tensor) -> RecallCalibrationState:
        differences = [
            (reference - refined).abs(),
            (
                F.avg_pool2d(reference, kernel_size=3, stride=1, padding=1)
                - F.avg_pool2d(refined, kernel_size=3, stride=1, padding=1)
            ).abs(),
            (
                F.avg_pool2d(reference, kernel_size=7, stride=1, padding=3)
                - F.avg_pool2d(refined, kernel_size=7, stride=1, padding=3)
            ).abs(),
        ]
        cues = torch.cat([
            difference.mean(dim=1, keepdim=True) for difference in differences
        ], dim=1)
        scale_weights = self.scale_router(cues).softmax(dim=1)
        weighted_difference = (
            torch.stack(differences, dim=1) * scale_weights.unsqueeze(2)
        ).sum(dim=1)
        confidence = torch.sigmoid(self.trust_head(cues))
        uncertainty = (1.0 - confidence) * (
            weighted_difference.mean(dim=1, keepdim=True).tanh()
        )
        mixed = (1.0 - uncertainty) * refined + uncertainty * reference
        consistency = torch.exp(-weighted_difference.mean(dim=1, keepdim=True))
        features = self.fuse(torch.cat([
            reference,
            refined,
            mixed,
            weighted_difference,
        ], dim=1))
        return RecallCalibrationState(features, uncertainty, consistency, _empty_bridge(reference))


class OrientedEndpointBridge(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.direction_head = nn.Conv2d(channels, len(DIRECTIONS), kernel_size=1)
        self.bridge_gate = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, state: RecallCalibrationState) -> RecallCalibrationState:
        x = state.features
        neighbors = _directional_neighbors(x)
        opposite = torch.roll(neighbors, shifts=4, dims=1)
        agreement = torch.exp(-(neighbors - opposite).abs().mean(dim=2))
        probabilities = (self.direction_head(x) + agreement).softmax(dim=1)
        weights = probabilities * agreement
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        message = (0.5 * (neighbors + opposite) * weights.unsqueeze(2)).sum(dim=1)
        endpoint_bridge = (
            torch.minimum(F.relu(neighbors), F.relu(opposite)) * weights.unsqueeze(2)
        ).sum(dim=1)
        gap = F.relu(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x)
        directional_consistency = (weights * agreement).sum(dim=1, keepdim=True)
        bridge_confidence = (
            directional_consistency
            * state.direction_consistency
            * (1.0 - state.uncertainty)
        ).clamp(0.0, 1.0)
        cues = torch.cat([
            bridge_confidence,
            state.direction_consistency,
            1.0 - state.uncertainty,
            gap.mean(dim=1, keepdim=True),
        ], dim=1)
        bridge_gate = torch.sigmoid(self.bridge_gate(cues))
        features = self.fuse(torch.cat([
            x,
            message,
            endpoint_bridge * bridge_gate,
            gap * bridge_gate,
        ], dim=1))
        return RecallCalibrationState(
            features,
            state.uncertainty,
            0.5 * (state.direction_consistency + directional_consistency),
            bridge_confidence,
        )


class GeodesicTopologyPropagator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.direction_head = nn.Conv2d(channels, len(DIRECTIONS), kernel_size=1)
        self.propagation_gate = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def _message(self, x: torch.Tensor):
        neighbors = _directional_neighbors(x)
        edge_cost = (neighbors - x.unsqueeze(1)).abs().mean(dim=2)
        probabilities = (self.direction_head(x) - edge_cost).softmax(dim=1)
        message = (neighbors * probabilities.unsqueeze(2)).sum(dim=1)
        consistency = torch.exp(-(probabilities * edge_cost).sum(dim=1, keepdim=True))
        concentration = probabilities.amax(dim=1, keepdim=True)
        return message, consistency, concentration

    def forward(self, state: RecallCalibrationState) -> RecallCalibrationState:
        x = state.features
        first_message, first_consistency, first_concentration = self._message(x)
        cues = torch.cat([
            first_consistency,
            first_concentration,
            1.0 - state.uncertainty,
            (x - first_message).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        propagation_gate = torch.sigmoid(self.propagation_gate(cues))
        first = x + propagation_gate * first_consistency * (first_message - x)
        second_message, second_consistency, second_concentration = self._message(first)
        second = first + propagation_gate * second_consistency * (second_message - first)
        path_consistency = torch.sqrt(
            (first_consistency * second_consistency).clamp_min(1e-6)
        )
        bridge_confidence = (
            path_consistency
            * torch.sqrt((first_concentration * second_concentration).clamp_min(1e-6))
            * (1.0 - state.uncertainty)
        ).clamp(0.0, 1.0)
        features = self.fuse(torch.cat([x, first, second, second_message], dim=1))
        return RecallCalibrationState(
            features,
            state.uncertainty,
            0.5 * (state.direction_consistency + path_consistency),
            bridge_confidence,
        )


class PrecisionRecallGuard(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fill_gate = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.suppress_gate = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, state: RecallCalibrationState, refined: torch.Tensor) -> torch.Tensor:
        local_support = F.max_pool2d(state.features, kernel_size=5, stride=1, padding=2)
        gap = F.relu(local_support - refined)
        local_mean = F.avg_pool2d(refined, kernel_size=3, stride=1, padding=1)
        noise = (refined - local_mean).abs()
        fill_cues = torch.cat([
            state.bridge_confidence,
            state.direction_consistency,
            1.0 - state.uncertainty,
            gap.mean(dim=1, keepdim=True),
        ], dim=1)
        suppress_cues = torch.cat([
            1.0 - state.bridge_confidence,
            1.0 - state.direction_consistency,
            state.uncertainty,
            noise.mean(dim=1, keepdim=True),
        ], dim=1)
        fill = torch.sigmoid(self.fill_gate(fill_cues)) * state.bridge_confidence
        suppress = torch.sigmoid(self.suppress_gate(suppress_cues)) * state.uncertainty
        balanced = refined * (1.0 - suppress) + local_mean * suppress + gap * fill
        return self.fuse(torch.cat([state.features, refined, balanced, gap * fill], dim=1))


class ConfidenceFillRefiner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale_router = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.keep_gate = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, state: RecallCalibrationState, refined: torch.Tensor) -> torch.Tensor:
        gap3 = F.relu(
            F.max_pool2d(state.features, kernel_size=3, stride=1, padding=1) - refined
        )
        gap7 = F.relu(
            F.max_pool2d(state.features, kernel_size=7, stride=1, padding=3) - refined
        )
        cues = torch.cat([
            state.bridge_confidence,
            state.direction_consistency,
            1.0 - state.uncertainty,
            (gap7 - gap3).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        scale_weights = self.scale_router(cues).softmax(dim=1)
        gap = scale_weights[:, :1] * gap3 + scale_weights[:, 1:] * gap7
        keep = torch.sigmoid(self.keep_gate(cues))
        fill_confidence = (
            keep
            * state.bridge_confidence
            * state.direction_consistency
            * (1.0 - state.uncertainty)
        )
        fallback = (1.0 - state.uncertainty) * refined + state.uncertainty * state.features
        calibrated = fallback + fill_confidence * gap
        return self.fuse(torch.cat([
            state.features,
            refined,
            calibrated,
            gap * fill_confidence,
        ], dim=1))


def make_bypass_module(name: str, channels: int) -> nn.Module:
    modules = {
        "eub": EntropyUncertaintyBypass,
        "dcb": DisagreementConfidenceBypass,
        "mcb": MultiScaleConfidenceBypass,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v9 bypass: {name}")
    return modules[name](channels)


def make_bridge_module(name: str, channels: int) -> nn.Module:
    modules = {
        "oeb": OrientedEndpointBridge,
        "gtp": GeodesicTopologyPropagator,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v9 bridge: {name}")
    return modules[name](channels)


def make_guard_module(name: str, channels: int) -> nn.Module:
    modules = {
        "prg": PrecisionRecallGuard,
        "cfr": ConfidenceFillRefiner,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v9 guard: {name}")
    return modules[name](channels)


class TripleStackV9Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V9_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v9_mode: {mode}. Expected one of {TRIPLE_STACK_V9_MODES}."
            )
        bypass_name, bridge_name, guard_name = mode.split("_")
        self.mode = mode
        self.wma = WidthAdaptiveMSA(channels)
        self.deg = HoGEdgeGateConv(in_dim=channels, nbins=nbins)
        self.ctv = CurvatureTopologyVoting(channels)
        self.bypass = make_bypass_module(bypass_name, channels)
        self.bridge = make_bridge_module(bridge_name, channels)
        self.dgb = DualGatedBoundaryBridge(channels)
        self.guard = make_guard_module(guard_name, channels)

        gate_names = ("wma", "ctv", "bypass", "bridge", "dgb", "guard", "final")
        for name in gate_names:
            setattr(self, f"{name}_gate", nn.Conv2d(channels * 2, channels, kernel_size=1))
        self.final_norm = make_gn(channels)
        self.wma_gamma = nn.Parameter(torch.tensor(0.08))
        self.ctv_gamma = nn.Parameter(torch.tensor(0.08))
        self.bypass_gamma = nn.Parameter(torch.tensor(0.04))
        self.bridge_gamma = nn.Parameter(torch.tensor(0.04))
        self.dgb_gamma = nn.Parameter(torch.tensor(0.08))
        self.guard_gamma = nn.Parameter(torch.tensor(0.04))
        self.final_gamma = nn.Parameter(torch.tensor(1.0))

    def _residual_gate(self, x, refined, gate, gamma):
        weight = torch.sigmoid(gate(torch.cat([x, refined], dim=1)))
        return x + gamma * weight * (refined - x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        wma = self._residual_gate(x, self.wma(x), self.wma_gate, self.wma_gamma)
        deg = self.deg(wma)
        ctv = self._residual_gate(deg, self.ctv(deg), self.ctv_gate, self.ctv_gamma)

        state = self.bypass(deg, ctv)
        bypass = self._residual_gate(
            ctv, state.features, self.bypass_gate, self.bypass_gamma
        )
        state = state._replace(features=bypass)
        bridged_state = self.bridge(state)
        bridge = self._residual_gate(
            bypass,
            bridged_state.features,
            self.bridge_gate,
            self.bridge_gamma,
        )
        bridged_state = bridged_state._replace(features=bridge)

        dgb = self._residual_gate(
            bridge,
            self.dgb(bridge),
            self.dgb_gate,
            self.dgb_gamma,
        )
        guarded = self._residual_gate(
            dgb,
            self.guard(bridged_state, dgb),
            self.guard_gate,
            self.guard_gamma,
        )
        guarded = self.final_norm(guarded)
        return self._residual_gate(identity, guarded, self.final_gate, self.final_gamma)
