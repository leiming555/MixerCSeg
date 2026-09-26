from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.experimental_triple_stack_v3 import (
    MultiScaleAxialConditioner,
    OrientationCurvatureVoting,
)
from models.layers import HoGEdgeGateConv


TRIPLE_STACK_V12_EVIDENCE = ("ase", "wfe", "tce")
TRIPLE_STACK_V12_ARBITERS = ("cpa", "bca")
TRIPLE_STACK_V12_REFINERS = ("pgg", "bcs")
TRIPLE_STACK_V12_MODES = [
    f"{evidence}_{arbiter}_{refiner}"
    for evidence in TRIPLE_STACK_V12_EVIDENCE
    for arbiter in TRIPLE_STACK_V12_ARBITERS
    for refiner in TRIPLE_STACK_V12_REFINERS
]


PAIR_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class ScaleEvidenceState(NamedTuple):
    features: torch.Tensor
    confidence: torch.Tensor


class ParallelEvidenceState(NamedTuple):
    input_features: torch.Tensor
    deg_features: torch.Tensor
    scale_features: torch.Tensor
    scale_confidence: torch.Tensor
    ocv_features: torch.Tensor
    orientation_confidence: torch.Tensor
    branch_weights: torch.Tensor
    conflict: torch.Tensor
    fused_features: torch.Tensor
    bridge_confidence: torch.Tensor
    refined_features: torch.Tensor


def _shift_zero(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    padded = F.pad(
        x,
        (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)),
    )
    start_y = max(-dy, 0)
    start_x = max(-dx, 0)
    return padded[..., start_y:start_y + height, start_x:start_x + width]


def _paired_messages(x: torch.Tensor, distance: int):
    messages = []
    support = []
    for dy, dx in PAIR_DIRECTIONS:
        forward = _shift_zero(x, distance * dy, distance * dx)
        backward = _shift_zero(x, -distance * dy, -distance * dx)
        messages.append(0.5 * (forward + backward))
        support.append(torch.exp(-(forward - backward).abs().mean(dim=1)))
    return torch.stack(messages, dim=1), torch.stack(support, dim=1)


def _normalized_difference(first: torch.Tensor, second: torch.Tensor):
    numerator = (first - second).abs().mean(dim=1, keepdim=True)
    denominator = (
        first.abs().mean(dim=1, keepdim=True)
        + second.abs().mean(dim=1, keepdim=True)
        + 1e-6
    )
    return (numerator / denominator).clamp(0.0, 1.0)


class AxialScaleEvidence(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.msa = MultiScaleAxialConditioner(channels)
        self.confidence_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> ScaleEvidenceState:
        msa = self.msa(x)
        local_mean = F.avg_pool2d(x, 5, 1, 2)
        row = x.mean(dim=-1, keepdim=True).expand_as(x)
        column = x.mean(dim=-2, keepdim=True).expand_as(x)
        axial_agreement = torch.exp(-(row - column).abs().mean(dim=1, keepdim=True))
        cues = torch.cat([
            (msa - x).abs().mean(dim=1, keepdim=True),
            (x - local_mean).abs().mean(dim=1, keepdim=True),
            axial_agreement,
            (row * column).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        confidence = torch.sigmoid(self.confidence_head(cues)) * axial_agreement
        features = self.fuse(torch.cat([x, msa, row, column], dim=1))
        return ScaleEvidenceState(features, confidence.clamp(0.0, 1.0))


class WidthFrequencyEvidence(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.router = nn.Conv2d(5, 4, kernel_size=3, padding=1)
        self.confidence_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> ScaleEvidenceState:
        low3 = F.avg_pool2d(x, 3, 1, 1)
        low7 = F.avg_pool2d(x, 7, 1, 3)
        high3 = x - low3
        high7 = x - low7
        width3 = F.relu(F.max_pool2d(x, 3, 1, 1) - x)
        width7 = F.relu(F.max_pool2d(x, 7, 1, 3) - x)
        branches = torch.stack([high3, high7, width3, width7], dim=1)
        summaries = branches.abs().mean(dim=2)
        frequency_agreement = torch.exp(
            -(high3 - high7).abs().mean(dim=1, keepdim=True)
        )
        weights = self.router(torch.cat([summaries, frequency_agreement], dim=1)).softmax(dim=1)
        mixed = (branches * weights.unsqueeze(2)).sum(dim=1)
        cues = torch.cat([
            frequency_agreement,
            weights.amax(dim=1, keepdim=True),
            (width3 - width7).abs().mean(dim=1, keepdim=True),
            mixed.abs().mean(dim=1, keepdim=True),
        ], dim=1)
        confidence = torch.sigmoid(self.confidence_head(cues)) * frequency_agreement
        features = self.fuse(torch.cat([x, high3, high7, width7, mixed], dim=1))
        return ScaleEvidenceState(features, confidence.clamp(0.0, 1.0))


class TextureConsistencyEvidence(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = ConvGNAct(
            channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels
        )
        self.vertical = ConvGNAct(
            channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels
        )
        self.confidence_head = nn.Conv2d(4, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor) -> ScaleEvidenceState:
        local3 = F.avg_pool2d(x, 3, 1, 1)
        local7 = F.avg_pool2d(x, 7, 1, 3)
        texture = (x - local3).abs()
        coarse_texture = (x - local7).abs()
        axial = self.horizontal(x) + self.vertical(x)
        line_consistency = torch.exp(
            -(texture - coarse_texture).abs().mean(dim=1, keepdim=True)
        )
        noise = (texture - axial.abs()).abs()
        cues = torch.cat([
            line_consistency,
            texture.mean(dim=1, keepdim=True),
            noise.mean(dim=1, keepdim=True),
            axial.abs().mean(dim=1, keepdim=True),
        ], dim=1)
        confidence = torch.sigmoid(self.confidence_head(cues)) * line_consistency
        features = self.fuse(torch.cat([x, axial, local3, local7, axial - texture], dim=1))
        return ScaleEvidenceState(features, confidence.clamp(0.0, 1.0))


class ConflictAwarePriorArbitration(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale_projection = ConvGNAct(
            channels, channels, kernel_size=1, padding=0, act=False
        )
        self.router = nn.Conv2d(5, 3, kernel_size=3, padding=1)
        self.orientation_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(
        self,
        input_features: torch.Tensor,
        deg_features: torch.Tensor,
        scale_state: ScaleEvidenceState,
        ocv_features: torch.Tensor,
    ) -> ParallelEvidenceState:
        scale = self.scale_projection(scale_state.features)
        deg_ocv = _normalized_difference(deg_features, ocv_features)
        deg_scale = _normalized_difference(deg_features, scale)
        ocv_scale = _normalized_difference(ocv_features, scale)
        conflict = ((deg_ocv + deg_scale + ocv_scale) / 3.0).clamp(0.0, 1.0)
        cues = torch.cat([
            scale_state.confidence,
            1.0 - conflict,
            deg_ocv,
            deg_scale,
            ocv_scale,
        ], dim=1)
        logits = self.router(cues)
        baseline_bias = torch.cat([conflict, -0.5 * conflict, -0.5 * conflict], dim=1)
        weights = (logits + baseline_bias).softmax(dim=1)
        weighted = (
            weights[:, :1] * deg_features
            + weights[:, 1:2] * ocv_features
            + weights[:, 2:] * scale
        )
        orientation_confidence = (
            torch.sigmoid(self.orientation_head(torch.cat([
                scale_state.confidence,
                1.0 - deg_ocv,
                1.0 - ocv_scale,
                weights[:, 1:2],
            ], dim=1)))
            * (1.0 - 0.5 * conflict)
        ).clamp(0.0, 1.0)
        fused = self.fuse(torch.cat([deg_features, ocv_features, scale, weighted], dim=1))
        zeros = scale_state.confidence.new_zeros(scale_state.confidence.shape)
        return ParallelEvidenceState(
            input_features,
            deg_features,
            scale,
            scale_state.confidence,
            ocv_features,
            orientation_confidence,
            weights,
            conflict,
            fused,
            zeros,
            fused,
        )


class BaselineConfidenceArbitration(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale_projection = ConvGNAct(
            channels, channels, kernel_size=1, padding=0, act=False
        )
        self.router = nn.Conv2d(5, 3, kernel_size=3, padding=1)
        self.orientation_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(
        self,
        input_features: torch.Tensor,
        deg_features: torch.Tensor,
        scale_state: ScaleEvidenceState,
        ocv_features: torch.Tensor,
    ) -> ParallelEvidenceState:
        scale = self.scale_projection(scale_state.features)
        deg_centered = deg_features - F.avg_pool2d(deg_features, 3, 1, 1)
        ocv_centered = ocv_features - F.avg_pool2d(ocv_features, 3, 1, 1)
        dot = (deg_centered * ocv_centered).mean(dim=1, keepdim=True)
        norm = torch.sqrt(
            deg_centered.pow(2).mean(dim=1, keepdim=True)
            * ocv_centered.pow(2).mean(dim=1, keepdim=True)
            + 1e-6
        )
        alignment = (0.5 * ((dot / norm).clamp(-1.0, 1.0) + 1.0)).clamp(0.0, 1.0)
        deg_ocv = _normalized_difference(deg_features, ocv_features)
        deg_scale = _normalized_difference(deg_features, scale)
        conflict = (0.5 * deg_ocv + 0.5 * (1.0 - alignment)).clamp(0.0, 1.0)
        cues = torch.cat([
            scale_state.confidence,
            alignment,
            1.0 - deg_ocv,
            1.0 - deg_scale,
            conflict,
        ], dim=1)
        logits = self.router(cues)
        baseline_bias = torch.cat([
            1.5 * conflict,
            alignment - conflict,
            scale_state.confidence - conflict,
        ], dim=1)
        weights = (logits + baseline_bias).softmax(dim=1)
        weighted = (
            weights[:, :1] * deg_features
            + weights[:, 1:2] * ocv_features
            + weights[:, 2:] * scale
        )
        orientation_confidence = (
            torch.sigmoid(self.orientation_head(torch.cat([
                alignment,
                1.0 - deg_ocv,
                weights[:, 1:2],
                scale_state.confidence,
            ], dim=1)))
            * (0.5 + 0.5 * alignment)
        ).clamp(0.0, 1.0)
        fused = self.fuse(torch.cat([deg_features, ocv_features, scale, weighted], dim=1))
        zeros = scale_state.confidence.new_zeros(scale_state.confidence.shape)
        return ParallelEvidenceState(
            input_features,
            deg_features,
            scale,
            scale_state.confidence,
            ocv_features,
            orientation_confidence,
            weights,
            conflict,
            fused,
            zeros,
            fused,
        )


class PrecisionGuidedGapGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.competition_head = nn.Conv2d(6, 2, kernel_size=3, padding=1)
        self.budget_head = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.fill_fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)
        self.delta_fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, state: ParallelEvidenceState) -> ParallelEvidenceState:
        messages, support = _paired_messages(state.fused_features, distance=2)
        route = support / support.sum(dim=1, keepdim=True).clamp_min(1e-6)
        bridge = (messages * route.unsqueeze(2)).sum(dim=1)
        path_support = (support * route).sum(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(state.fused_features, 3, 1, 1)
        local_max = F.max_pool2d(state.fused_features, 5, 1, 2)
        gap = F.relu(local_max - state.fused_features)
        noise = state.fused_features - local_mean
        probability = torch.sigmoid(state.fused_features)
        entropy = -(
            probability * torch.log(probability.clamp_min(1e-6))
            + (1.0 - probability) * torch.log((1.0 - probability).clamp_min(1e-6))
        ).mean(dim=1, keepdim=True) / 0.6931471805599453
        cues = torch.cat([
            state.scale_confidence,
            state.orientation_confidence,
            1.0 - state.conflict,
            path_support,
            gap.mean(dim=1, keepdim=True),
            entropy,
        ], dim=1)
        competition = self.competition_head(cues).softmax(dim=1)
        budget = (
            torch.sigmoid(self.budget_head(cues))
            * path_support
            * state.orientation_confidence
            * (1.0 - state.conflict)
        ).clamp(0.0, 1.0)
        fill = self.fill_fuse(torch.cat([
            state.fused_features,
            bridge,
            gap,
            state.ocv_features,
        ], dim=1))
        balanced = competition[:, :1] * fill - competition[:, 1:] * noise
        delta = torch.tanh(self.delta_fuse(torch.cat([
            balanced,
            state.fused_features,
            state.deg_features,
            local_mean,
        ], dim=1)))
        refined = state.fused_features + 0.35 * budget * delta
        return state._replace(bridge_confidence=budget, refined_features=refined)


class BilateralConsensusSuppressor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.confidence_head = nn.Conv2d(6, 1, kernel_size=5, padding=2)
        self.keep_head = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, state: ParallelEvidenceState) -> ParallelEvidenceState:
        messages1, support1 = _paired_messages(state.fused_features, distance=1)
        messages2, support2 = _paired_messages(state.fused_features, distance=2)
        bilateral_support = torch.minimum(support1, support2)
        route = bilateral_support / bilateral_support.sum(dim=1, keepdim=True).clamp_min(1e-6)
        bridge1 = (messages1 * route.unsqueeze(2)).sum(dim=1)
        bridge2 = (messages2 * route.unsqueeze(2)).sum(dim=1)
        bridge = 0.5 * (bridge1 + bridge2)
        path_support = (bilateral_support * route).sum(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(state.fused_features, 3, 1, 1)
        isolated = (state.fused_features - bridge1).abs()
        cues = torch.cat([
            state.scale_confidence,
            state.orientation_confidence,
            1.0 - state.conflict,
            path_support,
            isolated.mean(dim=1, keepdim=True),
            (state.fused_features - local_mean).abs().mean(dim=1, keepdim=True),
        ], dim=1)
        confidence = (
            torch.sigmoid(self.confidence_head(cues))
            * path_support
            * state.orientation_confidence
            * (1.0 - state.conflict)
        ).clamp(0.0, 1.0)
        keep = torch.sigmoid(self.keep_head(cues))
        candidate = self.fuse(torch.cat([
            state.fused_features,
            bridge1,
            bridge2,
            state.deg_features,
            state.ocv_features,
        ], dim=1))
        conservative = keep * candidate + (1.0 - keep) * state.deg_features
        refined = state.fused_features + 0.35 * confidence * (
            conservative - state.fused_features
        )
        return state._replace(bridge_confidence=confidence, refined_features=refined)


def make_evidence(name: str, channels: int) -> nn.Module:
    modules = {
        "ase": AxialScaleEvidence,
        "wfe": WidthFrequencyEvidence,
        "tce": TextureConsistencyEvidence,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v12 evidence: {name}")
    return modules[name](channels)


def make_arbiter(name: str, channels: int) -> nn.Module:
    modules = {
        "cpa": ConflictAwarePriorArbitration,
        "bca": BaselineConfidenceArbitration,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v12 arbiter: {name}")
    return modules[name](channels)


def make_refiner(name: str, channels: int) -> nn.Module:
    modules = {
        "pgg": PrecisionGuidedGapGate,
        "bcs": BilateralConsensusSuppressor,
    }
    if name not in modules:
        raise ValueError(f"Unsupported TripleStack-v12 refiner: {name}")
    return modules[name](channels)


class TripleStackV12Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V12_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v12_mode: {mode}. "
                f"Expected one of {TRIPLE_STACK_V12_MODES}."
            )
        evidence_name, arbiter_name, refiner_name = mode.split("_")
        self.mode = mode
        self.evidence = make_evidence(evidence_name, channels)
        self.deg = HoGEdgeGateConv(in_dim=channels, nbins=nbins)
        self.ocv = OrientationCurvatureVoting(channels)
        self.arbiter = make_arbiter(arbiter_name, channels)
        self.refiner = make_refiner(refiner_name, channels)
        self.final_norm = make_gn(channels)
        self.final_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.final_gamma = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor, return_state: bool = False):
        scale_state = self.evidence(x)
        deg = self.deg(x)
        ocv = self.ocv(deg)
        state = self.arbiter(x, deg, scale_state, ocv)
        state = self.refiner(state)
        candidate = self.final_norm(state.refined_features)
        gate = torch.sigmoid(self.final_gate(torch.cat([deg, candidate], dim=1)))
        output = deg + self.final_gamma * gate * (candidate - deg)
        if return_state:
            return output, state
        return output
