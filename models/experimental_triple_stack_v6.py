import torch
import torch.nn as nn

from models.experimental_modules import ConvGNAct, make_gn
from models.experimental_triple_stack_v4 import (
    AdaptiveMultiScaleCrossAxisConditioner,
    AnisotropicDynamicConditioner,
    DirectionalCurvatureTokenMixer,
    HessianEigenOrientationVoting,
    PrecisionGuidedSuppressor,
    TopologyGapCalibrator,
)
from models.experimental_triple_stack_v5 import (
    AnisotropicDynamicShiftConditioner,
    AxialCurvatureAligner,
    EigenOrientationConsensus,
    FrequencyAxialPyramid,
    MultiScaleWidthRouter,
    RidgeTopologyCrossAttention,
    SparseAxialTokenConditioner,
    UncertaintyPrecisionBridge,
)
from models.layers import HoGEdgeGateConv


TRIPLE_STACK_V6_MODES = [
    "hfr_ert_apb",
    "hfr_ert_ugc",
    "hfr_soc_apb",
    "hfr_soc_ugc",
    "wdr_ert_apb",
    "wdr_ert_ugc",
    "wdr_soc_apb",
    "wdr_soc_ugc",
    "cax_ert_apb",
    "cax_ert_ugc",
    "cax_soc_apb",
    "cax_soc_ugc",
    "gdp_ert_apb",
    "gdp_ert_ugc",
    "gdp_soc_apb",
    "gdp_soc_ugc",
    "rst_ert_apb",
    "rst_ert_ugc",
    "rst_soc_apb",
    "rst_soc_ugc",
]


class GatedPairFusion(nn.Module):
    def __init__(self, channels: int, first: nn.Module, second: nn.Module):
        super().__init__()
        self.first = first
        self.second = second
        self.router = nn.Conv2d(channels * 3, 2, kernel_size=1)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        first = self.first(x)
        second = self.second(x)
        weights = self.router(torch.cat([x, first, second], dim=1)).softmax(dim=1)
        mixed = weights[:, :1] * first + weights[:, 1:] * second
        disagreement = torch.abs(first - second)
        return self.fuse(torch.cat([x, mixed, disagreement], dim=1))


class HybridFrequencyShiftRouter(GatedPairFusion):
    def __init__(self, channels: int):
        super().__init__(
            channels,
            FrequencyAxialPyramid(channels),
            AnisotropicDynamicConditioner(channels),
        )


class WidthDirectionalRouter(GatedPairFusion):
    def __init__(self, channels: int):
        super().__init__(
            channels,
            MultiScaleWidthRouter(channels),
            AnisotropicDynamicShiftConditioner(channels),
        )


class CrossAxisScaleConditioner(GatedPairFusion):
    def __init__(self, channels: int):
        super().__init__(
            channels,
            AdaptiveMultiScaleCrossAxisConditioner(channels),
            AxialCurvatureAligner(channels),
        )


class GeodesicDirectionalPropagator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = ConvGNAct(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels)
        self.vertical = ConvGNAct(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels)
        self.diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.anti_diagonal = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.router = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def _propagate(self, features):
        evidence = torch.stack([item.mean(dim=1) for item in features], dim=1)
        weights = self.router(evidence).softmax(dim=1).unsqueeze(2)
        propagated = (torch.stack(features, dim=1) * weights).sum(dim=1)
        confidence = weights.amax(dim=1).expand_as(propagated)
        return propagated, confidence

    def forward(self, x):
        first, confidence = self._propagate([
            self.horizontal(0.5 * (torch.roll(x, 1, -1) + torch.roll(x, -1, -1))),
            self.vertical(0.5 * (torch.roll(x, 1, -2) + torch.roll(x, -1, -2))),
            self.diagonal(0.5 * (torch.roll(x, (1, 1), (-2, -1)) + torch.roll(x, (-1, -1), (-2, -1)))),
            self.anti_diagonal(0.5 * (torch.roll(x, (1, -1), (-2, -1)) + torch.roll(x, (-1, 1), (-2, -1)))),
        ])
        second, _ = self._propagate([
            torch.roll(first, 1, -1),
            torch.roll(first, 1, -2),
            torch.roll(first, (1, 1), (-2, -1)),
            torch.roll(first, (1, -1), (-2, -1)),
        ])
        return self.fuse(torch.cat([x, first, second, confidence], dim=1))


class RidgeScaleTokenConditioner(GatedPairFusion):
    def __init__(self, channels: int):
        super().__init__(
            channels,
            MultiScaleWidthRouter(channels),
            SparseAxialTokenConditioner(channels),
        )


class EigenRidgeTopologyMixer(GatedPairFusion):
    def __init__(self, channels: int):
        super().__init__(
            channels,
            EigenOrientationConsensus(channels),
            RidgeTopologyCrossAttention(channels),
        )


class StructureOrientationCrossAttention(GatedPairFusion):
    def __init__(self, channels: int):
        super().__init__(
            channels,
            HessianEigenOrientationVoting(channels),
            DirectionalCurvatureTokenMixer(channels),
        )


class AdaptivePrecisionBridge(GatedPairFusion):
    def __init__(self, channels: int):
        super().__init__(
            channels,
            UncertaintyPrecisionBridge(channels),
            PrecisionGuidedSuppressor(channels),
        )


class UncertaintyGapCompetition(GatedPairFusion):
    def __init__(self, channels: int):
        super().__init__(
            channels,
            UncertaintyPrecisionBridge(channels),
            TopologyGapCalibrator(channels),
        )


def make_point1_module(name: str, channels: int) -> nn.Module:
    if name == "hfr":
        return HybridFrequencyShiftRouter(channels)
    if name == "wdr":
        return WidthDirectionalRouter(channels)
    if name == "cax":
        return CrossAxisScaleConditioner(channels)
    if name == "gdp":
        return GeodesicDirectionalPropagator(channels)
    if name == "rst":
        return RidgeScaleTokenConditioner(channels)
    raise ValueError(f"Unsupported TripleStack-v6 point1: {name}")


def make_point2_module(name: str, channels: int) -> nn.Module:
    if name == "ert":
        return EigenRidgeTopologyMixer(channels)
    if name == "soc":
        return StructureOrientationCrossAttention(channels)
    raise ValueError(f"Unsupported TripleStack-v6 point2: {name}")


def make_point3_module(name: str, channels: int) -> nn.Module:
    if name == "apb":
        return AdaptivePrecisionBridge(channels)
    if name == "ugc":
        return UncertaintyGapCompetition(channels)
    raise ValueError(f"Unsupported TripleStack-v6 point3: {name}")


class TripleStackV6Block(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in TRIPLE_STACK_V6_MODES:
            raise ValueError(
                f"Unsupported triple_stack_v6_mode: {mode}. Expected one of {TRIPLE_STACK_V6_MODES}."
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
        self.point1_gamma = nn.Parameter(torch.tensor(0.04))
        self.point2_gamma = nn.Parameter(torch.tensor(0.04))
        self.point3_gamma = nn.Parameter(torch.tensor(0.04))
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
