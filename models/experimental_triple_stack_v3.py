import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.layers import HoGEdgeGateConv


TRIPLE_STACK_V3_MODES = [
    "dsc_htm_gbc",
    "dsc_htm_prb",
    "dsc_htm_egc",
    "dsc_rtm_gbc",
    "dsc_rtm_prb",
    "dsc_ocv_gbc",
    "dsc_llc_bns",
    "fsc_htm_gbc",
    "fsc_htm_prb",
    "fsc_rtm_gbc",
    "fsc_ocv_egc",
    "fsc_llc_bns",
    "msa_htm_gbc",
    "msa_htm_egc",
    "msa_rtm_prb",
    "msa_ocv_gbc",
    "sta_htm_gbc",
    "sta_rtm_prb",
    "sta_ocv_egc",
    "sta_llc_bns",
    "msa_ocv_prb",
    "msa_ocv_egc",
    "msa_htm_prb",
    "msa_rtm_gbc",
    "msa_rtm_egc",
    "msa_llc_gbc",
    "dsc_ocv_prb",
    "dsc_ocv_egc",
    "dsc_llc_gbc",
    "dsc_rtm_egc",
    "msa_llc_prb",
    "msa_llc_egc",
    "msa_llc_bns",
    "msa_ocv_bns",
    "msa_htm_bns",
    "msa_rtm_bns",
    "dsc_htm_bns",
    "dsc_rtm_bns",
    "dsc_ocv_bns",
    "dsc_llc_prb",
    "dsc_llc_egc",
    "fsc_ocv_gbc",
    "fsc_ocv_prb",
    "fsc_rtm_prb",
    "sta_ocv_gbc",
    "fsc_ocv_bns",
    "sta_ocv_prb",
    "sta_ocv_bns",
    "fsc_htm_bns",
    "fsc_htm_egc",
    "fsc_rtm_bns",
    "fsc_rtm_egc",
    "fsc_llc_gbc",
    "fsc_llc_prb",
    "fsc_llc_egc",
    "sta_htm_prb",
    "sta_htm_bns",
    "sta_htm_egc",
    "sta_rtm_gbc",
    "sta_rtm_bns",
    "id_id_id",
    "msa_id_id",
    "id_ocv_id",
    "id_id_gbc",
    "msa_ocv_id",
    "msa_id_gbc",
    "id_ocv_gbc",
]


def _depthwise_filter(x, kernel, padding=1):
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    return F.conv2d(x, kernel.repeat(x.shape[1], 1, 1, 1), padding=padding, groups=x.shape[1])


class DifferentialKernelsMixin:
    def _init_diff_kernels(self):
        dxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]).view(1, 1, 3, 3)
        dyy = torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        dxy = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) * 0.25
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sy = sx.transpose(-1, -2)
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("dxx", dxx)
        self.register_buffer("dyy", dyy)
        self.register_buffer("dxy", dxy)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)
        self.register_buffer("lap", lap)

    def _hessian_ridge(self, x):
        dxx = _depthwise_filter(x, self.dxx)
        dyy = _depthwise_filter(x, self.dyy)
        dxy = _depthwise_filter(x, self.dxy)
        ridge = torch.sqrt((dxx - dyy).pow(2) + 4.0 * dxy.pow(2) + 1e-6)
        return dxx, dyy, dxy, ridge

    def _sobel(self, x):
        gx = _depthwise_filter(x, self.sx)
        gy = _depthwise_filter(x, self.sy)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        return gx, gy, mag


class DirectionScaleConditioner(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.h3 = ConvGNAct(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels)
        self.v3 = ConvGNAct(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels)
        self.h7 = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.v7 = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 4, kernel_size=1))
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        gx, gy, mag = self._sobel(x)
        direction = torch.sigmoid(torch.abs(gx) + torch.abs(gy) - torch.abs(gx - gy))
        feats = torch.stack([self.h3(x), self.v3(x), self.h7(x), self.v7(x)], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        scale = (feats * weights).sum(dim=1)
        return self.fuse(torch.cat([x, scale, direction * mag], dim=1))


class FrequencyScaleConditioner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 3, kernel_size=1))
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        low3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        low7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        high3 = x - low3
        high7 = x - low7
        pooled = F.interpolate(
            F.adaptive_avg_pool2d(x, (4, 4)),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        mixed = (
            torch.stack([high3, high7, pooled], dim=1) * weights
        ).sum(dim=1)
        return self.fuse(torch.cat([x, high3, high7, mixed], dim=1))


class MultiScaleAxialConditioner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.h = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels)
        self.dh = ConvGNAct(channels, channels, kernel_size=(1, 5), padding=(0, 4), dilation=(1, 2), groups=channels)
        self.dv = ConvGNAct(channels, channels, kernel_size=(5, 1), padding=(4, 0), dilation=(2, 1), groups=channels)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        row = x.mean(dim=-1, keepdim=True).expand_as(x)
        col = x.mean(dim=-2, keepdim=True).expand_as(x)
        axial = self.h(x) + self.v(x) + self.dh(x) + self.dv(x)
        return self.fuse(torch.cat([x, axial, row, col, row * col], dim=1))


class StripTokenAligner(nn.Module):
    def __init__(self, channels: int, num_tokens: int = 4):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, channels) * 0.02)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.h = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        strip = self.h(x) + self.v(x)
        query = strip.mean(dim=(-2, -1))
        attn = torch.softmax(torch.matmul(query.unsqueeze(1), self.key(self.tokens).transpose(1, 2)), dim=-1)
        context = torch.matmul(attn, self.value(self.tokens)).squeeze(1)
        context = context[:, :, None, None].expand_as(x)
        gap = F.max_pool2d(strip, kernel_size=5, stride=1, padding=2) - strip
        return self.fuse(torch.cat([x, strip, context, gap], dim=1))


class HessianTopologyMixer(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.fuse = ConvGNAct(channels * 6, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian_ridge(x)
        _, _, mag = self._sobel(x)
        lap = torch.abs(_depthwise_filter(x, self.lap))
        plus = 0.25 * (
            torch.roll(x, 1, -1) + torch.roll(x, -1, -1) + torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
        )
        diag = 0.25 * (
            torch.roll(x, (1, 1), (-2, -1))
            + torch.roll(x, (-1, -1), (-2, -1))
            + torch.roll(x, (1, -1), (-2, -1))
            + torch.roll(x, (-1, 1), (-2, -1))
        )
        topology = torch.sigmoid(plus + diag - torch.abs(plus - diag)) * ridge
        return self.fuse(torch.cat([x, ridge, mag, lap, plus, topology], dim=1))


class RidgeTokenMixer(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int, num_tokens: int = 6):
        super().__init__()
        self._init_diff_kernels()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, channels) * 0.02)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian_ridge(x)
        gap = torch.clamp(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x, min=0.0)
        query = (x * torch.sigmoid(ridge + gap)).mean(dim=(-2, -1))
        attn = torch.softmax(torch.matmul(query.unsqueeze(1), self.key(self.tokens).transpose(1, 2)), dim=-1)
        context = torch.matmul(attn, self.value(self.tokens)).squeeze(1)
        context = context[:, :, None, None].expand_as(x)
        return self.fuse(torch.cat([x, ridge, gap, context], dim=1))


class OrientationCurvatureVoting(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.h = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        dxx, dyy, dxy, ridge = self._hessian_ridge(x)
        gx, gy, mag = self._sobel(x)
        horizontal = torch.sigmoid(torch.abs(gx) - torch.abs(gy)) * self.h(x)
        vertical = torch.sigmoid(torch.abs(gy) - torch.abs(gx)) * self.v(x)
        curvature = torch.sqrt((dxx + dyy).pow(2) + dxy.pow(2) + 1e-6)
        vote = torch.sigmoid(ridge - curvature) * (horizontal + vertical)
        return self.fuse(torch.cat([x, mag, ridge, curvature, vote], dim=1))


class LaplacianLineConsensus(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.h = ConvGNAct(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian_ridge(x)
        lap = torch.abs(_depthwise_filter(x, self.lap))
        line = self.h(x) + self.v(x)
        consensus = torch.sigmoid(ridge - lap) * line
        suppress = torch.sigmoid(line - lap) * x
        return self.fuse(torch.cat([x, ridge, lap, consensus, suppress], dim=1))


class GapBoundaryConfidence(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(3, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        p = torch.sigmoid(x)
        entropy = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        gap = torch.clamp(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x, min=0.0)
        boundary = torch.abs(x - local)
        conf = torch.sigmoid(self.spatial(torch.cat([
            gap.amax(dim=1, keepdim=True),
            boundary.mean(dim=1, keepdim=True),
            entropy.mean(dim=1, keepdim=True),
        ], dim=1)))
        bridge = x * (1.0 - conf) + gap * conf
        return self.fuse(torch.cat([x, gap, boundary, entropy, bridge], dim=1))


class PrecisionRecallBalancer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_max = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_min = -F.max_pool2d(-x, kernel_size=5, stride=1, padding=2)
        precision = torch.sigmoid(x - mean)
        recall = torch.sigmoid(local_max - x)
        balanced = x * precision + 0.5 * (local_max - local_min) * recall
        return self.fuse(torch.cat([x, precision, recall, local_max - local_min, balanced], dim=1))


class EndpointGapCompletion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 6, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        plus = 0.25 * (
            torch.roll(x, 1, -1) + torch.roll(x, -1, -1) + torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
        )
        endpoint = F.relu(x - plus)
        h_bridge = 0.5 * (torch.roll(endpoint, 2, -1) + torch.roll(endpoint, -2, -1))
        v_bridge = 0.5 * (torch.roll(endpoint, 2, -2) + torch.roll(endpoint, -2, -2))
        diag_bridge = 0.5 * (torch.roll(endpoint, (2, 2), (-2, -1)) + torch.roll(endpoint, (-2, -2), (-2, -1)))
        bridge = torch.sigmoid(h_bridge + v_bridge + diag_bridge) * (h_bridge + v_bridge + diag_bridge)
        return self.fuse(torch.cat([x, endpoint, h_bridge, v_bridge, diag_bridge, bridge], dim=1))


class BoundaryNoiseSuppressor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(3, 1, kernel_size=7, padding=3)
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_min = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        contrast = local_max - local_min
        p = torch.sigmoid(x)
        entropy = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        noise = torch.abs(x - mean) * torch.sigmoid(entropy - contrast)
        keep = torch.sigmoid(self.spatial(torch.cat([
            contrast.amax(dim=1, keepdim=True),
            entropy.mean(dim=1, keepdim=True),
            noise.mean(dim=1, keepdim=True),
        ], dim=1)))
        return self.proj(x * keep + mean * (1.0 - keep))


def make_point1_module(name: str, channels: int) -> nn.Module:
    if name == "id":
        return nn.Identity()
    if name == "dsc":
        return DirectionScaleConditioner(channels)
    if name == "fsc":
        return FrequencyScaleConditioner(channels)
    if name == "msa":
        return MultiScaleAxialConditioner(channels)
    if name == "sta":
        return StripTokenAligner(channels)
    raise ValueError(f"Unsupported TripleStack-v3 point1: {name}")


def make_point2_module(name: str, channels: int) -> nn.Module:
    if name == "id":
        return nn.Identity()
    if name == "htm":
        return HessianTopologyMixer(channels)
    if name == "rtm":
        return RidgeTokenMixer(channels)
    if name == "ocv":
        return OrientationCurvatureVoting(channels)
    if name == "llc":
        return LaplacianLineConsensus(channels)
    raise ValueError(f"Unsupported TripleStack-v3 point2: {name}")


def make_point3_module(name: str, channels: int) -> nn.Module:
    if name == "id":
        return nn.Identity()
    if name == "gbc":
        return GapBoundaryConfidence(channels)
    if name == "prb":
        return PrecisionRecallBalancer(channels)
    if name == "egc":
        return EndpointGapCompletion(channels)
    if name == "bns":
        return BoundaryNoiseSuppressor(channels)
    raise ValueError(f"Unsupported TripleStack-v3 point3: {name}")


class TripleStackV3Block(nn.Module):
    def __init__(
        self,
        channels: int,
        nbins: int,
        mode: str,
    ):
        super().__init__()
        if mode not in TRIPLE_STACK_V3_MODES:
            raise ValueError(f"Unsupported triple_stack_v3_mode: {mode}. Expected one of {TRIPLE_STACK_V3_MODES}.")
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
        p1 = self._residual_gate(x, self.point1(x), self.point1_gate, self.point1_gamma)
        deg = self.deg(p1)
        p2 = self._residual_gate(deg, self.point2(deg), self.point2_gate, self.point2_gamma)
        p3 = self._residual_gate(p2, self.point3(p2), self.point3_gate, self.point3_gamma)
        p3 = self.final_norm(p3)
        return self._residual_gate(identity, p3, self.final_gate, self.final_gamma)
