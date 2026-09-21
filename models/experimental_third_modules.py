import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn


THIRD_EXPERIMENTAL_MODULE_MODES = [
    "hessian_eigen_bridge_gate",
    "vesselness_scale_router",
    "ridge_orientation_strip_attn",
    "curvature_consistency_gate",
    "endpoint_gap_completion",
    "crack_width_adaptive_gate",
    "topology_crossing_context",
    "sparse_ridge_token_memory",
    "local_phase_quadrature_gate",
    "anisotropic_diffusion_refine",
    "centerline_confidence_calibrator",
    "ridge_gap_dilation_bank",
    "orthogonal_noise_suppressor",
    "multiscale_laplacian_fusion",
    "directional_second_order_mixer",
    "hessian_sobel_coupled_gate",
    "entropy_uncertainty_refiner",
    "strip_transformer_cross_gate",
    "pyramid_ridge_context_router",
    "frequency_highpass_residual_gate",
]


def _depthwise_filter(x, kernel):
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    return F.conv2d(x, kernel.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])


class HessianFilterMixin:
    def _init_hessian_kernels(self):
        dxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]).view(1, 1, 3, 3)
        dyy = torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        dxy = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) * 0.25
        self.register_buffer("dxx", dxx)
        self.register_buffer("dyy", dyy)
        self.register_buffer("dxy", dxy)

    def _hessian(self, x):
        dxx = _depthwise_filter(x, self.dxx)
        dyy = _depthwise_filter(x, self.dyy)
        dxy = _depthwise_filter(x, self.dxy)
        ridge = torch.sqrt((dxx - dyy).pow(2) + 4.0 * dxy.pow(2) + 1e-6)
        return dxx, dyy, dxy, ridge


class SobelFilterMixin:
    def _init_sobel_kernels(self):
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sy = sx.transpose(-1, -2)
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)
        self.register_buffer("lap", lap)

    def _sobel(self, x):
        gx = _depthwise_filter(x, self.sx)
        gy = _depthwise_filter(x, self.sy)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return gx, gy, mag


class HessianEigenBridgeGateBranch(nn.Module, HessianFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_hessian_kernels()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        dxx, dyy, dxy, ridge = self._hessian(x)
        trace = dxx + dyy
        bridge = torch.sigmoid(ridge - torch.abs(trace)) * x
        return self.fuse(torch.cat([x, ridge, torch.abs(dxy), bridge], dim=1))


class VesselnessScaleRouterBranch(nn.Module, HessianFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_hessian_kernels()
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 3, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        ridges = []
        for kernel_size in (3, 5, 7):
            smooth = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            _, _, _, ridge = self._hessian(smooth)
            ridges.append(ridge)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        fused = (torch.stack(ridges, dim=1) * weights).sum(dim=1)
        return self.proj(fused)


class RidgeOrientationStripAttnBranch(nn.Module, SobelFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_sobel_kernels()
        self.h = ConvGNAct(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 4, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        gx, gy, _ = self._sobel(x)
        diag_a = 0.5 * (torch.roll(x, shifts=(1, 1), dims=(-2, -1)) + torch.roll(x, shifts=(-1, -1), dims=(-2, -1)))
        diag_b = 0.5 * (torch.roll(x, shifts=(1, -1), dims=(-2, -1)) + torch.roll(x, shifts=(-1, 1), dims=(-2, -1)))
        feats = torch.stack([self.h(x), self.v(x), diag_a * torch.sigmoid(gx), diag_b * torch.sigmoid(gy)], dim=1)
        weights = self.router(torch.abs(gx) + torch.abs(gy)).softmax(dim=1).unsqueeze(2)
        return self.proj((feats * weights).sum(dim=1))


class CurvatureConsistencyGateBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        d2x = torch.roll(x, 1, -1) - 2.0 * x + torch.roll(x, -1, -1)
        d2y = torch.roll(x, 1, -2) - 2.0 * x + torch.roll(x, -1, -2)
        curvature = torch.sqrt(d2x * d2x + d2y * d2y + 1e-6)
        consistency = torch.sigmoid(-(torch.abs(d2x - d2y)))
        return self.fuse(torch.cat([x, curvature, consistency, x * consistency], dim=1))


class EndpointGapCompletionBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        cross = 0.25 * (
            torch.roll(x, 1, -1) + torch.roll(x, -1, -1) + torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
        )
        endpoints = F.relu(x - cross)
        gap_h = 0.5 * (torch.roll(endpoints, 2, -1) + torch.roll(endpoints, -2, -1))
        gap_v = 0.5 * (torch.roll(endpoints, 2, -2) + torch.roll(endpoints, -2, -2))
        return self.fuse(torch.cat([x, endpoints, gap_h, gap_v], dim=1))


class CrackWidthAdaptiveGateBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 3, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        bands = []
        for kernel_size in (3, 5, 7):
            local_max = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            bands.append((local_max - local_mean) * torch.sigmoid(x - local_mean))
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((torch.stack(bands, dim=1) * weights).sum(dim=1))


class TopologyCrossingContextBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        plus = 0.25 * (
            torch.roll(x, 1, -1) + torch.roll(x, -1, -1) + torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
        )
        diag = 0.25 * (
            torch.roll(x, (1, 1), (-2, -1))
            + torch.roll(x, (-1, -1), (-2, -1))
            + torch.roll(x, (1, -1), (-2, -1))
            + torch.roll(x, (-1, 1), (-2, -1))
        )
        crossing = torch.abs(plus - diag)
        return self.fuse(torch.cat([x, plus, diag, crossing], dim=1))


class SparseRidgeTokenMemoryBranch(nn.Module):
    def __init__(self, channels: int, num_tokens: int = 6):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, channels) * 0.02)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.proj = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        score = torch.sigmoid(x.mean(dim=1, keepdim=True))
        query = (x * score).mean(dim=(-2, -1))
        attn = torch.softmax(torch.matmul(query.unsqueeze(1), self.key(self.tokens).transpose(1, 2)), dim=-1)
        context = torch.matmul(attn, self.value(self.tokens)).squeeze(1)
        context = context[:, :, None, None].expand_as(x)
        return self.proj(torch.cat([x, context], dim=1))


class LocalPhaseQuadratureGateBranch(nn.Module, SobelFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_sobel_kernels()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        gx, gy, odd = self._sobel(x)
        even = torch.abs(_depthwise_filter(x, self.lap))
        phase_energy = torch.sqrt(odd * odd + even * even + 1e-6)
        gate = torch.sigmoid(phase_energy - torch.abs(gx - gy))
        return self.fuse(torch.cat([x * gate, phase_energy, gate], dim=1))


class AnisotropicDiffusionRefineBranch(nn.Module, SobelFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_sobel_kernels()
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        gx, gy, _ = self._sobel(x)
        h_smooth = 0.5 * (torch.roll(x, 1, -1) + torch.roll(x, -1, -1))
        v_smooth = 0.5 * (torch.roll(x, 1, -2) + torch.roll(x, -1, -2))
        gate_h = torch.sigmoid(-torch.abs(gx))
        gate_v = torch.sigmoid(-torch.abs(gy))
        refined = x + 0.5 * gate_h * (h_smooth - x) + 0.5 * gate_v * (v_smooth - x)
        return self.proj(refined)


class CenterlineConfidenceCalibratorBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        local_mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_max = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_min = -F.max_pool2d(-x, kernel_size=5, stride=1, padding=2)
        width = local_max - local_min
        confidence = torch.sigmoid((x - local_mean) * width)
        return self.fuse(torch.cat([x, confidence, width, x * confidence], dim=1))


class RidgeGapDilationBankBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        d3 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        d5 = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        d7 = F.max_pool2d(x, kernel_size=7, stride=1, padding=3)
        gap = torch.clamp(d5 - x, min=0.0)
        return self.fuse(torch.cat([x, d3 - x, d7 - d3, gap], dim=1))


class OrthogonalNoiseSuppressorBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.h = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        h = self.h(x)
        v = self.v(x)
        orthogonal_noise = torch.abs(h - v)
        keep = torch.sigmoid(h + v - orthogonal_noise)
        return self.fuse(torch.cat([x * keep, h, v], dim=1))


class MultiscaleLaplacianFusionBranch(nn.Module, SobelFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_sobel_kernels()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        laps = []
        for kernel_size in (1, 3, 5):
            smooth = x if kernel_size == 1 else F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            laps.append(torch.abs(_depthwise_filter(smooth, self.lap)))
        return self.fuse(torch.cat([x] + laps, dim=1))


class DirectionalSecondOrderMixerBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        d_h = torch.roll(x, 1, -1) - 2.0 * x + torch.roll(x, -1, -1)
        d_v = torch.roll(x, 1, -2) - 2.0 * x + torch.roll(x, -1, -2)
        d_a = torch.roll(x, (1, 1), (-2, -1)) - 2.0 * x + torch.roll(x, (-1, -1), (-2, -1))
        return self.fuse(torch.cat([x, d_h, d_v, d_a], dim=1))


class HessianSobelCoupledGateBranch(nn.Module, HessianFilterMixin, SobelFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_hessian_kernels()
        self._init_sobel_kernels()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian(x)
        _, _, sobel = self._sobel(x)
        gate = torch.sigmoid(ridge * sobel)
        return self.fuse(torch.cat([x * gate, ridge, sobel, gate], dim=1))


class EntropyUncertaintyRefinerBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        p = torch.sigmoid(x)
        entropy = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        pooled = torch.cat([entropy.mean(dim=1, keepdim=True), entropy.amax(dim=1, keepdim=True)], dim=1)
        uncertainty = torch.sigmoid(self.spatial(pooled))
        refined = x * (1.0 - uncertainty) + (x - entropy) * uncertainty
        return self.proj(refined)


class StripTransformerCrossGateBranch(nn.Module):
    def __init__(self, channels: int, pool_size: int = 4):
        super().__init__()
        num_heads = 1
        for heads in (4, 2, 1):
            if channels % heads == 0:
                num_heads = heads
                break
        self.pool_size = pool_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = make_gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.strip = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.proj = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(self.norm(x), (self.pool_size, self.pool_size))
        qkv = self.qkv(pooled).reshape(b, 3, self.num_heads, self.head_dim, self.pool_size * self.pool_size)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        context = attn @ v
        context = context.permute(0, 1, 3, 2).reshape(b, c, self.pool_size, self.pool_size)
        context = F.interpolate(context, size=(h, w), mode="bilinear", align_corners=False)
        return self.proj(torch.cat([self.strip(x), context], dim=1))


class PyramidRidgeContextRouterBranch(nn.Module, HessianFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_hessian_kernels()
        self.d1 = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.d2 = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.d4 = ConvGNAct(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels)
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 3, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian(x)
        feats = torch.stack([self.d1(ridge), self.d2(ridge), self.d4(ridge)], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((feats * weights).sum(dim=1))


class FrequencyHighpassResidualGateBranch(nn.Module, SobelFilterMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_sobel_kernels()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        low3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        low7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        high = (x - low3) + (low3 - low7)
        lap = torch.abs(_depthwise_filter(x, self.lap))
        gate = torch.sigmoid(high + lap)
        return self.fuse(torch.cat([x, high, lap, gate], dim=1))


def make_third_branch(mode: str, channels: int) -> nn.Module:
    branches = {
        "hessian_eigen_bridge_gate": HessianEigenBridgeGateBranch,
        "vesselness_scale_router": VesselnessScaleRouterBranch,
        "ridge_orientation_strip_attn": RidgeOrientationStripAttnBranch,
        "curvature_consistency_gate": CurvatureConsistencyGateBranch,
        "endpoint_gap_completion": EndpointGapCompletionBranch,
        "crack_width_adaptive_gate": CrackWidthAdaptiveGateBranch,
        "topology_crossing_context": TopologyCrossingContextBranch,
        "sparse_ridge_token_memory": SparseRidgeTokenMemoryBranch,
        "local_phase_quadrature_gate": LocalPhaseQuadratureGateBranch,
        "anisotropic_diffusion_refine": AnisotropicDiffusionRefineBranch,
        "centerline_confidence_calibrator": CenterlineConfidenceCalibratorBranch,
        "ridge_gap_dilation_bank": RidgeGapDilationBankBranch,
        "orthogonal_noise_suppressor": OrthogonalNoiseSuppressorBranch,
        "multiscale_laplacian_fusion": MultiscaleLaplacianFusionBranch,
        "directional_second_order_mixer": DirectionalSecondOrderMixerBranch,
        "hessian_sobel_coupled_gate": HessianSobelCoupledGateBranch,
        "entropy_uncertainty_refiner": EntropyUncertaintyRefinerBranch,
        "strip_transformer_cross_gate": StripTransformerCrossGateBranch,
        "pyramid_ridge_context_router": PyramidRidgeContextRouterBranch,
        "frequency_highpass_residual_gate": FrequencyHighpassResidualGateBranch,
    }
    if mode not in branches:
        raise ValueError(f"Unsupported exp_third_module_mode: {mode}. Expected one of {THIRD_EXPERIMENTAL_MODULE_MODES}.")
    return branches[mode](channels)


class ThirdExperimentalEnhancementModule(nn.Module):
    def __init__(
        self,
        channels: int,
        mode: str,
        reduction: int = 4,
        min_hidden_channels: int = 16,
    ):
        super().__init__()
        if mode not in THIRD_EXPERIMENTAL_MODULE_MODES:
            raise ValueError(
                f"Unsupported exp_third_module_mode: {mode}. Expected one of {THIRD_EXPERIMENTAL_MODULE_MODES}."
            )
        self.mode = mode
        hidden_channels = max(channels // reduction, min_hidden_channels)
        self.reduce = ConvGNAct(channels, hidden_channels, kernel_size=1, padding=0)
        self.branch = make_third_branch(mode, hidden_channels)
        self.expand = ConvGNAct(hidden_channels, channels, kernel_size=1, padding=0, act=False)
        attn_channels = max(channels // 8, 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, attn_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(attn_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.04))

    def forward(self, x):
        identity = x
        fused = self.expand(self.branch(self.reduce(x)))
        channel_attn = self.channel_attn(fused)
        spatial_pool = torch.cat(
            [fused.mean(dim=1, keepdim=True), fused.amax(dim=1, keepdim=True)],
            dim=1,
        )
        fused = fused * channel_attn * torch.sigmoid(self.spatial_attn(spatial_pool))
        gate = torch.sigmoid(self.gate(torch.cat([identity, fused], dim=1)))
        return identity + self.gamma * gate * fused
