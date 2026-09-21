import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn
from models.experimental_paper_stack import PaperStackEnhancementModule


RSGDI_V2_MODES = [
    "topology_bridge_gate",
    "width_adaptive_router",
    "endpoint_bilateral_completion",
    "ridge_token_memory",
    "hessian_laplacian_consensus",
    "frequency_gap_residual",
    "orientation_frequency_coupler",
    "boundary_entropy_hardening",
    "precision_recall_balance_gate",
    "anisotropic_scale_router",
    "subpixel_ridge_alignment",
    "local_global_ridge_fusion",
    "sparse_centerline_router",
    "strip_token_gap_attention",
    "multi_order_gap_refiner",
    "topology_preserving_suppressor",
    "directional_width_calibrator",
    "ridge_gap_transformer_lite",
    "uncertainty_gap_bridge",
    "ensemble_rsgdi_gate",
]


def _depthwise_filter(x, kernel):
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    return F.conv2d(x, kernel.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])


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
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return gx, gy, mag


class TopologyBridgeGateBranch(nn.Module):
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
        bridge = torch.sigmoid(plus + diag - torch.abs(plus - diag)) * x
        return self.fuse(torch.cat([x, plus, diag, bridge], dim=1))


class WidthAdaptiveRouterBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 4, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        bands = []
        for kernel_size in (3, 5, 7, 9):
            local_max = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            bands.append((local_max - local_mean) * torch.sigmoid(local_max - x))
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((torch.stack(bands, dim=1) * weights).sum(dim=1))


class EndpointBilateralCompletionBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        local = 0.25 * (
            torch.roll(x, 1, -1) + torch.roll(x, -1, -1) + torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
        )
        endpoint = F.relu(x - local)
        h_bridge = 0.5 * (torch.roll(endpoint, 2, -1) + torch.roll(endpoint, -2, -1))
        v_bridge = 0.5 * (torch.roll(endpoint, 2, -2) + torch.roll(endpoint, -2, -2))
        bilateral = torch.sigmoid(h_bridge + v_bridge) * (h_bridge + v_bridge)
        return self.fuse(torch.cat([x, endpoint, h_bridge, v_bridge, bilateral], dim=1))


class RidgeTokenMemoryBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int, num_tokens: int = 6):
        super().__init__()
        self._init_diff_kernels()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, channels) * 0.02)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.proj = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian_ridge(x)
        score = torch.sigmoid(ridge.mean(dim=1, keepdim=True))
        query = (x * score).mean(dim=(-2, -1))
        attn = torch.softmax(torch.matmul(query.unsqueeze(1), self.key(self.tokens).transpose(1, 2)), dim=-1)
        context = torch.matmul(attn, self.value(self.tokens)).squeeze(1)
        context = context[:, :, None, None].expand_as(x)
        return self.proj(torch.cat([x, ridge, context], dim=1))


class HessianLaplacianConsensusBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian_ridge(x)
        lap = torch.abs(_depthwise_filter(x, self.lap))
        consensus = torch.sigmoid(ridge - lap) * ridge
        return self.fuse(torch.cat([x, ridge, lap, consensus], dim=1))


class FrequencyGapResidualBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        high = x - low
        gap = torch.clamp(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x, min=0.0)
        lap = torch.abs(_depthwise_filter(x, self.lap))
        return self.fuse(torch.cat([x, high, gap, lap], dim=1))


class OrientationFrequencyCouplerBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        gx, gy, mag = self._sobel(x)
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high = x - low
        orient = torch.sigmoid(torch.abs(gx) + torch.abs(gy) - torch.abs(gx - gy))
        return self.fuse(torch.cat([x, mag, high, high * orient], dim=1))


class BoundaryEntropyHardeningBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        p = torch.sigmoid(x)
        entropy = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        boundary = torch.abs(x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        pooled = torch.cat([entropy.mean(dim=1, keepdim=True), boundary.amax(dim=1, keepdim=True)], dim=1)
        hard = torch.sigmoid(self.spatial(pooled))
        return self.proj(x + hard * (boundary - entropy))


class PrecisionRecallBalanceGateBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_max = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_min = -F.max_pool2d(-x, kernel_size=5, stride=1, padding=2)
        precision_gate = torch.sigmoid(x - mean)
        recall_gate = torch.sigmoid(local_max - x)
        balanced = x * precision_gate + 0.5 * (local_max - local_min) * recall_gate
        return self.fuse(torch.cat([x, precision_gate, recall_gate, balanced], dim=1))


class AnisotropicScaleRouterBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.h1 = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.h2 = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 6), dilation=(1, 2), groups=channels)
        self.v1 = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.v2 = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(6, 0), dilation=(2, 1), groups=channels)
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 4, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        feats = torch.stack([self.h1(x), self.h2(x), self.v1(x), self.v2(x)], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((feats * weights).sum(dim=1))


class SubpixelRidgeAlignmentBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.expand = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        aligned = F.pixel_shuffle(self.expand(x), upscale_factor=2)
        aligned = F.avg_pool2d(aligned, kernel_size=2, stride=2)
        _, _, _, ridge = self._hessian_ridge(aligned)
        return self.fuse(torch.cat([x, aligned, aligned * torch.sigmoid(ridge)], dim=1))


class LocalGlobalRidgeFusionBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int, pool_size: int = 4):
        super().__init__()
        self._init_diff_kernels()
        self.pool_size = pool_size
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian_ridge(x)
        global_ctx = F.adaptive_avg_pool2d(ridge, (self.pool_size, self.pool_size))
        global_ctx = F.interpolate(global_ctx, size=x.shape[-2:], mode="bilinear", align_corners=False)
        local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.fuse(torch.cat([x, ridge, local, global_ctx], dim=1))


class SparseCenterlineRouterBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 3, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        _, _, _, ridge = self._hessian_ridge(x)
        center = torch.sigmoid(ridge - F.avg_pool2d(ridge, kernel_size=5, stride=1, padding=2))
        thin = F.relu(x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        dense = F.max_pool2d(thin, kernel_size=5, stride=1, padding=2)
        feats = torch.stack([x * center, thin, dense], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((feats * weights).sum(dim=1))


class StripTokenGapAttentionBranch(nn.Module):
    def __init__(self, channels: int, num_tokens: int = 4):
        super().__init__()
        self.h = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels)
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, channels) * 0.02)
        self.value = nn.Linear(channels, channels, bias=False)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        strip = self.h(x) + self.v(x)
        gap = torch.clamp(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x, min=0.0)
        query = gap.mean(dim=(-2, -1))
        attn = torch.softmax(torch.matmul(query.unsqueeze(1), self.tokens.transpose(1, 2)), dim=-1)
        context = torch.matmul(attn, self.value(self.tokens)).squeeze(1)
        context = context[:, :, None, None].expand_as(x)
        return self.fuse(torch.cat([x, strip, gap, context], dim=1))


class MultiOrderGapRefinerBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        x1 = self.dw(x)
        x2 = self.dw(x1)
        gap = torch.clamp(F.max_pool2d(x2, kernel_size=5, stride=1, padding=2) - x2, min=0.0)
        return self.fuse(torch.cat([x, x1, x2, gap], dim=1))


class TopologyPreservingSuppressorBranch(nn.Module):
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
        isolated = torch.abs(x - 0.5 * (plus + diag))
        keep = torch.sigmoid(plus + diag - isolated)
        return self.fuse(torch.cat([x * keep, plus, diag, isolated], dim=1))


class DirectionalWidthCalibratorBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.fuse = ConvGNAct(channels * 5, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        gx, gy, mag = self._sobel(x)
        width3 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1) - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        width7 = F.max_pool2d(x, kernel_size=7, stride=1, padding=3) - F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        direction = torch.sigmoid(torch.abs(gx) + torch.abs(gy))
        calibrated = (width3 + width7) * direction
        return self.fuse(torch.cat([x, mag, width3, width7, calibrated], dim=1))


class RidgeGapTransformerLiteBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int, pool_size: int = 4):
        super().__init__()
        self._init_diff_kernels()
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
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        _, _, _, ridge = self._hessian_ridge(x)
        gap = torch.clamp(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x, min=0.0)
        pooled = F.adaptive_avg_pool2d(self.norm(ridge + gap), (self.pool_size, self.pool_size))
        qkv = self.qkv(pooled).reshape(b, 3, self.num_heads, self.head_dim, self.pool_size * self.pool_size)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        context = attn @ v
        context = context.permute(0, 1, 3, 2).reshape(b, c, self.pool_size, self.pool_size)
        context = F.interpolate(context, size=(h, w), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([ridge, gap, context], dim=1))


class UncertaintyGapBridgeBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        p = torch.sigmoid(x)
        entropy = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        gap = torch.clamp(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x, min=0.0)
        uncertainty = torch.sigmoid(self.spatial(torch.cat([entropy.mean(dim=1, keepdim=True), gap.amax(dim=1, keepdim=True)], dim=1)))
        bridge = x * (1.0 - uncertainty) + gap * uncertainty
        return self.fuse(torch.cat([x, entropy, bridge], dim=1))


class EnsembleRSGDIGateBranch(nn.Module, DifferentialKernelsMixin):
    def __init__(self, channels: int):
        super().__init__()
        self._init_diff_kernels()
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 4, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        size = x.shape[-2:]
        scale = F.interpolate(F.avg_pool2d(x, kernel_size=4, stride=4, ceil_mode=True), size=size, mode="bilinear", align_corners=False)
        _, _, _, ridge = self._hessian_ridge(x)
        gap = torch.clamp(F.max_pool2d(x, kernel_size=5, stride=1, padding=2) - x, min=0.0)
        p = torch.sigmoid(x)
        entropy = -(p * torch.log(p + 1e-6) + (1.0 - p) * torch.log(1.0 - p + 1e-6))
        feats = torch.stack([scale, ridge, gap, x - entropy], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((feats * weights).sum(dim=1))


def make_rsgdi_v2_branch(mode: str, channels: int) -> nn.Module:
    branches = {
        "topology_bridge_gate": TopologyBridgeGateBranch,
        "width_adaptive_router": WidthAdaptiveRouterBranch,
        "endpoint_bilateral_completion": EndpointBilateralCompletionBranch,
        "ridge_token_memory": RidgeTokenMemoryBranch,
        "hessian_laplacian_consensus": HessianLaplacianConsensusBranch,
        "frequency_gap_residual": FrequencyGapResidualBranch,
        "orientation_frequency_coupler": OrientationFrequencyCouplerBranch,
        "boundary_entropy_hardening": BoundaryEntropyHardeningBranch,
        "precision_recall_balance_gate": PrecisionRecallBalanceGateBranch,
        "anisotropic_scale_router": AnisotropicScaleRouterBranch,
        "subpixel_ridge_alignment": SubpixelRidgeAlignmentBranch,
        "local_global_ridge_fusion": LocalGlobalRidgeFusionBranch,
        "sparse_centerline_router": SparseCenterlineRouterBranch,
        "strip_token_gap_attention": StripTokenGapAttentionBranch,
        "multi_order_gap_refiner": MultiOrderGapRefinerBranch,
        "topology_preserving_suppressor": TopologyPreservingSuppressorBranch,
        "directional_width_calibrator": DirectionalWidthCalibratorBranch,
        "ridge_gap_transformer_lite": RidgeGapTransformerLiteBranch,
        "uncertainty_gap_bridge": UncertaintyGapBridgeBranch,
        "ensemble_rsgdi_gate": EnsembleRSGDIGateBranch,
    }
    if mode not in branches:
        raise ValueError(f"Unsupported rsgdi_v2_mode: {mode}. Expected one of {RSGDI_V2_MODES}.")
    return branches[mode](channels)


class RSGDIV2EnhancementModule(nn.Module):
    def __init__(
        self,
        channels: int,
        mode: str,
        reduction: int = 4,
        min_hidden_channels: int = 16,
    ):
        super().__init__()
        if mode not in RSGDI_V2_MODES:
            raise ValueError(f"Unsupported rsgdi_v2_mode: {mode}. Expected one of {RSGDI_V2_MODES}.")
        self.mode = mode
        self.base_stack = PaperStackEnhancementModule(channels=channels, mode="saf_rgp_rgd")
        hidden_channels = max(channels // reduction, min_hidden_channels)
        self.reduce = ConvGNAct(channels, hidden_channels, kernel_size=1, padding=0)
        self.branch = make_rsgdi_v2_branch(mode, hidden_channels)
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
        self.base_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.refine_gate = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.base_gamma = nn.Parameter(torch.tensor(0.5))
        self.refine_gamma = nn.Parameter(torch.tensor(0.03))

    def forward(self, x):
        identity = x
        base = self.base_stack(x)
        fused = self.expand(self.branch(self.reduce(base)))
        channel_attn = self.channel_attn(fused)
        spatial_pool = torch.cat([fused.mean(dim=1, keepdim=True), fused.amax(dim=1, keepdim=True)], dim=1)
        fused = fused * channel_attn * torch.sigmoid(self.spatial_attn(spatial_pool))
        base_gate = torch.sigmoid(self.base_gate(torch.cat([identity, base], dim=1)))
        refine_gate = torch.sigmoid(self.refine_gate(torch.cat([identity, base, fused], dim=1)))
        return identity + self.base_gamma * base_gate * (base - identity) + self.refine_gamma * refine_gate * fused
