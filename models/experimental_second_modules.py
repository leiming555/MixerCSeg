import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct, make_gn


SECOND_EXPERIMENTAL_MODULE_MODES = [
    "soft_morph_gradient_gate",
    "ridge_hessian_context",
    "orientation_vote_attention",
    "thin_skeleton_refine",
    "subpixel_context_shuffle",
    "deformable_shift_proxy",
    "anisotropic_dilation_bank",
    "local_contrast_calibrator",
    "rank_channel_mixer",
    "edge_token_memory",
    "hog_residual_proxy",
    "adaptive_threshold_context",
    "multi_kernel_separable_attn",
    "boundary_uncertainty_suppressor",
    "line_endpoint_context",
    "micro_transformer_pool",
    "global_crack_prior_gate",
    "texture_suppression_gate",
    "cross_axis_hadamard_mixer",
    "progressive_rf_gate",
]


class SoftMorphGradientBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        gradient = dilated - eroded
        return self.fuse(torch.cat([x, gradient, x * torch.sigmoid(gradient)], dim=1))


class RidgeHessianContextBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        dxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]).view(1, 1, 3, 3)
        dyy = torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        dxy = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) * 0.25
        self.register_buffer("dxx", dxx)
        self.register_buffer("dyy", dyy)
        self.register_buffer("dxy", dxy)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def _filter(self, x, kernel):
        return F.conv2d(x, kernel.to(dtype=x.dtype).repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])

    def forward(self, x):
        dxx = self._filter(x, self.dxx)
        dyy = self._filter(x, self.dyy)
        dxy = self._filter(x, self.dxy)
        ridge = torch.sqrt((dxx - dyy).pow(2) + 4.0 * dxy.pow(2) + 1e-6)
        return self.fuse(torch.cat([x, ridge, torch.abs(dxx), torch.abs(dyy)], dim=1))


class OrientationVoteAttentionBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.h = ConvGNAct(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels)
        self.v = ConvGNAct(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        diag_a = 0.5 * (torch.roll(x, shifts=(1, 1), dims=(-2, -1)) + torch.roll(x, shifts=(-1, -1), dims=(-2, -1)))
        diag_b = 0.5 * (torch.roll(x, shifts=(1, -1), dims=(-2, -1)) + torch.roll(x, shifts=(-1, 1), dims=(-2, -1)))
        return self.fuse(torch.cat([self.h(x), self.v(x), diag_a, diag_b], dim=1))


class ThinSkeletonRefineBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        thin = F.relu(x - local_mean)
        peak = x * torch.sigmoid(x - local_max)
        return self.fuse(torch.cat([x, thin, peak], dim=1))


class SubpixelContextShuffleBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        shuffled = F.pixel_shuffle(self.expand(x), upscale_factor=2)
        shuffled = F.avg_pool2d(shuffled, kernel_size=2, stride=2)
        return self.fuse(torch.cat([x, shuffled], dim=1))


class DeformableShiftProxyBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 4, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        shifts = torch.stack(
            [
                torch.roll(x, shifts=1, dims=-1),
                torch.roll(x, shifts=-1, dims=-1),
                torch.roll(x, shifts=1, dims=-2),
                torch.roll(x, shifts=-1, dims=-2),
            ],
            dim=1,
        )
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((shifts * weights).sum(dim=1))


class AnisotropicDilationBankBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.h1 = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.h2 = ConvGNAct(channels, channels, kernel_size=(1, 7), padding=(0, 6), dilation=(1, 2), groups=channels)
        self.v1 = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.v2 = ConvGNAct(channels, channels, kernel_size=(7, 1), padding=(6, 0), dilation=(2, 1), groups=channels)
        self.fuse = ConvGNAct(channels * 4, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.h1(x), self.h2(x), self.v1(x), self.v2(x)], dim=1))


class LocalContrastCalibratorBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        var = F.avg_pool2d(x * x, kernel_size=5, stride=1, padding=2) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return self.fuse(torch.cat([x - mean, std, x], dim=1))


class RankChannelMixerBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        rank_channels = max(channels // 4, 4)
        self.net = nn.Sequential(
            ConvGNAct(channels, rank_channels, kernel_size=1, padding=0),
            ConvGNAct(rank_channels, rank_channels, kernel_size=3, padding=1, groups=rank_channels),
            ConvGNAct(rank_channels, channels, kernel_size=1, padding=0, act=False),
        )

    def forward(self, x):
        return self.net(x)


class EdgeTokenMemoryBranch(nn.Module):
    def __init__(self, channels: int, num_tokens: int = 4):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, channels) * 0.02)
        self.value = nn.Linear(channels, channels, bias=False)
        self.proj = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        query = x.mean(dim=(-2, -1))
        attn = torch.softmax(torch.matmul(query.unsqueeze(1), self.tokens.transpose(1, 2)), dim=-1)
        context = torch.matmul(attn, self.value(self.tokens)).squeeze(1)
        context = context[:, :, None, None].expand_as(x)
        return self.proj(torch.cat([x, context], dim=1))


class HogResidualProxyBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sy = sx.transpose(-1, -2)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def _filter(self, x, kernel):
        return F.conv2d(x, kernel.to(dtype=x.dtype).repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])

    def forward(self, x):
        gx = self._filter(x, self.sx)
        gy = self._filter(x, self.sy)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        signed = gx * torch.sigmoid(gy)
        return self.fuse(torch.cat([x, mag, signed], dim=1))


class AdaptiveThresholdContextBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        gate = torch.sigmoid((x - mean) * 2.0)
        return self.fuse(torch.cat([x * gate, mean, gate], dim=1))


class MultiKernelSeparableAttnBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.k3 = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.k5 = ConvGNAct(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.k9 = ConvGNAct(channels, channels, kernel_size=9, padding=4, groups=channels)
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 3, kernel_size=1))
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        feats = torch.stack([self.k3(x), self.k5(x), self.k9(x)], dim=1)
        weights = self.router(x).softmax(dim=1).unsqueeze(2)
        return self.proj((feats * weights).sum(dim=1))


class BoundaryUncertaintySuppressorBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.uncertainty = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        boundary = torch.abs(x - local)
        pooled = torch.cat([boundary.mean(dim=1, keepdim=True), boundary.amax(dim=1, keepdim=True)], dim=1)
        suppress = torch.sigmoid(self.uncertainty(pooled))
        refined = x * (1.0 - suppress) + boundary * suppress
        return self.fuse(torch.cat([refined, boundary], dim=1))


class LineEndpointContextBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        cross = 0.25 * (
            torch.roll(x, shifts=1, dims=-1)
            + torch.roll(x, shifts=-1, dims=-1)
            + torch.roll(x, shifts=1, dims=-2)
            + torch.roll(x, shifts=-1, dims=-2)
        )
        endpoints = F.relu(x - cross)
        continuity = 0.5 * (torch.roll(x, shifts=2, dims=-1) + torch.roll(x, shifts=-2, dims=-1))
        return self.fuse(torch.cat([x, endpoints, continuity], dim=1))


class MicroTransformerPoolBranch(nn.Module):
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
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(self.norm(x), (self.pool_size, self.pool_size))
        qkv = self.qkv(pooled).reshape(b, 3, self.num_heads, self.head_dim, self.pool_size * self.pool_size)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 1, 3, 2).reshape(b, c, self.pool_size, self.pool_size)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return self.proj(out)


class GlobalCrackPriorGateBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        attn_channels = max(channels // 4, 4)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, attn_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(attn_channels, channels, kernel_size=1),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.proj = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        spatial_pool = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        gate = torch.sigmoid(self.channel(x) + self.spatial(spatial_pool))
        return self.proj(x * gate)


class TextureSuppressionGateBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.proj = ConvGNAct(channels, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high = x - low
        pooled = torch.cat([torch.abs(high).mean(dim=1, keepdim=True), torch.abs(high).amax(dim=1, keepdim=True)], dim=1)
        keep = torch.sigmoid(self.gate(pooled))
        return self.proj(low + high * keep)


class CrossAxisHadamardMixerBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        row = x.mean(dim=-1, keepdim=True).expand_as(x)
        col = x.mean(dim=-2, keepdim=True).expand_as(x)
        hadamard = row * col
        return self.fuse(torch.cat([x, x * torch.sigmoid(hadamard), row + col], dim=1))


class ProgressiveRFGateBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.dw2 = ConvGNAct(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.dw3 = ConvGNAct(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels)
        self.fuse = ConvGNAct(channels * 3, channels, kernel_size=1, padding=0, act=False)

    def forward(self, x):
        x1 = self.dw1(x)
        x2 = self.dw2(x + x1)
        x3 = self.dw3(x1 + x2)
        return self.fuse(torch.cat([x1, x2, x3], dim=1))


def make_second_branch(mode: str, channels: int) -> nn.Module:
    branches = {
        "soft_morph_gradient_gate": SoftMorphGradientBranch,
        "ridge_hessian_context": RidgeHessianContextBranch,
        "orientation_vote_attention": OrientationVoteAttentionBranch,
        "thin_skeleton_refine": ThinSkeletonRefineBranch,
        "subpixel_context_shuffle": SubpixelContextShuffleBranch,
        "deformable_shift_proxy": DeformableShiftProxyBranch,
        "anisotropic_dilation_bank": AnisotropicDilationBankBranch,
        "local_contrast_calibrator": LocalContrastCalibratorBranch,
        "rank_channel_mixer": RankChannelMixerBranch,
        "edge_token_memory": EdgeTokenMemoryBranch,
        "hog_residual_proxy": HogResidualProxyBranch,
        "adaptive_threshold_context": AdaptiveThresholdContextBranch,
        "multi_kernel_separable_attn": MultiKernelSeparableAttnBranch,
        "boundary_uncertainty_suppressor": BoundaryUncertaintySuppressorBranch,
        "line_endpoint_context": LineEndpointContextBranch,
        "micro_transformer_pool": MicroTransformerPoolBranch,
        "global_crack_prior_gate": GlobalCrackPriorGateBranch,
        "texture_suppression_gate": TextureSuppressionGateBranch,
        "cross_axis_hadamard_mixer": CrossAxisHadamardMixerBranch,
        "progressive_rf_gate": ProgressiveRFGateBranch,
    }
    if mode not in branches:
        raise ValueError(f"Unsupported exp_second_module_mode: {mode}. Expected one of {SECOND_EXPERIMENTAL_MODULE_MODES}.")
    return branches[mode](channels)


class SecondExperimentalEnhancementModule(nn.Module):
    def __init__(
        self,
        channels: int,
        mode: str,
        reduction: int = 4,
        min_hidden_channels: int = 16,
    ):
        super().__init__()
        if mode not in SECOND_EXPERIMENTAL_MODULE_MODES:
            raise ValueError(
                f"Unsupported exp_second_module_mode: {mode}. Expected one of {SECOND_EXPERIMENTAL_MODULE_MODES}."
            )
        self.mode = mode
        hidden_channels = max(channels // reduction, min_hidden_channels)
        self.reduce = ConvGNAct(channels, hidden_channels, kernel_size=1, padding=0)
        self.branch = make_second_branch(mode, hidden_channels)
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
        self.gamma = nn.Parameter(torch.tensor(0.05))

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
