import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental_modules import ConvGNAct
from models.experimental_third_modules import ThirdExperimentalEnhancementModule
from models.layers import HoGEdgeGateConv


DRSGI_MODES = [
    "pre_gate",
    "parallel_gate",
    "sandwich_gate",
    "pre_scale_ridge_post_gap",
    "direction_ridge_cross",
    "direction_gap_cross",
    "full_cross_gate",
    "full_cross_gate_post_rgd",
]


def _depthwise_filter(x, kernel):
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    return F.conv2d(x, kernel.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])


class RidgeScaleGapInteractionCore(nn.Module):
    def __init__(
        self,
        channels: int,
        variant: str = "full",
        reduction: int = 4,
        min_hidden_channels: int = 16,
    ):
        super().__init__()
        if variant not in {"full", "ridge", "gap", "scale_ridge"}:
            raise ValueError(f"Unsupported DRSGI core variant: {variant}")
        self.variant = variant
        hidden_channels = max(channels // reduction, min_hidden_channels)
        self.reduce = ConvGNAct(channels, hidden_channels, kernel_size=1, padding=0)
        self.scale_fuse = ConvGNAct(hidden_channels * 3, hidden_channels, kernel_size=1, padding=0, act=False)
        self.ridge_fuse = ConvGNAct(hidden_channels * 4, hidden_channels, kernel_size=1, padding=0, act=False)
        self.gap_fuse = ConvGNAct(hidden_channels * 4, hidden_channels, kernel_size=1, padding=0, act=False)
        self.direction_fuse = ConvGNAct(hidden_channels * 3, hidden_channels, kernel_size=1, padding=0, act=False)
        self.ridge_gate = nn.Sequential(nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1), nn.Sigmoid())
        self.gap_gate = nn.Sequential(nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1), nn.Sigmoid())
        self.fuse = ConvGNAct(hidden_channels * 4, channels, kernel_size=1, padding=0, act=False)
        attn_channels = max(channels // 8, 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, attn_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(attn_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.residual_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.06))

        dxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]).view(1, 1, 3, 3)
        dyy = torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        dxy = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) * 0.25
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sy = sx.transpose(-1, -2)
        self.register_buffer("dxx", dxx)
        self.register_buffer("dyy", dyy)
        self.register_buffer("dxy", dxy)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)

    def _scale_context(self, x):
        size = x.shape[-2:]
        p2 = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        p4 = F.avg_pool2d(x, kernel_size=4, stride=4, ceil_mode=True)
        p2 = F.interpolate(p2, size=size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=size, mode="bilinear", align_corners=False)
        return self.scale_fuse(torch.cat([x, p2, p4], dim=1))

    def _ridge_context(self, x):
        dxx = _depthwise_filter(x, self.dxx)
        dyy = _depthwise_filter(x, self.dyy)
        dxy = _depthwise_filter(x, self.dxy)
        ridge = torch.sqrt((dxx - dyy).pow(2) + 4.0 * dxy.pow(2) + 1e-6)
        return self.ridge_fuse(torch.cat([x, ridge, torch.abs(dxx), torch.abs(dyy)], dim=1))

    def _gap_context(self, x):
        d3 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        d5 = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        d7 = F.max_pool2d(x, kernel_size=7, stride=1, padding=3)
        gap = torch.clamp(d5 - x, min=0.0)
        return self.gap_fuse(torch.cat([x, d3 - x, d7 - d3, gap], dim=1))

    def _direction_context(self, x):
        gx = _depthwise_filter(x, self.sx)
        gy = _depthwise_filter(x, self.sy)
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-6)
        anisotropy = torch.abs(gx - gy)
        return self.direction_fuse(torch.cat([x, magnitude, anisotropy], dim=1))

    def forward(self, x):
        identity = x
        reduced = self.reduce(x)
        scale = self._scale_context(reduced)
        ridge = self._ridge_context(reduced)
        gap = self._gap_context(reduced)
        direction = self._direction_context(reduced)

        ridge_gate = self.ridge_gate(torch.cat([direction, ridge], dim=1))
        gap_gate = self.gap_gate(torch.cat([direction, gap], dim=1))
        if self.variant == "ridge":
            ridge = ridge * (1.0 + ridge_gate)
            gap = gap * gap_gate
        elif self.variant == "gap":
            ridge = ridge * ridge_gate
            gap = gap * (1.0 + gap_gate)
        elif self.variant == "scale_ridge":
            ridge = ridge * (1.0 + ridge_gate)
            gap = gap * 0.5 * gap_gate
        else:
            ridge = ridge * (1.0 + ridge_gate)
            gap = gap * (1.0 + gap_gate)

        fused = self.fuse(torch.cat([scale, ridge, gap, direction], dim=1))
        channel_attn = self.channel_attn(fused)
        spatial_pool = torch.cat([fused.mean(dim=1, keepdim=True), fused.amax(dim=1, keepdim=True)], dim=1)
        fused = fused * channel_attn * torch.sigmoid(self.spatial_attn(spatial_pool))
        gate = torch.sigmoid(self.residual_gate(torch.cat([identity, fused], dim=1)))
        return identity + self.gamma * gate * fused


class DEGRidgeScaleGapInteractionModule(nn.Module):
    def __init__(self, channels: int, nbins: int, mode: str):
        super().__init__()
        if mode not in DRSGI_MODES:
            raise ValueError(f"Unsupported drsgi_mode: {mode}. Expected one of {DRSGI_MODES}.")
        self.mode = mode
        self.deg = HoGEdgeGateConv(in_dim=channels, nbins=nbins)
        pre_variants = {
            "pre_gate": "full",
            "parallel_gate": "full",
            "sandwich_gate": "full",
            "pre_scale_ridge_post_gap": "scale_ridge",
            "direction_ridge_cross": "ridge",
            "direction_gap_cross": "gap",
            "full_cross_gate": "full",
            "full_cross_gate_post_rgd": "full",
        }
        post_variants = {
            "sandwich_gate": "full",
            "pre_scale_ridge_post_gap": "gap",
        }
        self.pre_core = RidgeScaleGapInteractionCore(channels, variant=pre_variants[mode])
        self.post_core = (
            RidgeScaleGapInteractionCore(channels, variant=post_variants[mode])
            if mode in post_variants
            else None
        )
        self.post_rgd = (
            ThirdExperimentalEnhancementModule(channels=channels, mode="ridge_gap_dilation_bank")
            if mode == "full_cross_gate_post_rgd"
            else None
        )
        self.pair_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.triple_gate = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.35))

    def _fuse_pair(self, identity, x):
        gate = torch.sigmoid(self.pair_gate(torch.cat([identity, x], dim=1)))
        return identity + self.gamma * gate * (x - identity)

    def _fuse_triple(self, identity, x, guide):
        gate = torch.sigmoid(self.triple_gate(torch.cat([identity, x, guide], dim=1)))
        mixed = x + self.gamma * gate * (guide - x)
        return self._fuse_pair(identity, mixed)

    def forward(self, x):
        if self.mode == "pre_gate":
            return self._fuse_pair(x, self.deg(self.pre_core(x)))
        if self.mode == "parallel_gate":
            deg = self.deg(x)
            guide = self.pre_core(x)
            return self._fuse_triple(x, deg, guide)
        if self.mode == "sandwich_gate":
            pre = self.pre_core(x)
            deg = self.deg(pre)
            post = self.post_core(deg)
            return self._fuse_triple(x, deg, post)
        if self.mode == "pre_scale_ridge_post_gap":
            pre = self.pre_core(x)
            deg = self.deg(pre)
            post = self.post_core(deg)
            return self._fuse_triple(x, deg, post)
        if self.mode == "direction_ridge_cross":
            return self._fuse_pair(x, self.deg(self.pre_core(x)))
        if self.mode == "direction_gap_cross":
            return self._fuse_pair(x, self.deg(self.pre_core(x)))
        if self.mode == "full_cross_gate":
            pre = self.pre_core(x)
            deg = self.deg(pre)
            return self._fuse_pair(x, deg)
        if self.mode == "full_cross_gate_post_rgd":
            pre = self.pre_core(x)
            deg = self.deg(pre)
            post = self.post_rgd(deg)
            return self._fuse_triple(x, deg, post)
        raise RuntimeError(f"Unhandled drsgi_mode: {self.mode}")
