import torch
import torch.nn as nn

from models.experimental_modules import ExperimentalEnhancementModule
from models.experimental_second_modules import SecondExperimentalEnhancementModule
from models.experimental_third_modules import ThirdExperimentalEnhancementModule


PAPER_STACK_MODES = [
    "saf_rgp",
    "saf_ons",
    "saf_eut",
    "saf_stc",
    "saf_rgd",
    "rgp_ons",
    "rgp_eut",
    "rgp_stc",
    "rgp_rgd",
    "ons_eut",
    "saf_rgp_ons",
    "saf_rgp_eut",
    "saf_rgp_stc",
    "saf_rgp_rgd",
    "saf_ons_eut",
    "saf_ons_stc",
    "saf_ons_rgd",
    "saf_eut_stc",
    "saf_eut_rgd",
    "saf_stc_rgd",
    "rgp_ons_eut",
    "rgp_ons_stc",
    "rgp_ons_rgd",
    "rgp_eut_stc",
    "rgp_eut_rgd",
    "rgp_stc_rgd",
    "ons_eut_stc",
    "ons_eut_rgd",
    "ons_stc_rgd",
    "eut_stc_rgd",
    "saf_rgp_heb",
    "saf_rgp_vsr",
    "saf_rgp_ros",
    "saf_rgp_ccg",
    "saf_rgp_egc",
    "saf_rgp_cwa",
    "saf_rgp_tcc",
    "saf_rgp_srt",
    "saf_rgp_lpq",
    "saf_rgp_adr",
    "saf_rgp_cfc",
    "saf_rgp_mlf",
    "saf_rgp_dsm",
    "saf_rgp_hsc",
    "saf_rgp_pcr",
    "saf_rgp_fhr",
    "saf_eut_vsr",
    "saf_eut_cwa",
    "saf_ons_mlf",
    "saf_stc_pcr",
]


def make_paper_component(name: str, channels: int) -> nn.Module:
    if name == "saf":
        return ExperimentalEnhancementModule(channels=channels, mode="scale_adaptive_fusion")
    if name == "rgp":
        return SecondExperimentalEnhancementModule(channels=channels, mode="ridge_hessian_context")
    if name == "ons":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="orthogonal_noise_suppressor")
    if name == "eut":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="entropy_uncertainty_refiner")
    if name == "stc":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="strip_transformer_cross_gate")
    if name == "rgd":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="ridge_gap_dilation_bank")
    if name == "heb":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="hessian_eigen_bridge_gate")
    if name == "vsr":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="vesselness_scale_router")
    if name == "ros":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="ridge_orientation_strip_attn")
    if name == "ccg":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="curvature_consistency_gate")
    if name == "egc":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="endpoint_gap_completion")
    if name == "cwa":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="crack_width_adaptive_gate")
    if name == "tcc":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="topology_crossing_context")
    if name == "srt":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="sparse_ridge_token_memory")
    if name == "lpq":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="local_phase_quadrature_gate")
    if name == "adr":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="anisotropic_diffusion_refine")
    if name == "cfc":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="centerline_confidence_calibrator")
    if name == "mlf":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="multiscale_laplacian_fusion")
    if name == "dsm":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="directional_second_order_mixer")
    if name == "hsc":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="hessian_sobel_coupled_gate")
    if name == "pcr":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="pyramid_ridge_context_router")
    if name == "fhr":
        return ThirdExperimentalEnhancementModule(channels=channels, mode="frequency_highpass_residual_gate")
    raise ValueError(f"Unsupported paper stack component: {name}")


class PaperStackEnhancementModule(nn.Module):
    def __init__(self, channels: int, mode: str):
        super().__init__()
        if mode not in PAPER_STACK_MODES:
            raise ValueError(f"Unsupported paper_stack_mode: {mode}. Expected one of {PAPER_STACK_MODES}.")
        self.mode = mode
        self.components = nn.ModuleList(
            [make_paper_component(name, channels) for name in mode.split("_")]
        )
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x
        stacked = x
        for component in self.components:
            stacked = component(stacked)
        delta = stacked - identity
        gate = torch.sigmoid(self.gate(torch.cat([identity, stacked], dim=1)))
        return identity + self.gamma * gate * delta
