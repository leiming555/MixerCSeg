import argparse
from pathlib import Path

import pytest
import torch

from main import get_args_parser
from models.encoder.vss_block import VSS
from models.experimental_triple_stack_v3 import TRIPLE_STACK_V3_MODES
from models.experimental_triple_stack_v4 import TRIPLE_STACK_V4_MODES
from models.experimental_triple_stack_v5 import TRIPLE_STACK_V5_MODES
from models.experimental_triple_stack_v6 import TRIPLE_STACK_V6_MODES
from models.experimental_triple_stack_v7 import (
    CurvatureTopologyVoting,
    DualGatedBoundaryBridge,
    TRIPLE_STACK_V7_MODES,
    WidthAdaptiveMSA,
)
from models.experimental_triple_stack_v8 import TRIPLE_STACK_V8_MODES
from models.experimental_triple_stack_v9 import (
    ConfidenceFillRefiner,
    DisagreementConfidenceBypass,
    EntropyUncertaintyBypass,
    GeodesicTopologyPropagator,
    MultiScaleConfidenceBypass,
    OrientedEndpointBridge,
    PrecisionRecallGuard,
    RecallCalibrationState,
    TRIPLE_STACK_V9_MODES,
    TripleStackV9Block,
    _shift_zero,
)
from models.layers import HoGEdgeGateConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_triple_stack_v9_modes_are_unique_and_new():
    previous_modes = (
        set(TRIPLE_STACK_V3_MODES)
        | set(TRIPLE_STACK_V4_MODES)
        | set(TRIPLE_STACK_V5_MODES)
        | set(TRIPLE_STACK_V6_MODES)
        | set(TRIPLE_STACK_V7_MODES)
        | set(TRIPLE_STACK_V8_MODES)
    )
    assert len(TRIPLE_STACK_V9_MODES) == 12
    assert len(set(TRIPLE_STACK_V9_MODES)) == 12
    assert set(TRIPLE_STACK_V9_MODES).isdisjoint(previous_modes)


def test_triple_stack_v9_modes_cover_three_by_two_by_two_grid():
    combinations = {tuple(mode.split("_")) for mode in TRIPLE_STACK_V9_MODES}
    assert {item[0] for item in combinations} == {"eub", "dcb", "mcb"}
    assert {item[1] for item in combinations} == {"oeb", "gtp"}
    assert {item[2] for item in combinations} == {"prg", "cfr"}
    assert len(combinations) == 3 * 2 * 2


@pytest.mark.parametrize(
    "mode,component_types",
    [
        (
            "eub_oeb_prg",
            (EntropyUncertaintyBypass, OrientedEndpointBridge, PrecisionRecallGuard),
        ),
        (
            "dcb_gtp_cfr",
            (DisagreementConfidenceBypass, GeodesicTopologyPropagator, ConfidenceFillRefiner),
        ),
        (
            "mcb_oeb_cfr",
            (MultiScaleConfidenceBypass, OrientedEndpointBridge, ConfidenceFillRefiner),
        ),
    ],
)
def test_triple_stack_v9_component_mapping_and_v7_chain(mode, component_types):
    block = TripleStackV9Block(channels=16, nbins=18, mode=mode)
    assert isinstance(block.wma, WidthAdaptiveMSA)
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.ctv, CurvatureTopologyVoting)
    assert isinstance(block.bypass, component_types[0])
    assert isinstance(block.bridge, component_types[1])
    assert isinstance(block.dgb, DualGatedBoundaryBridge)
    assert isinstance(block.guard, component_types[2])


@pytest.mark.parametrize(
    "module_type",
    [EntropyUncertaintyBypass, DisagreementConfidenceBypass, MultiScaleConfidenceBypass],
)
def test_bypass_builds_finite_calibration_state(module_type):
    reference = torch.randn(2, 16, 12, 10)
    refined = torch.randn_like(reference)
    state = module_type(16)(reference, refined)
    assert isinstance(state, RecallCalibrationState)
    assert state.features.shape == reference.shape
    for cue in (
        state.uncertainty,
        state.direction_consistency,
        state.bridge_confidence,
    ):
        assert cue.shape == (2, 1, 12, 10)
        assert torch.isfinite(cue).all()
        assert cue.min() >= 0.0
        assert cue.max() <= 1.0


@pytest.mark.parametrize("bridge_type", [OrientedEndpointBridge, GeodesicTopologyPropagator])
@pytest.mark.parametrize("guard_type", [PrecisionRecallGuard, ConfidenceFillRefiner])
def test_shared_state_flows_from_bypass_through_bridge_to_guard(bridge_type, guard_type):
    reference = torch.randn(1, 16, 12, 10)
    refined = torch.randn_like(reference)
    bypass_state = EntropyUncertaintyBypass(16)(reference, refined)
    bridged_state = bridge_type(16)(bypass_state)
    output = guard_type(16)(bridged_state, refined)

    assert bridged_state.features.shape == reference.shape
    assert bridged_state.bridge_confidence.shape == (1, 1, 12, 10)
    assert torch.isfinite(bridged_state.bridge_confidence).all()
    assert output.shape == reference.shape
    assert torch.isfinite(output).all()


def test_triple_stack_v9_modes_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V9_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        module = TripleStackV9Block(channels=16, nbins=18, mode=mode)
        y = module(x)
        y.mean().backward()
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_zero_padded_shift_does_not_wrap_boundaries():
    x = torch.zeros(1, 1, 4, 4)
    x[..., 0, 0] = 1.0
    shifted_right = _shift_zero(x, 0, 1)
    shifted_down = _shift_zero(x, 1, 0)
    assert shifted_right[..., 0, 1].item() == 1.0
    assert shifted_right[..., 0, 0].item() == 0.0
    assert shifted_down[..., 1, 0].item() == 1.0
    assert shifted_down[..., 0, 0].item() == 0.0


def test_encoder_inserts_triple_stack_v9_after_transmixer():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        nbins=18,
        use_triple_stack_v9=True,
        triple_stack_v9_mode="eub_oeb_prg",
    )
    block = vss.blocks[0][1]
    assert isinstance(block, TripleStackV9Block)
    assert isinstance(block.deg, HoGEdgeGateConv)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_triple_stack_v3": True, "triple_stack_v3_mode": "msa_ocv_gbc"},
        {"use_triple_stack_v4": True, "triple_stack_v4_mode": "adc_eov_pgs"},
        {"use_triple_stack_v5": True, "triple_stack_v5_mode": "fap_rtc_upb"},
        {"use_triple_stack_v6": True, "triple_stack_v6_mode": "hfr_ert_apb"},
        {"use_triple_stack_v7": True, "triple_stack_v7_mode": "wma_ctv_dgb"},
        {"use_triple_stack_v8": True, "triple_stack_v8_mode": "cwc_lsf_obp"},
        {"use_ccem": True},
        {"use_paper_stack": True},
        {"use_exp_module": True},
    ],
)
def test_encoder_rejects_triple_stack_v9_with_other_experimental_modules(kwargs):
    with pytest.raises(ValueError, match="use_triple_stack_v9 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v9=True,
            triple_stack_v9_mode="eub_oeb_prg",
            **kwargs,
        )


def test_parser_and_checkpoint_args_include_triple_stack_v9_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v9"].default is False
    assert actions["triple_stack_v9_mode"].choices == TRIPLE_STACK_V9_MODES
    assert '"use_triple_stack_v9"' in infer_source
    assert '"triple_stack_v9_mode"' in infer_source
    assert "TripleStack-v9 enabled" in profile_source
    assert "TripleStack-v9 mode" in profile_source
    assert "use_triple_stack_v9=getattr(args, 'use_triple_stack_v9', False)" in segmentor_source


def test_default_encoder_path_remains_original_degconv():
    vss = VSS(in_dim=16, depth=1, state_dim=8, nbins=18)
    assert isinstance(vss.blocks[0][1], HoGEdgeGateConv)
