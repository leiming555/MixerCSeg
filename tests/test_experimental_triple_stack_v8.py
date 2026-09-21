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
from models.experimental_triple_stack_v7 import TRIPLE_STACK_V7_MODES
from models.experimental_triple_stack_v8 import (
    AnisotropicWidthConsensus,
    ContinuousWidthConditioner,
    CurvatureOrientationTokenField,
    DirectionField,
    LearnableSteerableRidgeField,
    OrientationBidirectionalPropagator,
    SpectralWidthAligner,
    TopologyReliabilityCalibrator,
    TRIPLE_STACK_V8_MODES,
    TripleStackV8Block,
    _shift_zero,
)
from models.layers import HoGEdgeGateConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_triple_stack_v8_modes_are_unique_and_new():
    previous_modes = (
        set(TRIPLE_STACK_V3_MODES)
        | set(TRIPLE_STACK_V4_MODES)
        | set(TRIPLE_STACK_V5_MODES)
        | set(TRIPLE_STACK_V6_MODES)
        | set(TRIPLE_STACK_V7_MODES)
    )
    assert len(TRIPLE_STACK_V8_MODES) == 12
    assert len(set(TRIPLE_STACK_V8_MODES)) == 12
    assert set(TRIPLE_STACK_V8_MODES).isdisjoint(previous_modes)


def test_triple_stack_v8_modes_cover_three_by_two_by_two_grid():
    combinations = {tuple(mode.split("_")) for mode in TRIPLE_STACK_V8_MODES}
    assert {item[0] for item in combinations} == {"cwc", "swa", "awc"}
    assert {item[1] for item in combinations} == {"lsf", "cot"}
    assert {item[2] for item in combinations} == {"obp", "trc"}
    assert len(combinations) == 3 * 2 * 2


@pytest.mark.parametrize(
    "mode,component_types",
    [
        (
            "cwc_lsf_obp",
            (ContinuousWidthConditioner, LearnableSteerableRidgeField, OrientationBidirectionalPropagator),
        ),
        (
            "swa_cot_trc",
            (SpectralWidthAligner, CurvatureOrientationTokenField, TopologyReliabilityCalibrator),
        ),
        (
            "awc_lsf_trc",
            (AnisotropicWidthConsensus, LearnableSteerableRidgeField, TopologyReliabilityCalibrator),
        ),
    ],
)
def test_triple_stack_v8_component_mapping(mode, component_types):
    block = TripleStackV8Block(channels=16, nbins=18, mode=mode)
    assert isinstance(block.point1, component_types[0])
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.point2, component_types[1])
    assert isinstance(block.point3, component_types[2])


@pytest.mark.parametrize(
    "module_type",
    [LearnableSteerableRidgeField, CurvatureOrientationTokenField],
)
def test_direction_field_is_normalized_and_finite(module_type):
    x = torch.randn(2, 16, 12, 10, requires_grad=True)
    field = module_type(16)(x)
    assert isinstance(field, DirectionField)
    assert field.features.shape == x.shape
    assert field.probabilities.shape == (2, 8, 12, 10)
    assert field.confidence.shape == (2, 1, 12, 10)
    assert torch.allclose(
        field.probabilities.sum(dim=1),
        torch.ones(2, 12, 10),
        atol=1e-5,
    )
    assert torch.isfinite(field.features).all()
    assert torch.isfinite(field.probabilities).all()
    assert torch.isfinite(field.confidence).all()
    assert field.confidence.min() >= 0.0
    assert field.confidence.max() <= 1.0


def test_triple_stack_v8_modes_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V8_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        module = TripleStackV8Block(channels=16, nbins=18, mode=mode)
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


def test_encoder_inserts_triple_stack_v8_after_transmixer():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        nbins=18,
        use_triple_stack_v8=True,
        triple_stack_v8_mode="cwc_lsf_obp",
    )
    assert isinstance(vss.blocks[0][1], TripleStackV8Block)
    assert isinstance(vss.blocks[0][1].deg, HoGEdgeGateConv)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_triple_stack_v3": True, "triple_stack_v3_mode": "msa_ocv_gbc"},
        {"use_triple_stack_v4": True, "triple_stack_v4_mode": "adc_eov_pgs"},
        {"use_triple_stack_v5": True, "triple_stack_v5_mode": "fap_rtc_upb"},
        {"use_triple_stack_v6": True, "triple_stack_v6_mode": "hfr_ert_apb"},
        {"use_triple_stack_v7": True, "triple_stack_v7_mode": "wma_ctv_dgb"},
        {"use_ccem": True},
        {"use_paper_stack": True},
        {"use_exp_module": True},
    ],
)
def test_encoder_rejects_triple_stack_v8_with_other_experimental_modules(kwargs):
    with pytest.raises(ValueError, match="use_triple_stack_v8 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v8=True,
            triple_stack_v8_mode="cwc_lsf_obp",
            **kwargs,
        )


def test_parser_and_checkpoint_args_include_triple_stack_v8_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v8"].default is False
    assert actions["triple_stack_v8_mode"].choices == TRIPLE_STACK_V8_MODES
    assert '"use_triple_stack_v8"' in infer_source
    assert '"triple_stack_v8_mode"' in infer_source
    assert "TripleStack-v8 enabled" in profile_source
    assert "TripleStack-v8 mode" in profile_source
    assert "use_triple_stack_v8=getattr(args, 'use_triple_stack_v8', False)" in segmentor_source


def test_default_encoder_path_remains_original_degconv():
    vss = VSS(in_dim=16, depth=1, state_dim=8, nbins=18)
    assert isinstance(vss.blocks[0][1], HoGEdgeGateConv)
