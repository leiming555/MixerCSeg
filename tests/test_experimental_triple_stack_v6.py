import argparse
from pathlib import Path

import pytest
import torch

from main import get_args_parser
from models.encoder.vss_block import VSS
from models.experimental_triple_stack_v3 import TRIPLE_STACK_V3_MODES
from models.experimental_triple_stack_v4 import TRIPLE_STACK_V4_MODES
from models.experimental_triple_stack_v5 import TRIPLE_STACK_V5_MODES
from models.experimental_triple_stack_v6 import (
    AdaptivePrecisionBridge,
    EigenRidgeTopologyMixer,
    HybridFrequencyShiftRouter,
    StructureOrientationCrossAttention,
    TRIPLE_STACK_V6_MODES,
    TripleStackV6Block,
    UncertaintyGapCompetition,
)
from models.layers import HoGEdgeGateConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_triple_stack_v6_modes_are_unique_and_new():
    assert len(TRIPLE_STACK_V6_MODES) == 20
    assert len(set(TRIPLE_STACK_V6_MODES)) == 20
    assert set(TRIPLE_STACK_V6_MODES).isdisjoint(TRIPLE_STACK_V3_MODES)
    assert set(TRIPLE_STACK_V6_MODES).isdisjoint(TRIPLE_STACK_V4_MODES)
    assert set(TRIPLE_STACK_V6_MODES).isdisjoint(TRIPLE_STACK_V5_MODES)


def test_triple_stack_v6_modules_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V6_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        module = TripleStackV6Block(channels=16, nbins=18, mode=mode)
        y = module(x)
        y.mean().backward()

        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_triple_stack_v6_keeps_three_points_around_original_degconv():
    block = TripleStackV6Block(channels=16, nbins=18, mode="hfr_ert_apb")
    assert isinstance(block.point1, HybridFrequencyShiftRouter)
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.point2, EigenRidgeTopologyMixer)
    assert isinstance(block.point3, AdaptivePrecisionBridge)

    cross_block = TripleStackV6Block(channels=16, nbins=18, mode="rst_soc_ugc")
    assert isinstance(cross_block.point2, StructureOrientationCrossAttention)
    assert isinstance(cross_block.point3, UncertaintyGapCompetition)


def test_encoder_inserts_triple_stack_v6_after_transmixer():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        nbins=18,
        use_triple_stack_v6=True,
        triple_stack_v6_mode="hfr_ert_apb",
    )
    assert isinstance(vss.blocks[0][1], TripleStackV6Block)
    assert isinstance(vss.blocks[0][1].deg, HoGEdgeGateConv)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_triple_stack_v3": True, "triple_stack_v3_mode": "msa_ocv_gbc"},
        {"use_triple_stack_v4": True, "triple_stack_v4_mode": "adc_eov_pgs"},
        {"use_triple_stack_v5": True, "triple_stack_v5_mode": "fap_rtc_upb"},
        {"use_ccem": True},
    ],
)
def test_encoder_rejects_triple_stack_v6_with_other_experimental_modules(kwargs):
    with pytest.raises(ValueError, match="use_triple_stack_v6 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v6=True,
            triple_stack_v6_mode="hfr_ert_apb",
            **kwargs,
        )


def test_parser_and_checkpoint_args_include_triple_stack_v6_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v6"].default is False
    assert actions["triple_stack_v6_mode"].choices == TRIPLE_STACK_V6_MODES
    assert '"use_triple_stack_v6"' in infer_source
    assert '"triple_stack_v6_mode"' in infer_source
    assert "TripleStack-v6 enabled" in profile_source
    assert "TripleStack-v6 mode" in profile_source
    assert "use_triple_stack_v6=getattr(args, 'use_triple_stack_v6', False)" in segmentor_source
