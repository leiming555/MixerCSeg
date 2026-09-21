import argparse
from pathlib import Path

import pytest
import torch

from main import get_args_parser
from models.deg_ridge_scale_gap_interaction import DRSGI_MODES
from models.encoder.vss_block import VSS
from models.experimental_modules import EXPERIMENTAL_MODULE_MODES
from models.experimental_paper_stack import PAPER_STACK_MODES
from models.experimental_second_modules import SECOND_EXPERIMENTAL_MODULE_MODES
from models.experimental_third_modules import THIRD_EXPERIMENTAL_MODULE_MODES
from models.experimental_triple_stack_v3 import TRIPLE_STACK_V3_MODES
from models.experimental_triple_stack_v4 import (
    AdaptiveMultiScaleCrossAxisConditioner,
    DirectionalCurvatureTokenMixer,
    HessianEigenOrientationVoting,
    TopologyGapCalibrator,
    TRIPLE_STACK_V4_MODES,
    TripleStackV4Block,
)
from models.layers import HoGEdgeGateConv
from models.rsgdi_v2_modules import RSGDI_V2_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_triple_stack_v4_modes_are_unique_and_new():
    assert len(TRIPLE_STACK_V4_MODES) == 12
    assert len(set(TRIPLE_STACK_V4_MODES)) == 12
    assert set(TRIPLE_STACK_V4_MODES).isdisjoint(TRIPLE_STACK_V3_MODES)
    assert set(TRIPLE_STACK_V4_MODES).isdisjoint(EXPERIMENTAL_MODULE_MODES)
    assert set(TRIPLE_STACK_V4_MODES).isdisjoint(SECOND_EXPERIMENTAL_MODULE_MODES)
    assert set(TRIPLE_STACK_V4_MODES).isdisjoint(THIRD_EXPERIMENTAL_MODULE_MODES)
    assert set(TRIPLE_STACK_V4_MODES).isdisjoint(PAPER_STACK_MODES)
    assert set(TRIPLE_STACK_V4_MODES).isdisjoint(DRSGI_MODES)
    assert set(TRIPLE_STACK_V4_MODES).isdisjoint(RSGDI_V2_MODES)


def test_triple_stack_v4_modules_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V4_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        module = TripleStackV4Block(channels=16, nbins=18, mode=mode)
        y = module(x)
        y.mean().backward()

        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_triple_stack_v4_keeps_three_points_around_original_degconv():
    block = TripleStackV4Block(channels=16, nbins=18, mode="amc_eov_tgc")

    assert isinstance(block.point1, AdaptiveMultiScaleCrossAxisConditioner)
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.point2, HessianEigenOrientationVoting)
    assert isinstance(block.point3, TopologyGapCalibrator)

    token_block = TripleStackV4Block(channels=16, nbins=18, mode="spc_dct_pgs")
    assert isinstance(token_block.point2, DirectionalCurvatureTokenMixer)


def test_encoder_inserts_triple_stack_v4_after_transmixer():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        mlp_ratio=2.0,
        nbins=18,
        use_triple_stack_v4=True,
        triple_stack_v4_mode="amc_eov_tgc",
    )

    assert isinstance(vss.blocks[0][1], TripleStackV4Block)
    assert vss.blocks[0][1].mode == "amc_eov_tgc"
    assert isinstance(vss.blocks[0][1].deg, HoGEdgeGateConv)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_triple_stack_v3": True, "triple_stack_v3_mode": "msa_ocv_gbc"},
        {"use_paper_stack": True, "paper_stack_mode": "saf_rgp_rgd"},
        {"use_exp_module": True, "exp_module_mode": "scale_adaptive_fusion"},
        {"use_ccem": True},
    ],
)
def test_encoder_rejects_triple_stack_v4_with_other_experimental_modules(kwargs):
    with pytest.raises(ValueError, match="use_triple_stack_v4 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v4=True,
            triple_stack_v4_mode="amc_eov_tgc",
            **kwargs,
        )


def test_parser_and_checkpoint_args_include_triple_stack_v4_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    main_source = (PROJECT_ROOT / "main.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v4"].default is False
    assert actions["triple_stack_v4_mode"].choices == TRIPLE_STACK_V4_MODES
    assert "use_triple_stack_v4 -> " in main_source
    assert "triple_stack_v4_mode -> " in main_source
    assert '"use_triple_stack_v4"' in infer_source
    assert '"triple_stack_v4_mode"' in infer_source
    assert "TripleStack-v4 enabled" in profile_source
    assert "TripleStack-v4 mode" in profile_source
    assert "use_triple_stack_v4=getattr(args, 'use_triple_stack_v4', False)" in segmentor_source
