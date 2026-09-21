import argparse
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from main import get_args_parser
from models.deg_ridge_scale_gap_interaction import DRSGI_MODES
from models.encoder.vss_block import VSS
from models.experimental_modules import EXPERIMENTAL_MODULE_MODES
from models.experimental_paper_stack import PAPER_STACK_MODES
from models.experimental_second_modules import SECOND_EXPERIMENTAL_MODULE_MODES
from models.experimental_third_modules import THIRD_EXPERIMENTAL_MODULE_MODES
from models.experimental_triple_stack_v3 import (
    TRIPLE_STACK_V3_MODES,
    TripleStackV3Block,
)
from models.layers import HoGEdgeGateConv
from models.rsgdi_v2_modules import RSGDI_V2_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_triple_stack_v3_modes_are_unique_and_new():
    new_modes = {
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
    }
    ablation_modes = {
        "id_id_id",
        "msa_id_id",
        "id_ocv_id",
        "id_id_gbc",
        "msa_ocv_id",
        "msa_id_gbc",
        "id_ocv_gbc",
        "msa_ocv_gbc",
    }

    assert len(TRIPLE_STACK_V3_MODES) == 67
    assert len(set(TRIPLE_STACK_V3_MODES)) == 67
    assert new_modes.issubset(TRIPLE_STACK_V3_MODES)
    assert ablation_modes.issubset(TRIPLE_STACK_V3_MODES)
    assert set(TRIPLE_STACK_V3_MODES).isdisjoint(EXPERIMENTAL_MODULE_MODES)
    assert set(TRIPLE_STACK_V3_MODES).isdisjoint(SECOND_EXPERIMENTAL_MODULE_MODES)
    assert set(TRIPLE_STACK_V3_MODES).isdisjoint(THIRD_EXPERIMENTAL_MODULE_MODES)
    assert set(TRIPLE_STACK_V3_MODES).isdisjoint(PAPER_STACK_MODES)
    assert set(TRIPLE_STACK_V3_MODES).isdisjoint(DRSGI_MODES)
    assert set(TRIPLE_STACK_V3_MODES).isdisjoint(RSGDI_V2_MODES)


def test_triple_stack_v3_identity_ablation_points():
    module = TripleStackV3Block(channels=16, nbins=18, mode="id_id_id")

    assert isinstance(module.point1, nn.Identity)
    assert isinstance(module.point2, nn.Identity)
    assert isinstance(module.point3, nn.Identity)


def test_triple_stack_v3_modules_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V3_MODES:
        x = torch.randn(1, 16, 32, 32, requires_grad=True)
        module = TripleStackV3Block(channels=16, nbins=18, mode=mode)
        y = module(x)
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_encoder_inserts_triple_stack_v3_after_transmixer_and_wraps_degconv():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        mlp_ratio=2.0,
        nbins=18,
        use_triple_stack_v3=True,
        triple_stack_v3_mode="dsc_htm_gbc",
    )

    assert isinstance(vss.blocks[0][1], TripleStackV3Block)
    assert vss.blocks[0][1].mode == "dsc_htm_gbc"
    assert isinstance(vss.blocks[0][1].deg, HoGEdgeGateConv)


def test_encoder_rejects_triple_stack_v3_with_other_experimental_modules():
    with pytest.raises(ValueError, match="use_triple_stack_v3 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v3=True,
            triple_stack_v3_mode="dsc_htm_gbc",
            use_exp_module=True,
            exp_module_mode="scale_adaptive_fusion",
        )

    with pytest.raises(ValueError, match="use_paper_stack cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v3=True,
            triple_stack_v3_mode="dsc_htm_gbc",
            use_paper_stack=True,
            paper_stack_mode="saf_rgp_rgd",
        )


def test_parser_and_checkpoint_args_include_triple_stack_v3_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    main_source = (PROJECT_ROOT / "main.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v3"].default is False
    assert actions["triple_stack_v3_mode"].choices == TRIPLE_STACK_V3_MODES
    assert "--use_triple_stack_v3" in main_source
    assert "--triple_stack_v3_mode" in main_source
    assert "use_triple_stack_v3 -> " in main_source
    assert "triple_stack_v3_mode -> " in main_source
    assert "\"use_triple_stack_v3\"" in infer_source
    assert "\"triple_stack_v3_mode\"" in infer_source
    assert "TripleStack-v3 enabled" in profile_source
    assert "TripleStack-v3 mode" in profile_source
    assert "use_triple_stack_v3=getattr(args, 'use_triple_stack_v3', False)" in segmentor_source
