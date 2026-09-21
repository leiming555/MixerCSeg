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
from models.layers import HoGEdgeGateConv
from models.rsgdi_v2_modules import RSGDI_V2_MODES, RSGDIV2EnhancementModule


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_rsgdi_v2_modes_are_unique_and_new():
    assert len(RSGDI_V2_MODES) == 20
    assert len(set(RSGDI_V2_MODES)) == 20
    assert set(RSGDI_V2_MODES).isdisjoint(EXPERIMENTAL_MODULE_MODES)
    assert set(RSGDI_V2_MODES).isdisjoint(SECOND_EXPERIMENTAL_MODULE_MODES)
    assert set(RSGDI_V2_MODES).isdisjoint(THIRD_EXPERIMENTAL_MODULE_MODES)
    assert set(RSGDI_V2_MODES).isdisjoint(PAPER_STACK_MODES)
    assert set(RSGDI_V2_MODES).isdisjoint(DRSGI_MODES)


def test_rsgdi_v2_modules_preserve_shape_and_backward():
    for mode in RSGDI_V2_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        module = RSGDIV2EnhancementModule(channels=16, mode=mode)
        y = module(x)
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_encoder_inserts_rsgdi_v2_after_degconv():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        mlp_ratio=2.0,
        nbins=18,
        use_rsgdi_v2=True,
        rsgdi_v2_mode="topology_bridge_gate",
    )

    assert isinstance(vss.blocks[0][1], HoGEdgeGateConv)
    assert isinstance(vss.blocks[0][2], RSGDIV2EnhancementModule)
    assert vss.blocks[0][2].mode == "topology_bridge_gate"


def test_encoder_rejects_rsgdi_v2_with_other_experimental_modules():
    with pytest.raises(ValueError, match="use_rsgdi_v2 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_rsgdi_v2=True,
            rsgdi_v2_mode="topology_bridge_gate",
            use_paper_stack=True,
            paper_stack_mode="saf_rgp_rgd",
        )


def test_parser_and_checkpoint_args_include_rsgdi_v2_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    main_source = (PROJECT_ROOT / "main.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_rsgdi_v2"].default is False
    assert actions["rsgdi_v2_mode"].choices == RSGDI_V2_MODES
    assert "--use_rsgdi_v2" in main_source
    assert "--rsgdi_v2_mode" in main_source
    assert "use_rsgdi_v2 -> " in main_source
    assert "rsgdi_v2_mode -> " in main_source
    assert "\"use_rsgdi_v2\"" in infer_source
    assert "\"rsgdi_v2_mode\"" in infer_source
    assert "RSGDI-v2 enabled" in profile_source
    assert "RSGDI-v2 mode" in profile_source
    assert "use_rsgdi_v2=getattr(args, 'use_rsgdi_v2', False)" in segmentor_source
