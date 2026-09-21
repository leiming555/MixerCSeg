import argparse
from pathlib import Path

import pytest
import torch

from main import get_args_parser
from models.deg_ridge_scale_gap_interaction import (
    DRSGI_MODES,
    DEGRidgeScaleGapInteractionModule,
)
from models.encoder.vss_block import VSS
from models.experimental_modules import EXPERIMENTAL_MODULE_MODES
from models.experimental_paper_stack import PAPER_STACK_MODES
from models.experimental_second_modules import SECOND_EXPERIMENTAL_MODULE_MODES
from models.experimental_third_modules import THIRD_EXPERIMENTAL_MODULE_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_drsgi_modes_are_unique_and_new():
    assert len(DRSGI_MODES) == 8
    assert len(set(DRSGI_MODES)) == 8
    assert set(DRSGI_MODES).isdisjoint(EXPERIMENTAL_MODULE_MODES)
    assert set(DRSGI_MODES).isdisjoint(SECOND_EXPERIMENTAL_MODULE_MODES)
    assert set(DRSGI_MODES).isdisjoint(THIRD_EXPERIMENTAL_MODULE_MODES)
    assert set(DRSGI_MODES).isdisjoint(PAPER_STACK_MODES)


def test_drsgi_modules_preserve_shape_and_backward():
    for mode in DRSGI_MODES:
        x = torch.randn(1, 16, 32, 32, requires_grad=True)
        module = DEGRidgeScaleGapInteractionModule(channels=16, nbins=18, mode=mode)
        y = module(x)
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_encoder_replaces_degconv_with_drsgi_when_enabled():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        mlp_ratio=2.0,
        nbins=18,
        use_drsgi=True,
        drsgi_mode="pre_gate",
    )

    assert isinstance(vss.blocks[0][1], DEGRidgeScaleGapInteractionModule)
    assert vss.blocks[0][1].mode == "pre_gate"


def test_encoder_rejects_drsgi_with_other_experimental_modules():
    with pytest.raises(ValueError, match="use_drsgi cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_drsgi=True,
            drsgi_mode="pre_gate",
            use_paper_stack=True,
            paper_stack_mode="saf_rgp_rgd",
        )


def test_parser_and_checkpoint_args_include_drsgi_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    main_source = (PROJECT_ROOT / "main.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()

    assert actions["use_drsgi"].default is False
    assert actions["drsgi_mode"].choices == DRSGI_MODES
    assert "--use_drsgi" in main_source
    assert "--drsgi_mode" in main_source
    assert "use_drsgi -> " in main_source
    assert "drsgi_mode -> " in main_source
    assert "\"use_drsgi\"" in infer_source
    assert "\"drsgi_mode\"" in infer_source
    assert "DRSGI enabled" in profile_source
    assert "DRSGI mode" in profile_source
