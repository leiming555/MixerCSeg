import argparse
from pathlib import Path

import pytest
import torch

from main import get_args_parser
from models.encoder.vss_block import VSS
from models.experimental_modules import EXPERIMENTAL_MODULE_MODES
from models.experimental_paper_stack import (
    PAPER_STACK_MODES,
    PaperStackEnhancementModule,
)
from models.experimental_second_modules import SECOND_EXPERIMENTAL_MODULE_MODES
from models.experimental_third_modules import THIRD_EXPERIMENTAL_MODULE_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_paper_stack_modes_are_unique_and_new():
    extra_modes = {
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
    }

    assert len(PAPER_STACK_MODES) == 50
    assert len(set(PAPER_STACK_MODES)) == 50
    assert extra_modes.issubset(PAPER_STACK_MODES)
    assert set(PAPER_STACK_MODES).isdisjoint(EXPERIMENTAL_MODULE_MODES)
    assert set(PAPER_STACK_MODES).isdisjoint(SECOND_EXPERIMENTAL_MODULE_MODES)
    assert set(PAPER_STACK_MODES).isdisjoint(THIRD_EXPERIMENTAL_MODULE_MODES)


def test_paper_stack_modules_preserve_shape_and_backward():
    for mode in PAPER_STACK_MODES:
        x = torch.randn(1, 16, 13, 17, requires_grad=True)
        module = PaperStackEnhancementModule(channels=16, mode=mode)
        y = module(x)
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_encoder_inserts_paper_stack_after_degconv_and_before_exp_modules():
    encoder_source = (PROJECT_ROOT / "models/encoder/vss_block.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert "from models.experimental_paper_stack import PaperStackEnhancementModule" in encoder_source
    assert encoder_source.index("HoGEdgeGateConv(") < encoder_source.index("if use_paper_stack:")
    assert encoder_source.index("if use_paper_stack:") < encoder_source.index("if use_edrm:")
    assert encoder_source.index("if use_paper_stack:") < encoder_source.index("if use_exp_module:")
    assert "PaperStackEnhancementModule(" in encoder_source
    assert "mode=paper_stack_mode" in encoder_source
    assert "use_paper_stack=use_paper_stack" in encoder_source
    assert "paper_stack_mode=paper_stack_mode" in encoder_source
    assert "use_paper_stack=getattr(args, 'use_paper_stack', False)" in segmentor_source
    assert "paper_stack_mode=getattr(args, 'paper_stack_mode', 'saf_rgp')" in segmentor_source


def test_encoder_rejects_paper_stack_with_exp_modules():
    with pytest.raises(ValueError, match="use_paper_stack cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            use_paper_stack=True,
            paper_stack_mode="saf_rgp",
            use_exp_module=True,
            exp_module_mode="scale_adaptive_fusion",
        )


def test_parser_and_checkpoint_args_include_paper_stack_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    main_source = (PROJECT_ROOT / "main.py").read_text()

    assert actions["use_paper_stack"].default is False
    assert actions["paper_stack_mode"].choices == PAPER_STACK_MODES
    assert "--use_paper_stack" in main_source
    assert "--paper_stack_mode" in main_source
    assert "use_paper_stack -> " in main_source
    assert "paper_stack_mode -> " in main_source
    assert "\"use_paper_stack\"" in infer_source
    assert "\"paper_stack_mode\"" in infer_source
