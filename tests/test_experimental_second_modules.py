import argparse
from pathlib import Path

import torch

from main import get_args_parser
from models.experimental_modules import (
    EXPERIMENTAL_MODULE_MODES,
    ExperimentalEnhancementModule,
)
from models.experimental_second_modules import (
    SECOND_EXPERIMENTAL_MODULE_MODES,
    SecondExperimentalEnhancementModule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_second_experimental_modes_are_unique_and_new():
    assert len(SECOND_EXPERIMENTAL_MODULE_MODES) == 20
    assert len(set(SECOND_EXPERIMENTAL_MODULE_MODES)) == 20
    assert set(SECOND_EXPERIMENTAL_MODULE_MODES).isdisjoint(EXPERIMENTAL_MODULE_MODES)


def test_second_experimental_modules_preserve_shape_and_backward():
    for mode in SECOND_EXPERIMENTAL_MODULE_MODES:
        x = torch.randn(1, 16, 13, 17, requires_grad=True)
        module = SecondExperimentalEnhancementModule(channels=16, mode=mode)
        y = module(x)
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_scale_adaptive_then_second_module_sequence_preserves_shape():
    for mode in SECOND_EXPERIMENTAL_MODULE_MODES:
        x = torch.randn(1, 16, 13, 17, requires_grad=True)
        scale_module = ExperimentalEnhancementModule(channels=16, mode="scale_adaptive_fusion")
        second_module = SecondExperimentalEnhancementModule(channels=16, mode=mode)

        y = second_module(scale_module(x))
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_encoder_orders_second_module_after_scale_module_and_before_ccem():
    encoder_source = (PROJECT_ROOT / "models/encoder/vss_block.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert "from models.experimental_second_modules import SecondExperimentalEnhancementModule" in encoder_source
    assert encoder_source.index("if use_exp_module:") < encoder_source.index("if use_exp_second_module:")
    assert encoder_source.index("if use_exp_second_module:") < encoder_source.index("if use_ccem:")
    assert "exp_module_mode != \"scale_adaptive_fusion\"" in encoder_source
    assert "SecondExperimentalEnhancementModule(" in encoder_source
    assert "mode=exp_second_module_mode" in encoder_source
    assert "use_exp_second_module=use_exp_second_module" in encoder_source
    assert "exp_second_module_mode=exp_second_module_mode" in encoder_source
    assert "use_exp_second_module=getattr(args, 'use_exp_second_module', False)" in segmentor_source
    assert "exp_second_module_mode=getattr(args, 'exp_second_module_mode', 'soft_morph_gradient_gate')" in segmentor_source


def test_parser_and_checkpoint_args_include_second_module_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    main_source = (PROJECT_ROOT / "main.py").read_text()

    assert actions["use_exp_second_module"].default is False
    assert actions["exp_second_module_mode"].choices == SECOND_EXPERIMENTAL_MODULE_MODES
    assert "--use_exp_second_module" in main_source
    assert "--exp_second_module_mode" in main_source
    assert "use_exp_second_module -> " in main_source
    assert "exp_second_module_mode -> " in main_source
    assert "\"use_exp_second_module\"" in infer_source
    assert "\"exp_second_module_mode\"" in infer_source
