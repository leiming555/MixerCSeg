import argparse
from pathlib import Path

import pytest
import torch

from main import get_args_parser
from models.encoder.vss_block import VSS
from models.experimental_modules import (
    EXPERIMENTAL_MODULE_MODES,
    ExperimentalEnhancementModule,
)
from models.experimental_second_modules import (
    SECOND_EXPERIMENTAL_MODULE_MODES,
    SecondExperimentalEnhancementModule,
)
from models.experimental_third_modules import (
    THIRD_EXPERIMENTAL_MODULE_MODES,
    ThirdExperimentalEnhancementModule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_third_experimental_modes_are_unique_and_new():
    assert len(THIRD_EXPERIMENTAL_MODULE_MODES) == 20
    assert len(set(THIRD_EXPERIMENTAL_MODULE_MODES)) == 20
    assert set(THIRD_EXPERIMENTAL_MODULE_MODES).isdisjoint(EXPERIMENTAL_MODULE_MODES)
    assert set(THIRD_EXPERIMENTAL_MODULE_MODES).isdisjoint(SECOND_EXPERIMENTAL_MODULE_MODES)


def test_third_experimental_modules_preserve_shape_and_backward():
    for mode in THIRD_EXPERIMENTAL_MODULE_MODES:
        x = torch.randn(1, 16, 13, 17, requires_grad=True)
        module = ThirdExperimentalEnhancementModule(channels=16, mode=mode)
        y = module(x)
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_scale_then_ridge_then_third_sequence_preserves_shape():
    for mode in THIRD_EXPERIMENTAL_MODULE_MODES:
        x = torch.randn(1, 16, 13, 17, requires_grad=True)
        scale_module = ExperimentalEnhancementModule(channels=16, mode="scale_adaptive_fusion")
        ridge_module = SecondExperimentalEnhancementModule(channels=16, mode="ridge_hessian_context")
        third_module = ThirdExperimentalEnhancementModule(channels=16, mode=mode)

        y = third_module(ridge_module(scale_module(x)))
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_encoder_orders_third_module_after_ridge_and_before_ccem():
    encoder_source = (PROJECT_ROOT / "models/encoder/vss_block.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert "from models.experimental_third_modules import ThirdExperimentalEnhancementModule" in encoder_source
    assert encoder_source.index("if use_exp_second_module:") < encoder_source.index("if use_exp_third_module:")
    assert encoder_source.index("if use_exp_third_module:") < encoder_source.index("if use_ccem:")
    assert "exp_module_mode != \"scale_adaptive_fusion\"" in encoder_source
    assert "exp_second_module_mode != \"ridge_hessian_context\"" in encoder_source
    assert "ThirdExperimentalEnhancementModule(" in encoder_source
    assert "mode=exp_third_module_mode" in encoder_source
    assert "use_exp_third_module=use_exp_third_module" in encoder_source
    assert "exp_third_module_mode=exp_third_module_mode" in encoder_source
    assert "use_exp_third_module=getattr(args, 'use_exp_third_module', False)" in segmentor_source
    assert "exp_third_module_mode=getattr(args, 'exp_third_module_mode', 'hessian_eigen_bridge_gate')" in segmentor_source


def test_encoder_rejects_third_module_without_ridge_second_module():
    with pytest.raises(ValueError, match="use_exp_third_module requires"):
        VSS(
            in_dim=16,
            depth=1,
            use_exp_module=True,
            exp_module_mode="scale_adaptive_fusion",
            use_exp_second_module=True,
            exp_second_module_mode="soft_morph_gradient_gate",
            use_exp_third_module=True,
            exp_third_module_mode="hessian_eigen_bridge_gate",
        )


def test_parser_and_checkpoint_args_include_third_module_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    main_source = (PROJECT_ROOT / "main.py").read_text()

    assert actions["use_exp_third_module"].default is False
    assert actions["exp_third_module_mode"].choices == THIRD_EXPERIMENTAL_MODULE_MODES
    assert "--use_exp_third_module" in main_source
    assert "--exp_third_module_mode" in main_source
    assert "use_exp_third_module -> " in main_source
    assert "exp_third_module_mode -> " in main_source
    assert "\"use_exp_third_module\"" in infer_source
    assert "\"exp_third_module_mode\"" in infer_source
