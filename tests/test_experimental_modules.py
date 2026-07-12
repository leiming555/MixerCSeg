from pathlib import Path

import torch

from models.experimental_modules import (
    EXPERIMENTAL_MODULE_MODES,
    ExperimentalEnhancementModule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_experimental_modules_preserve_shape_and_backward():
    for mode in EXPERIMENTAL_MODULE_MODES:
        x = torch.randn(1, 16, 13, 17, requires_grad=True)
        module = ExperimentalEnhancementModule(channels=16, mode=mode)
        y = module(x)
        loss = y.mean()
        loss.backward()

        assert y.shape == x.shape
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_experimental_module_modes_are_unique_and_complete():
    assert len(EXPERIMENTAL_MODULE_MODES) == 20
    assert len(set(EXPERIMENTAL_MODULE_MODES)) == 20


def test_encoder_and_segmentor_pass_experimental_module_flags():
    encoder_source = (PROJECT_ROOT / "models/encoder/vss_block.py").read_text()
    segmentor_source = (
        PROJECT_ROOT / "models/segmentor/MixerCSeg.py"
    ).read_text()
    main_source = (PROJECT_ROOT / "main.py").read_text()

    assert "from models.experimental_modules import ExperimentalEnhancementModule" in encoder_source
    assert "if use_exp_module:" in encoder_source
    assert "ExperimentalEnhancementModule(" in encoder_source
    assert "mode=exp_module_mode" in encoder_source
    assert "use_exp_module=use_exp_module" in encoder_source
    assert "exp_module_mode=exp_module_mode" in encoder_source
    assert "use_exp_module=getattr(args, 'use_exp_module', False)" in segmentor_source
    assert "exp_module_mode=getattr(args, 'exp_module_mode', 'win_attn_strip_gate')" in segmentor_source
    assert "--use_exp_module" in main_source
    assert "--exp_module_mode" in main_source
