import ast
from pathlib import Path
import torch

from models.ccem import CrackContinuityEnhancementModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ccem_modes_preserve_shape():
    x = torch.randn(1, 16, 16, 16)

    for mode in ["full", "no_local", "no_strip", "no_dilation", "no_gate"]:
        module = CrackContinuityEnhancementModule(channels=16, mode=mode)
        y = module(x)
        assert y.shape == x.shape


def test_encoder_optionally_adds_ccem_after_degconv():
    source = (PROJECT_ROOT / "models/encoder/vss_block.py").read_text()
    tree = ast.parse(source)

    vss_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VSS"
    )
    sequential = next(
        node
        for node in ast.walk(vss_class)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Sequential"
    )

    assert any(isinstance(arg, ast.Starred) for arg in sequential.args)
    assert "if use_ccem:" in source
    assert "CrackContinuityEnhancementModule(channels=in_dim, mode=ccem_mode)" in source


def test_encoder_passes_ccem_flags_from_args():
    source = (PROJECT_ROOT / "models/encoder/vss_block.py").read_text()
    segmentor_source = (
        PROJECT_ROOT / "models/segmentor/MixerCSeg.py"
    ).read_text()

    assert "use_ccem=use_ccem" in source
    assert "ccem_mode=ccem_mode" in source
    assert "use_ccem=getattr(args, 'use_ccem', False)" in segmentor_source
    assert "ccem_mode=getattr(args, 'ccem_mode', 'full')" in segmentor_source
    assert "MultiScaleContextModule" not in source


def test_encoder_outputs_feed_srf_then_segmentation_head():
    segmentor_source = (
        PROJECT_ROOT / "models/segmentor/MixerCSeg.py"
    ).read_text()
    decoder_source = (PROJECT_ROOT / "models/decoder/SRF.py").read_text()

    assert "outs = self.backbone(samples)" in segmentor_source
    assert "out = self.decoder(outs)" in segmentor_source
    assert "BoundaryRefinementModule" not in decoder_source
    assert "x = self.brm(x)" not in decoder_source
    assert "SkeletonGuidanceHead" not in decoder_source
    assert "x = self.linear_pred(x)" in decoder_source
