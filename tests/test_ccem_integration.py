import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_encoder_block_orders_transmixer_degconv_enhancement():
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

    module_names = [
        arg.func.id
        for arg in sequential.args
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
    ]

    assert module_names == [
        "TransMixer",
        "HoGEdgeGateConv",
        "enhancement_module",
    ]


def test_encoder_routes_ccem_to_f1_f2_and_mscm_to_f3_f4():
    source = (PROJECT_ROOT / "models/encoder/vss_block.py").read_text()

    assert "from models.mscm import MultiScaleContextModule" in source
    assert "if i_layer < 2" in source
    assert "CrackContinuityEnhancementModule" in source
    assert "MultiScaleContextModule" in source
    assert "enhancement_module=enhancement_module" in source


def test_encoder_outputs_feed_srf_then_segmentation_head():
    segmentor_source = (
        PROJECT_ROOT / "models/segmentor/MixerCSeg.py"
    ).read_text()
    decoder_source = (PROJECT_ROOT / "models/decoder/SRF.py").read_text()

    assert "outs = self.backbone(samples)" in segmentor_source
    assert "out = self.decoder(outs)" in segmentor_source
    assert "BoundaryRefinementModule" not in decoder_source
    assert "x = self.brm(x)" not in decoder_source
    assert "x = self.linear_pred(x)" in decoder_source
