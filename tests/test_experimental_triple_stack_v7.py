import argparse
from pathlib import Path

import pytest
import torch

from main import get_args_parser
from models.encoder.vss_block import VSS
from models.experimental_triple_stack_v3 import (
    GapBoundaryConfidence,
    MultiScaleAxialConditioner,
    OrientationCurvatureVoting,
    TRIPLE_STACK_V3_MODES,
)
from models.experimental_triple_stack_v4 import TRIPLE_STACK_V4_MODES
from models.experimental_triple_stack_v5 import TRIPLE_STACK_V5_MODES
from models.experimental_triple_stack_v6 import TRIPLE_STACK_V6_MODES
from models.experimental_triple_stack_v7 import (
    CrossAxisConsensusMSA,
    CurvatureTopologyVoting,
    DualGatedBoundaryBridge,
    MultiScaleEigenVoting,
    ProgressiveBridgeCalibrator,
    ReliabilityRoutedMSA,
    TRIPLE_STACK_V7_ABLATION_MODES,
    TRIPLE_STACK_V7_MODES,
    TRIPLE_STACK_V7_SEARCH_MODES,
    TRIPLE_STACK_V7_REMOVAL_MODES,
    TripleStackV7Block,
    WidthAdaptiveMSA,
    _shift_zero,
)
from models.layers import HoGEdgeGateConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_triple_stack_v7_modes_are_unique_and_new():
    assert len(TRIPLE_STACK_V7_SEARCH_MODES) == 20
    assert len(TRIPLE_STACK_V7_ABLATION_MODES) == 6
    assert len(TRIPLE_STACK_V7_MODES) == 30
    assert len(set(TRIPLE_STACK_V7_MODES)) == 30
    assert set(TRIPLE_STACK_V7_SEARCH_MODES).isdisjoint(TRIPLE_STACK_V7_ABLATION_MODES)
    assert set(TRIPLE_STACK_V7_MODES).intersection(TRIPLE_STACK_V3_MODES) == {"id_id_id"}
    assert set(TRIPLE_STACK_V7_MODES).isdisjoint(TRIPLE_STACK_V4_MODES)
    assert set(TRIPLE_STACK_V7_MODES).isdisjoint(TRIPLE_STACK_V5_MODES)
    assert set(TRIPLE_STACK_V7_MODES).isdisjoint(TRIPLE_STACK_V6_MODES)


def test_triple_stack_v7_modes_cover_five_by_two_by_two_grid():
    combinations = {tuple(mode.split("_")) for mode in TRIPLE_STACK_V7_SEARCH_MODES}
    assert {item[0] for item in combinations} == {"rma", "wma", "fma", "tma", "cma"}
    assert {item[1] for item in combinations} == {"mev", "ctv"}
    assert {item[2] for item in combinations} == {"dgb", "pbc"}
    assert len(combinations) == 5 * 2 * 2


@pytest.mark.parametrize("mode", TRIPLE_STACK_V7_REMOVAL_MODES)
def test_removal_modes_use_identity_and_keep_wrapper(mode):
    block = TripleStackV7Block(16, 18, mode)
    for index, name in enumerate(mode.split("_"), start=1):
        assert isinstance(getattr(block, f"point{index}"), torch.nn.Identity) == (name == "id")
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.final_gate, torch.nn.Conv2d)
    assert not isinstance(block.final_norm, torch.nn.Identity)


def test_triple_stack_v7_ablation_modes_map_to_expected_components():
    expected = {
        "wma_ocv_gbc": (WidthAdaptiveMSA, OrientationCurvatureVoting, GapBoundaryConfidence),
        "msa_ctv_gbc": (MultiScaleAxialConditioner, CurvatureTopologyVoting, GapBoundaryConfidence),
        "msa_ocv_dgb": (MultiScaleAxialConditioner, OrientationCurvatureVoting, DualGatedBoundaryBridge),
        "wma_ctv_gbc": (WidthAdaptiveMSA, CurvatureTopologyVoting, GapBoundaryConfidence),
        "wma_ocv_dgb": (WidthAdaptiveMSA, OrientationCurvatureVoting, DualGatedBoundaryBridge),
        "msa_ctv_dgb": (MultiScaleAxialConditioner, CurvatureTopologyVoting, DualGatedBoundaryBridge),
    }
    assert set(expected) == set(TRIPLE_STACK_V7_ABLATION_MODES)

    for mode, component_types in expected.items():
        block = TripleStackV7Block(channels=16, nbins=18, mode=mode)
        assert isinstance(block.point1, component_types[0])
        assert isinstance(block.deg, HoGEdgeGateConv)
        assert isinstance(block.point2, component_types[1])
        assert isinstance(block.point3, component_types[2])


def test_triple_stack_v7_modules_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V7_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        module = TripleStackV7Block(channels=16, nbins=18, mode=mode)
        y = module(x)
        y.mean().backward()

        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_triple_stack_v7_keeps_best_base_chain_around_original_degconv():
    block = TripleStackV7Block(channels=16, nbins=18, mode="rma_mev_dgb")
    assert isinstance(block.point1, ReliabilityRoutedMSA)
    assert isinstance(block.point1.base, MultiScaleAxialConditioner)
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.point2, MultiScaleEigenVoting)
    assert isinstance(block.point2.base, OrientationCurvatureVoting)
    assert isinstance(block.point3, DualGatedBoundaryBridge)
    assert isinstance(block.point3.base, GapBoundaryConfidence)

    cross_block = TripleStackV7Block(channels=16, nbins=18, mode="cma_ctv_pbc")
    assert isinstance(cross_block.point1, CrossAxisConsensusMSA)
    assert isinstance(cross_block.point2, CurvatureTopologyVoting)
    assert isinstance(cross_block.point3, ProgressiveBridgeCalibrator)


def test_zero_padded_shift_does_not_wrap_boundaries():
    x = torch.zeros(1, 1, 4, 4)
    x[..., 0, 0] = 1.0
    shifted_right = _shift_zero(x, 0, 1)
    shifted_down = _shift_zero(x, 1, 0)
    assert shifted_right[..., 0, 1].item() == 1.0
    assert shifted_right[..., 0, 0].item() == 0.0
    assert shifted_down[..., 1, 0].item() == 1.0
    assert shifted_down[..., 0, 0].item() == 0.0


def test_encoder_inserts_triple_stack_v7_after_transmixer():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        nbins=18,
        use_triple_stack_v7=True,
        triple_stack_v7_mode="rma_mev_dgb",
    )
    assert isinstance(vss.blocks[0][1], TripleStackV7Block)
    assert isinstance(vss.blocks[0][1].deg, HoGEdgeGateConv)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_triple_stack_v3": True, "triple_stack_v3_mode": "msa_ocv_gbc"},
        {"use_triple_stack_v4": True, "triple_stack_v4_mode": "adc_eov_pgs"},
        {"use_triple_stack_v5": True, "triple_stack_v5_mode": "fap_rtc_upb"},
        {"use_triple_stack_v6": True, "triple_stack_v6_mode": "hfr_ert_apb"},
        {"use_ccem": True},
        {"use_paper_stack": True},
    ],
)
def test_encoder_rejects_triple_stack_v7_with_other_experimental_modules(kwargs):
    with pytest.raises(ValueError, match="use_triple_stack_v7 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v7=True,
            triple_stack_v7_mode="rma_mev_dgb",
            **kwargs,
        )


def test_parser_and_checkpoint_args_include_triple_stack_v7_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v7"].default is False
    assert actions["triple_stack_v7_mode"].choices == TRIPLE_STACK_V7_MODES
    assert '"use_triple_stack_v7"' in infer_source
    assert '"triple_stack_v7_mode"' in infer_source
    assert "TripleStack-v7 enabled" in profile_source
    assert "TripleStack-v7 mode" in profile_source
    assert "use_triple_stack_v7=getattr(args, 'use_triple_stack_v7', False)" in segmentor_source
