import argparse
from pathlib import Path

import pytest
import torch

from main import get_args_parser
from models.encoder.vss_block import VSS
from models.experimental_triple_stack_v3 import TRIPLE_STACK_V3_MODES
from models.experimental_triple_stack_v4 import TRIPLE_STACK_V4_MODES
from models.experimental_triple_stack_v5 import TRIPLE_STACK_V5_MODES
from models.experimental_triple_stack_v6 import TRIPLE_STACK_V6_MODES
from models.experimental_triple_stack_v7 import (
    CurvatureTopologyVoting,
    DualGatedBoundaryBridge,
    TRIPLE_STACK_V7_MODES,
    WidthAdaptiveMSA,
)
from models.experimental_triple_stack_v8 import TRIPLE_STACK_V8_MODES
from models.experimental_triple_stack_v9 import TRIPLE_STACK_V9_MODES
from models.experimental_triple_stack_v10 import (
    AgreementReliabilityDecomposer,
    AnchoredCalibrationState,
    BudgetedPrecisionRecallGuard,
    DistributionMatchedCalibrator,
    DualPathBridgeRouter,
    PathConsensusRouter,
    TopologyConsistencyDecomposer,
    TRIPLE_STACK_V10_MODES,
    TripleStackV10Block,
    WidthFrequencyDecomposer,
    _shift_zero,
)
from models.layers import HoGEdgeGateConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_triple_stack_v10_modes_are_unique_and_new():
    previous_modes = (
        set(TRIPLE_STACK_V3_MODES)
        | set(TRIPLE_STACK_V4_MODES)
        | set(TRIPLE_STACK_V5_MODES)
        | set(TRIPLE_STACK_V6_MODES)
        | set(TRIPLE_STACK_V7_MODES)
        | set(TRIPLE_STACK_V8_MODES)
        | set(TRIPLE_STACK_V9_MODES)
    )
    assert len(TRIPLE_STACK_V10_MODES) == 12
    assert len(set(TRIPLE_STACK_V10_MODES)) == 12
    assert set(TRIPLE_STACK_V10_MODES).isdisjoint(previous_modes)


def test_triple_stack_v10_modes_cover_three_by_two_by_two_grid():
    combinations = {tuple(mode.split("_")) for mode in TRIPLE_STACK_V10_MODES}
    assert {item[0] for item in combinations} == {"ard", "wfd", "tcd"}
    assert {item[1] for item in combinations} == {"dbr", "pcr"}
    assert {item[2] for item in combinations} == {"bpg", "dmc"}
    assert len(combinations) == 3 * 2 * 2


@pytest.mark.parametrize(
    "mode,component_types",
    [
        (
            "ard_dbr_bpg",
            (AgreementReliabilityDecomposer, DualPathBridgeRouter, BudgetedPrecisionRecallGuard),
        ),
        (
            "wfd_pcr_dmc",
            (WidthFrequencyDecomposer, PathConsensusRouter, DistributionMatchedCalibrator),
        ),
        (
            "tcd_dbr_dmc",
            (TopologyConsistencyDecomposer, DualPathBridgeRouter, DistributionMatchedCalibrator),
        ),
    ],
)
def test_triple_stack_v10_mapping_and_v7_anchor(mode, component_types):
    block = TripleStackV10Block(channels=16, nbins=18, mode=mode)
    assert isinstance(block.wma, WidthAdaptiveMSA)
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.ctv, CurvatureTopologyVoting)
    assert isinstance(block.dgb, DualGatedBoundaryBridge)
    assert isinstance(block.decomposer, component_types[0])
    assert isinstance(block.router, component_types[1])
    assert isinstance(block.calibrator, component_types[2])


@pytest.mark.parametrize(
    "decomposer_type",
    [AgreementReliabilityDecomposer, WidthFrequencyDecomposer, TopologyConsistencyDecomposer],
)
def test_decomposer_builds_finite_anchored_state(decomposer_type):
    tensors = [torch.randn(2, 16, 12, 10) for _ in range(4)]
    state = decomposer_type(16)(*tensors)
    assert isinstance(state, AnchoredCalibrationState)
    assert state.base_features.shape == tensors[-1].shape
    assert state.weak_evidence.shape == tensors[-1].shape
    assert state.reliability.shape == (2, 1, 12, 10)
    assert state.direction_probabilities.shape == (2, 8, 12, 10)
    assert torch.allclose(
        state.direction_probabilities.sum(dim=1),
        torch.ones(2, 12, 10),
        atol=1e-6,
    )
    assert torch.isfinite(state.weak_evidence).all()
    assert state.reliability.min() >= 0.0
    assert state.reliability.max() <= 1.0


@pytest.mark.parametrize("router_type", [DualPathBridgeRouter, PathConsensusRouter])
@pytest.mark.parametrize(
    "calibrator_type",
    [BudgetedPrecisionRecallGuard, DistributionMatchedCalibrator],
)
def test_state_flows_through_router_and_calibrator(router_type, calibrator_type):
    tensors = [torch.randn(1, 16, 12, 10) for _ in range(4)]
    state = AgreementReliabilityDecomposer(16)(*tensors)
    state = router_type(16)(state)
    assert state.direction_probabilities.shape == (1, 8, 12, 10)
    assert torch.allclose(
        state.direction_probabilities.sum(dim=1),
        torch.ones(1, 12, 10),
        atol=1e-5,
    )
    assert state.bridge_delta.shape == tensors[-1].shape
    assert state.bridge_confidence.shape == (1, 1, 12, 10)
    assert state.bridge_confidence.min() >= 0.0
    assert state.bridge_confidence.max() <= 1.0

    state = calibrator_type(16)(state)
    assert state.correction.shape == tensors[-1].shape
    assert state.correction_budget.shape == (1, 1, 12, 10)
    assert torch.isfinite(state.correction).all()
    assert state.correction_budget.min() >= 0.0
    assert state.correction_budget.max() <= 1.0


def test_triple_stack_v10_modes_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V10_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        module = TripleStackV10Block(channels=16, nbins=18, mode=mode)
        y = module(x)
        y.mean().backward()
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_zero_sidecar_gamma_matches_v7_anchor_exactly():
    block = TripleStackV10Block(channels=16, nbins=18, mode="ard_dbr_bpg")
    block.eval()
    x = torch.randn(1, 16, 16, 16)
    with torch.no_grad():
        anchored = block.forward_v7_anchor(x)
        block.sidecar_gamma.zero_()
        output = block(x)
    assert torch.equal(output, anchored)


def test_zero_padded_shift_does_not_wrap_boundaries():
    x = torch.zeros(1, 1, 4, 4)
    x[..., 0, 0] = 1.0
    shifted_right = _shift_zero(x, 0, 1)
    shifted_down = _shift_zero(x, 1, 0)
    assert shifted_right[..., 0, 1].item() == 1.0
    assert shifted_right[..., 0, 0].item() == 0.0
    assert shifted_down[..., 1, 0].item() == 1.0
    assert shifted_down[..., 0, 0].item() == 0.0


def test_encoder_inserts_triple_stack_v10_after_transmixer():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        nbins=18,
        use_triple_stack_v10=True,
        triple_stack_v10_mode="ard_dbr_bpg",
    )
    assert isinstance(vss.blocks[0][1], TripleStackV10Block)
    assert isinstance(vss.blocks[0][1].deg, HoGEdgeGateConv)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_triple_stack_v3": True, "triple_stack_v3_mode": "msa_ocv_gbc"},
        {"use_triple_stack_v4": True, "triple_stack_v4_mode": "adc_eov_pgs"},
        {"use_triple_stack_v5": True, "triple_stack_v5_mode": "fap_rtc_upb"},
        {"use_triple_stack_v6": True, "triple_stack_v6_mode": "hfr_ert_apb"},
        {"use_triple_stack_v7": True, "triple_stack_v7_mode": "wma_ctv_dgb"},
        {"use_triple_stack_v8": True, "triple_stack_v8_mode": "cwc_lsf_obp"},
        {"use_triple_stack_v9": True, "triple_stack_v9_mode": "eub_oeb_prg"},
        {"use_ccem": True},
        {"use_paper_stack": True},
        {"use_exp_module": True},
    ],
)
def test_encoder_rejects_v10_with_other_experimental_modules(kwargs):
    with pytest.raises(ValueError, match="use_triple_stack_v10 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v10=True,
            triple_stack_v10_mode="ard_dbr_bpg",
            **kwargs,
        )


def test_parser_and_checkpoint_args_include_v10_options():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v10"].default is False
    assert actions["triple_stack_v10_mode"].choices == TRIPLE_STACK_V10_MODES
    assert '"use_triple_stack_v10"' in infer_source
    assert '"triple_stack_v10_mode"' in infer_source
    assert "TripleStack-v10 enabled" in profile_source
    assert "TripleStack-v10 mode" in profile_source
    assert "use_triple_stack_v10=getattr(args, 'use_triple_stack_v10', False)" in segmentor_source


def test_default_encoder_path_remains_original_degconv():
    vss = VSS(in_dim=16, depth=1, state_dim=8, nbins=18)
    assert isinstance(vss.blocks[0][1], HoGEdgeGateConv)
