import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from main import get_args_parser, main
from models.encoder.vss_block import VSS
from models.experimental_triple_stack_v3 import TRIPLE_STACK_V3_MODES
from models.experimental_triple_stack_v4 import TRIPLE_STACK_V4_MODES
from models.experimental_triple_stack_v5 import TRIPLE_STACK_V5_MODES
from models.experimental_triple_stack_v6 import TRIPLE_STACK_V6_MODES
from models.experimental_triple_stack_v7 import TRIPLE_STACK_V7_MODES
from models.experimental_triple_stack_v8 import TRIPLE_STACK_V8_MODES
from models.experimental_triple_stack_v9 import TRIPLE_STACK_V9_MODES
from models.experimental_triple_stack_v10 import TRIPLE_STACK_V10_MODES
from models.experimental_triple_stack_v11 import (
    ConsensusGuidedBoundaryBridge,
    CoupledPriorState,
    FrequencyPreservedMSA,
    LocalWidthMSA,
    PairwiseCurvatureVoting,
    PrecisionBudgetSuppressor,
    ReliabilityAwareMSA,
    ReliabilityCalibratedVoting,
    ScaleConsensusMSA,
    TRIPLE_STACK_V11_CONDITIONERS,
    TRIPLE_STACK_V11_MODES,
    TRIPLE_STACK_V11_REFINERS,
    TRIPLE_STACK_V11_VOTERS,
    TripleStackV11Block,
    _shift_zero,
)
from models.layers import HoGEdgeGateConv
from tools.summarize_triple_stack_v11 import assigned_gpu, collect_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE = PROJECT_ROOT / "logs/triple_stack_v11_crackmap_full16_valselect_gpu12_queue.sh"


def test_v11_modes_are_unique_new_and_cover_grid():
    previous = set().union(
        TRIPLE_STACK_V3_MODES,
        TRIPLE_STACK_V4_MODES,
        TRIPLE_STACK_V5_MODES,
        TRIPLE_STACK_V6_MODES,
        TRIPLE_STACK_V7_MODES,
        TRIPLE_STACK_V8_MODES,
        TRIPLE_STACK_V9_MODES,
        TRIPLE_STACK_V10_MODES,
    )
    combinations = {tuple(mode.split("_")) for mode in TRIPLE_STACK_V11_MODES}
    assert len(TRIPLE_STACK_V11_MODES) == 16
    assert len(set(TRIPLE_STACK_V11_MODES)) == 16
    assert set(TRIPLE_STACK_V11_MODES).isdisjoint(previous)
    assert {parts[0] for parts in combinations} == set(TRIPLE_STACK_V11_CONDITIONERS)
    assert {parts[1] for parts in combinations} == set(TRIPLE_STACK_V11_VOTERS)
    assert {parts[2] for parts in combinations} == set(TRIPLE_STACK_V11_REFINERS)


@pytest.mark.parametrize(
    "mode,types",
    [
        (
            "ram_rcv_cgb",
            (ReliabilityAwareMSA, ReliabilityCalibratedVoting, ConsensusGuidedBoundaryBridge),
        ),
        (
            "scm_pcv_pbs",
            (ScaleConsensusMSA, PairwiseCurvatureVoting, PrecisionBudgetSuppressor),
        ),
        (
            "fpm_rcv_pbs",
            (FrequencyPreservedMSA, ReliabilityCalibratedVoting, PrecisionBudgetSuppressor),
        ),
        (
            "lwm_pcv_cgb",
            (LocalWidthMSA, PairwiseCurvatureVoting, ConsensusGuidedBoundaryBridge),
        ),
    ],
)
def test_v11_component_mapping_and_original_deg(mode, types):
    block = TripleStackV11Block(channels=16, nbins=18, mode=mode)
    assert isinstance(block.conditioner, types[0])
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.voter, types[1])
    assert isinstance(block.refiner, types[2])


@pytest.mark.parametrize(
    "conditioner_type",
    [ReliabilityAwareMSA, ScaleConsensusMSA, FrequencyPreservedMSA, LocalWidthMSA],
)
def test_conditioners_return_finite_features_and_reliability(conditioner_type):
    x = torch.randn(2, 16, 14, 12)
    state = conditioner_type(16)(x)
    assert state.features.shape == x.shape
    assert state.reliability.shape == (2, 1, 14, 12)
    assert torch.isfinite(state.features).all()
    assert torch.isfinite(state.reliability).all()
    assert state.reliability.min() >= 0.0
    assert state.reliability.max() <= 1.0


@pytest.mark.parametrize("voter_type", [ReliabilityCalibratedVoting, PairwiseCurvatureVoting])
def test_voters_consume_scale_state_and_normalize_orientation(voter_type):
    conditioned = torch.randn(2, 16, 14, 12)
    deg = torch.randn(2, 16, 14, 12)
    reliability = torch.rand(2, 1, 14, 12)
    state = voter_type(16)(conditioned, reliability, deg)
    assert isinstance(state, CoupledPriorState)
    assert state.voted_features.shape == deg.shape
    assert state.orientation_weights.shape == (2, 4, 14, 12)
    assert torch.allclose(
        state.orientation_weights.sum(dim=1),
        torch.ones(2, 14, 12),
        atol=1e-5,
    )
    assert state.orientation_reliability.shape == reliability.shape
    assert state.orientation_reliability.min() >= 0.0
    assert state.orientation_reliability.max() <= 1.0


@pytest.mark.parametrize(
    "refiner_type", [ConsensusGuidedBoundaryBridge, PrecisionBudgetSuppressor]
)
def test_refiners_consume_coupled_prior_state(refiner_type):
    conditioned = torch.randn(1, 16, 14, 12)
    deg = torch.randn(1, 16, 14, 12)
    reliability = torch.rand(1, 1, 14, 12)
    state = ReliabilityCalibratedVoting(16)(conditioned, reliability, deg)
    refined = refiner_type(16)(state)
    assert refined.refined_features.shape == deg.shape
    assert refined.bridge_confidence.shape == reliability.shape
    assert refined.bridge_confidence.min() >= 0.0
    assert refined.bridge_confidence.max() <= 1.0
    assert torch.isfinite(refined.refined_features).all()


def test_all_v11_modes_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V11_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        block = TripleStackV11Block(channels=16, nbins=18, mode=mode)
        output, state = block(x, return_state=True)
        output.mean().backward()
        assert output.shape == x.shape
        assert state.scale_reliability.shape == (1, 1, 16, 16)
        assert state.orientation_weights.shape == (1, 4, 16, 16)
        assert torch.isfinite(output).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_zero_padded_shift_does_not_wrap_boundaries():
    x = torch.zeros(1, 1, 4, 4)
    x[..., 0, 0] = 1.0
    shifted_right = _shift_zero(x, 0, 1)
    shifted_down = _shift_zero(x, 1, 0)
    assert shifted_right[..., 0, 1].item() == 1.0
    assert shifted_right[..., 0, 0].item() == 0.0
    assert shifted_down[..., 1, 0].item() == 1.0
    assert shifted_down[..., 0, 0].item() == 0.0


def test_encoder_inserts_v11_after_transmixer():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        nbins=18,
        use_triple_stack_v11=True,
        triple_stack_v11_mode="ram_rcv_cgb",
    )
    assert isinstance(vss.blocks[0][1], TripleStackV11Block)
    assert isinstance(vss.blocks[0][1].deg, HoGEdgeGateConv)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_triple_stack_v3": True, "triple_stack_v3_mode": "msa_ocv_gbc"},
        {"use_triple_stack_v4": True},
        {"use_triple_stack_v5": True},
        {"use_triple_stack_v6": True},
        {"use_triple_stack_v7": True},
        {"use_triple_stack_v8": True},
        {"use_triple_stack_v9": True},
        {"use_triple_stack_v10": True},
        {"use_ccem": True},
        {"use_paper_stack": True},
        {"use_exp_module": True},
    ],
)
def test_encoder_rejects_v11_with_other_experimental_modules(kwargs):
    with pytest.raises(ValueError, match="use_triple_stack_v11 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v11=True,
            triple_stack_v11_mode="ram_rcv_cgb",
            **kwargs,
        )


def test_main_rejects_v11_with_auxiliary_loss_before_training():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args(["--use_triple_stack_v11", "--use_tversky"])
    with pytest.raises(ValueError, match="use_triple_stack_v11 cannot be combined"):
        main(args)


def test_parser_checkpoint_profile_and_segmentor_include_v11():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v11"].default is False
    assert actions["triple_stack_v11_mode"].choices == TRIPLE_STACK_V11_MODES
    assert '"use_triple_stack_v11"' in infer_source
    assert '"triple_stack_v11_mode"' in infer_source
    assert "TripleStack-v11 enabled" in profile_source
    assert "use_triple_stack_v11=getattr(args, 'use_triple_stack_v11', False)" in segmentor_source


def test_queue_dry_run_has_eight_jobs_per_gpu_and_no_gpu0():
    result = subprocess.run(
        ["bash", str(QUEUE), "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("DRYRUN TRAIN")]
    assert len(lines) == 16
    assert sum("gpu=1" in line for line in lines) == 8
    assert sum("gpu=2" in line for line in lines) == 8
    assert not any("gpu=0" in line for line in lines)
    assert all("validation_selection=true" in line for line in lines)


def _checkpoint_args(mode):
    values = {
        "dataset_path": "dataset/CrackMap",
        "seed": 42,
        "epochs": 50,
        "nbins": 180,
        "BCELoss_ratio": 0.87,
        "DiceLoss_ratio": 0.13,
        "validation_selection": True,
        "use_triple_stack_v11": True,
        "triple_stack_v11_mode": mode,
    }
    for name in (
        "use_ccem", "use_tversky", "use_boundary_loss", "use_exp_module",
        "use_exp_second_module", "use_exp_third_module", "use_paper_stack",
        "use_drsgi", "use_rsgdi_v2", "use_triple_stack_v3",
        "use_triple_stack_v4", "use_triple_stack_v5", "use_triple_stack_v6",
        "use_triple_stack_v7", "use_triple_stack_v8", "use_triple_stack_v9",
        "use_triple_stack_v10",
    ):
        values[name] = False
    return SimpleNamespace(**values)


def test_validation_collection_requires_complete_protocol_and_ranks_without_test(tmp_path):
    for index, mode in enumerate(TRIPLE_STACK_V11_MODES):
        run_dir = tmp_path / f"v11_{mode}_seed42" / "run"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint_best_val.pth"
        torch.save({"args": _checkpoint_args(mode)}, checkpoint)
        (run_dir / "checkpoint49.pth").write_bytes(b"final")
        selection = {
            "mIoU": 0.70 + index / 1000,
            "F1": 0.65 + index / 1000,
            "Precision": 0.7,
            "Recall": 0.8,
            "epoch": 10,
            "split": "val",
            "fixed_threshold": 0.5,
        }
        (run_dir / "training_complete.json").write_text(json.dumps({
            "completed_epochs": 50,
            "selection": selection,
            "checkpoint": str(checkpoint),
        }))

    rows = collect_validation(tmp_path)
    assert len(rows) == 16
    assert rows[0]["mode"] == TRIPLE_STACK_V11_MODES[-1]
    assert rows[-1]["mode"] == TRIPLE_STACK_V11_MODES[0]
    assert sum(row["gpu"] == 1 for row in rows) == 8
    assert sum(row["gpu"] == 2 for row in rows) == 8
    assert all(assigned_gpu(row["mode"]) == row["gpu"] for row in rows)


def test_default_encoder_path_remains_original_degconv():
    vss = VSS(in_dim=16, depth=1, state_dim=8, nbins=18)
    assert isinstance(vss.blocks[0][1], HoGEdgeGateConv)
