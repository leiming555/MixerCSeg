import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from main import get_args_parser, main
from models.encoder.vss_block import VSS
from models.experimental_triple_stack_v3 import (
    OrientationCurvatureVoting,
    TRIPLE_STACK_V3_MODES,
)
from models.experimental_triple_stack_v4 import TRIPLE_STACK_V4_MODES
from models.experimental_triple_stack_v5 import TRIPLE_STACK_V5_MODES
from models.experimental_triple_stack_v6 import TRIPLE_STACK_V6_MODES
from models.experimental_triple_stack_v7 import TRIPLE_STACK_V7_MODES
from models.experimental_triple_stack_v8 import TRIPLE_STACK_V8_MODES
from models.experimental_triple_stack_v9 import TRIPLE_STACK_V9_MODES
from models.experimental_triple_stack_v10 import TRIPLE_STACK_V10_MODES
from models.experimental_triple_stack_v11 import TRIPLE_STACK_V11_MODES
from models.experimental_triple_stack_v12 import (
    AxialScaleEvidence,
    BaselineConfidenceArbitration,
    BilateralConsensusSuppressor,
    ConflictAwarePriorArbitration,
    ParallelEvidenceState,
    PrecisionGuidedGapGate,
    ScaleEvidenceState,
    TRIPLE_STACK_V12_ARBITERS,
    TRIPLE_STACK_V12_EVIDENCE,
    TRIPLE_STACK_V12_MODES,
    TRIPLE_STACK_V12_REFINERS,
    TextureConsistencyEvidence,
    TripleStackV12Block,
    WidthFrequencyEvidence,
    _shift_zero,
)
from models.layers import HoGEdgeGateConv
from tools.summarize_triple_stack_v12 import assigned_gpu, collect_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE = PROJECT_ROOT / "logs/triple_stack_v12_crackmap_full12_valselect_gpu12_queue.sh"


def test_v12_modes_are_unique_new_and_cover_grid():
    previous = set().union(
        TRIPLE_STACK_V3_MODES,
        TRIPLE_STACK_V4_MODES,
        TRIPLE_STACK_V5_MODES,
        TRIPLE_STACK_V6_MODES,
        TRIPLE_STACK_V7_MODES,
        TRIPLE_STACK_V8_MODES,
        TRIPLE_STACK_V9_MODES,
        TRIPLE_STACK_V10_MODES,
        TRIPLE_STACK_V11_MODES,
    )
    combinations = {tuple(mode.split("_")) for mode in TRIPLE_STACK_V12_MODES}
    assert len(TRIPLE_STACK_V12_MODES) == 12
    assert len(set(TRIPLE_STACK_V12_MODES)) == 12
    assert set(TRIPLE_STACK_V12_MODES).isdisjoint(previous)
    assert {parts[0] for parts in combinations} == set(TRIPLE_STACK_V12_EVIDENCE)
    assert {parts[1] for parts in combinations} == set(TRIPLE_STACK_V12_ARBITERS)
    assert {parts[2] for parts in combinations} == set(TRIPLE_STACK_V12_REFINERS)


@pytest.mark.parametrize(
    "mode,types",
    [
        (
            "ase_cpa_pgg",
            (AxialScaleEvidence, ConflictAwarePriorArbitration, PrecisionGuidedGapGate),
        ),
        (
            "wfe_bca_bcs",
            (WidthFrequencyEvidence, BaselineConfidenceArbitration, BilateralConsensusSuppressor),
        ),
        (
            "tce_cpa_bcs",
            (TextureConsistencyEvidence, ConflictAwarePriorArbitration, BilateralConsensusSuppressor),
        ),
    ],
)
def test_v12_mapping_preserves_deg_and_ocv(mode, types):
    block = TripleStackV12Block(channels=16, nbins=18, mode=mode)
    assert isinstance(block.evidence, types[0])
    assert isinstance(block.deg, HoGEdgeGateConv)
    assert isinstance(block.ocv, OrientationCurvatureVoting)
    assert isinstance(block.arbiter, types[1])
    assert isinstance(block.refiner, types[2])


@pytest.mark.parametrize(
    "evidence_type", [AxialScaleEvidence, WidthFrequencyEvidence, TextureConsistencyEvidence]
)
def test_scale_evidence_is_finite_and_bounded(evidence_type):
    x = torch.randn(2, 16, 14, 12)
    state = evidence_type(16)(x)
    assert isinstance(state, ScaleEvidenceState)
    assert state.features.shape == x.shape
    assert state.confidence.shape == (2, 1, 14, 12)
    assert torch.isfinite(state.features).all()
    assert state.confidence.min() >= 0.0
    assert state.confidence.max() <= 1.0


@pytest.mark.parametrize(
    "arbiter_type", [ConflictAwarePriorArbitration, BaselineConfidenceArbitration]
)
def test_arbitration_normalizes_weights_and_keeps_baseline_state(arbiter_type):
    input_features = torch.randn(2, 16, 14, 12)
    deg = torch.randn(2, 16, 14, 12)
    ocv = torch.randn(2, 16, 14, 12)
    scale = ScaleEvidenceState(
        torch.randn(2, 16, 14, 12),
        torch.rand(2, 1, 14, 12),
    )
    state = arbiter_type(16)(input_features, deg, scale, ocv)
    assert isinstance(state, ParallelEvidenceState)
    assert state.deg_features is deg
    assert state.fused_features.shape == deg.shape
    assert state.branch_weights.shape == (2, 3, 14, 12)
    assert torch.allclose(
        state.branch_weights.sum(dim=1),
        torch.ones(2, 14, 12),
        atol=1e-5,
    )
    assert state.conflict.min() >= 0.0
    assert state.conflict.max() <= 1.0
    assert state.orientation_confidence.min() >= 0.0
    assert state.orientation_confidence.max() <= 1.0


@pytest.mark.parametrize("refiner_type", [PrecisionGuidedGapGate, BilateralConsensusSuppressor])
def test_refiner_consumes_conflict_and_returns_bounded_confidence(refiner_type):
    x = torch.randn(1, 16, 14, 12)
    scale = AxialScaleEvidence(16)(x)
    deg = torch.randn_like(x)
    ocv = torch.randn_like(x)
    state = ConflictAwarePriorArbitration(16)(x, deg, scale, ocv)
    refined = refiner_type(16)(state)
    assert refined.refined_features.shape == x.shape
    assert refined.bridge_confidence.shape == (1, 1, 14, 12)
    assert refined.bridge_confidence.min() >= 0.0
    assert refined.bridge_confidence.max() <= 1.0
    assert torch.isfinite(refined.refined_features).all()


def test_all_v12_modes_preserve_shape_and_backward():
    for mode in TRIPLE_STACK_V12_MODES:
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        block = TripleStackV12Block(channels=16, nbins=18, mode=mode)
        output, state = block(x, return_state=True)
        output.mean().backward()
        assert output.shape == x.shape
        assert state.branch_weights.shape == (1, 3, 16, 16)
        assert torch.isfinite(output).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_zero_final_gamma_matches_original_deg_anchor_exactly():
    block = TripleStackV12Block(channels=16, nbins=18, mode="ase_cpa_pgg")
    block.eval()
    x = torch.randn(1, 16, 16, 16)
    with torch.no_grad():
        anchor = block.deg(x)
        block.final_gamma.zero_()
        output = block(x)
    assert torch.equal(output, anchor)


def test_zero_padded_shift_does_not_wrap_boundaries():
    x = torch.zeros(1, 1, 4, 4)
    x[..., 0, 0] = 1.0
    shifted_right = _shift_zero(x, 0, 1)
    shifted_down = _shift_zero(x, 1, 0)
    assert shifted_right[..., 0, 1].item() == 1.0
    assert shifted_right[..., 0, 0].item() == 0.0
    assert shifted_down[..., 1, 0].item() == 1.0
    assert shifted_down[..., 0, 0].item() == 0.0


def test_encoder_inserts_v12_after_transmixer():
    vss = VSS(
        in_dim=16,
        depth=1,
        state_dim=8,
        nbins=18,
        use_triple_stack_v12=True,
        triple_stack_v12_mode="ase_cpa_pgg",
    )
    assert isinstance(vss.blocks[0][1], TripleStackV12Block)
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
        {"use_triple_stack_v11": True},
        {"use_ccem": True},
        {"use_paper_stack": True},
        {"use_exp_module": True},
    ],
)
def test_encoder_rejects_v12_with_other_experimental_modules(kwargs):
    with pytest.raises(ValueError, match="use_triple_stack_v12 cannot be combined"):
        VSS(
            in_dim=16,
            depth=1,
            state_dim=8,
            use_triple_stack_v12=True,
            triple_stack_v12_mode="ase_cpa_pgg",
            **kwargs,
        )


def test_main_rejects_v12_with_auxiliary_loss_before_training():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args(["--use_triple_stack_v12", "--use_tversky"])
    with pytest.raises(ValueError, match="use_triple_stack_v12 cannot be combined"):
        main(args)


def test_parser_checkpoint_profile_and_segmentor_include_v12():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    actions = {action.dest: action for action in parser._actions}
    infer_source = (PROJECT_ROOT / "tools/infer_probability.py").read_text()
    profile_source = (PROJECT_ROOT / "tools/profile_model.py").read_text()
    segmentor_source = (PROJECT_ROOT / "models/segmentor/MixerCSeg.py").read_text()

    assert actions["use_triple_stack_v12"].default is False
    assert actions["triple_stack_v12_mode"].choices == TRIPLE_STACK_V12_MODES
    assert '"use_triple_stack_v12"' in infer_source
    assert '"triple_stack_v12_mode"' in infer_source
    assert "TripleStack-v12 enabled" in profile_source
    assert "use_triple_stack_v12=getattr(args, 'use_triple_stack_v12', False)" in segmentor_source


def test_queue_dry_run_has_six_jobs_per_gpu_and_no_gpu0():
    result = subprocess.run(
        ["bash", str(QUEUE), "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("DRYRUN TRAIN")]
    assert len(lines) == 12
    assert sum("gpu=1" in line for line in lines) == 6
    assert sum("gpu=2" in line for line in lines) == 6
    assert not any("gpu=0" in line for line in lines)
    assert all("validation_selection=true" in line for line in lines)


def checkpoint_args(mode):
    values = {
        "dataset_path": "dataset/CrackMap",
        "seed": 42,
        "epochs": 50,
        "nbins": 180,
        "BCELoss_ratio": 0.87,
        "DiceLoss_ratio": 0.13,
        "validation_selection": True,
        "use_triple_stack_v12": True,
        "triple_stack_v12_mode": mode,
    }
    for name in (
        "use_ccem", "use_tversky", "use_boundary_loss", "use_exp_module",
        "use_exp_second_module", "use_exp_third_module", "use_paper_stack",
        "use_drsgi", "use_rsgdi_v2", "use_triple_stack_v3",
        "use_triple_stack_v4", "use_triple_stack_v5", "use_triple_stack_v6",
        "use_triple_stack_v7", "use_triple_stack_v8", "use_triple_stack_v9",
        "use_triple_stack_v10", "use_triple_stack_v11",
    ):
        values[name] = False
    return SimpleNamespace(**values)


def test_validation_collection_requires_complete_protocol_and_ranks_without_test(tmp_path):
    for index, mode in enumerate(TRIPLE_STACK_V12_MODES):
        run_dir = tmp_path / f"v12_{mode}_seed42" / "run"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint_best_val.pth"
        torch.save({"args": checkpoint_args(mode)}, checkpoint)
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
    assert len(rows) == 12
    assert rows[0]["mode"] == TRIPLE_STACK_V12_MODES[-1]
    assert rows[-1]["mode"] == TRIPLE_STACK_V12_MODES[0]
    assert sum(row["gpu"] == 1 for row in rows) == 6
    assert sum(row["gpu"] == 2 for row in rows) == 6
    assert all(assigned_gpu(row["mode"]) == row["gpu"] for row in rows)


def test_default_encoder_path_remains_original_degconv():
    vss = VSS(in_dim=16, depth=1, state_dim=8, nbins=18)
    assert isinstance(vss.blocks[0][1], HoGEdgeGateConv)
