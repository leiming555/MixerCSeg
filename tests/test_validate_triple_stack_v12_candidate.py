import copy
from types import SimpleNamespace

import pytest

from tools.validate_triple_stack_v12_candidate import (
    METRICS,
    MODE,
    STAGE1_JOBS,
    STAGE2_JOBS,
    final_decision,
    seed42_record,
    stage1_decision,
    validate_candidate_checkpoint,
    wait_for_gpu,
)


def row(dataset, seed, miou, f1, ods=None, ois=None):
    return {
        "dataset": dataset,
        "seed": seed,
        "mIoU_fixed": miou,
        "F1_fixed": f1,
        "P_fixed": 0.8,
        "R_fixed": 0.8,
        "ODS_F1": f1 if ods is None else ods,
        "OIS_F1": f1 if ois is None else ois,
    }


def references():
    result = {}
    for seed in (42, 3407, 2026, 1234):
        result[("CrackMap", "baseline", seed)] = row("CrackMap", seed, 0.78, 0.74)
        result[("CrackMap", "msa_ocv_gbc", seed)] = row(
            "CrackMap", seed, 0.80, 0.77, 0.78, 0.79
        )
    for dataset in ("DeepCrack", "CamCrack789"):
        result[(dataset, "baseline", 42)] = row(dataset, 42, 0.85, 0.83)
        result[(dataset, "msa_ocv_gbc", 42)] = row(dataset, 42, 0.86, 0.84)
    return result


def checkpoint_args(**updates):
    values = {
        "epochs": 50,
        "seed": 42,
        "nbins": 180,
        "dataset_path": "dataset/CrackMap",
        "BCELoss_ratio": 0.87,
        "DiceLoss_ratio": 0.13,
        "use_triple_stack_v12": True,
        "triple_stack_v12_mode": MODE,
        "use_ccem": False,
        "use_tversky": False,
        "use_boundary_loss": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_fixed_jobs_use_three_gpus_then_two_conditional_jobs():
    assert STAGE1_JOBS == [
        {"dataset": "CrackMap", "seed": 3407, "gpu": 0},
        {"dataset": "CrackMap", "seed": 2026, "gpu": 1},
        {"dataset": "CrackMap", "seed": 1234, "gpu": 2},
    ]
    assert STAGE2_JOBS == [
        {"dataset": "DeepCrack", "seed": 42, "gpu": 0},
        {"dataset": "CamCrack789", "seed": 42, "gpu": 1},
    ]


def test_wait_for_gpu_accepts_assigned_physical_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        "tools.validate_triple_stack_v12_candidate.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="6, 0\n"),
    )
    wait_for_gpu(0)


def test_wait_for_gpu_rejects_visible_device_mismatch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="GPU mismatch"):
        wait_for_gpu(0)


def test_checkpoint_contract_accepts_only_frozen_v12_candidate():
    checkpoint = {
        "args": checkpoint_args(),
        "selection": {"split": "val", "fixed_threshold": 0.5},
    }
    validate_candidate_checkpoint(checkpoint, "CrackMap", 42)

    wrong = copy.deepcopy(checkpoint)
    wrong["args"].use_ccem = True
    with pytest.raises(ValueError, match="Unexpected active"):
        validate_candidate_checkpoint(wrong, "CrackMap", 42)

    wrong = copy.deepcopy(checkpoint)
    wrong["args"].triple_stack_v12_mode = "ase_cpa_pgg"
    with pytest.raises(ValueError, match="Wrong TripleStack-v12 mode"):
        validate_candidate_checkpoint(wrong, "CrackMap", 42)


def test_imported_seed42_result_is_complete_and_validation_selected():
    imported = seed42_record()
    assert imported["dataset"] == "CrackMap"
    assert imported["seed"] == 42
    assert imported["model"] == MODE
    assert imported["selection_split"] == "val"
    assert imported["mIoU_fixed"] == pytest.approx(0.8054275270036997)


def test_stage1_promotes_only_when_every_rule_passes():
    rows = [
        row("CrackMap", seed, 0.82, 0.79, 0.79, 0.80)
        for seed in (42, 3407, 2026, 1234)
    ]
    decision = stage1_decision(rows, references())
    assert decision["promoted"]
    assert decision["paired_v3_wins"] == 4
    assert all(decision["checks"].values())


def test_stage1_rejects_mean_win_with_too_few_paired_wins():
    rows = [row("CrackMap", 42, 0.86, 0.84, 0.81, 0.82)]
    rows += [
        row("CrackMap", seed, 0.799, 0.769, 0.79, 0.80)
        for seed in (3407, 2026, 1234)
    ]
    decision = stage1_decision(rows, references())
    assert not decision["promoted"]
    assert not decision["checks"]["paired_v3_wins_at_least_3"]


def test_stage1_rejects_when_both_threshold_metrics_drop():
    rows = [
        row("CrackMap", seed, 0.82, 0.79, 0.777, 0.787)
        for seed in (42, 3407, 2026, 1234)
    ]
    decision = stage1_decision(rows, references())
    assert not decision["promoted"]
    assert not decision["checks"]["ods_ois_not_both_below_v3_by_0_002"]


def test_final_promotion_requires_baseline_wins_v3_tolerance_and_strict_win():
    refs = references()
    rows = [
        row("DeepCrack", 42, 0.861, 0.841),
        row("CamCrack789", 42, 0.859, 0.839),
    ]
    assert final_decision(rows, refs)["promoted"]

    below_baseline = copy.deepcopy(rows)
    below_baseline[0]["mIoU_fixed"] = 0.84
    assert not final_decision(below_baseline, refs)["promoted"]

    no_strict_win = [
        row("DeepCrack", 42, 0.859, 0.839),
        row("CamCrack789", 42, 0.859, 0.839),
    ]
    assert not final_decision(no_strict_win, refs)["promoted"]


def test_metric_contract_is_complete():
    assert METRICS == [
        "mIoU_fixed",
        "F1_fixed",
        "P_fixed",
        "R_fixed",
        "ODS_F1",
        "OIS_F1",
    ]
