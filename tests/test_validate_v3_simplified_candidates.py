import copy
from types import SimpleNamespace

import pytest

from tools.validate_v3_simplified_candidates import (
    CANDIDATES,
    METRICS,
    STAGE1_JOBS,
    STAGE2_DATASETS,
    candidate_decision,
    choose_candidate,
    final_decision,
    imported_seed42_record,
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
    for dataset, _ in STAGE2_DATASETS:
        result[(dataset, "baseline", 42)] = row(dataset, 42, 0.85, 0.83)
        result[(dataset, "msa_ocv_gbc", 42)] = row(dataset, 42, 0.86, 0.84)
    return result


def checkpoint_args(mode, **updates):
    values = {
        "epochs": 50,
        "seed": 42,
        "nbins": 180,
        "dataset_path": "dataset/CrackMap",
        "BCELoss_ratio": 0.87,
        "DiceLoss_ratio": 0.13,
        "use_triple_stack_v3": True,
        "triple_stack_v3_mode": mode,
        "use_ccem": False,
        "use_tversky": False,
        "use_boundary_loss": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_fixed_jobs_assign_two_candidates_to_each_gpu():
    assert CANDIDATES == ("id_ocv_gbc", "msa_id_gbc")
    assert len(STAGE1_JOBS) == 6
    assert {job["gpu"] for job in STAGE1_JOBS} == {0, 1, 2}
    for gpu in (0, 1, 2):
        jobs = [job for job in STAGE1_JOBS if job["gpu"] == gpu]
        assert [job["mode"] for job in jobs] == list(CANDIDATES)
    assert STAGE2_DATASETS == (("DeepCrack", 0), ("CamCrack789", 1))


def test_wait_for_gpu_accepts_assignment_and_rejects_mismatch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        "tools.validate_v3_simplified_candidates.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="6, 0\n"),
    )
    wait_for_gpu(1)
    with pytest.raises(ValueError, match="GPU mismatch"):
        wait_for_gpu(2)


@pytest.mark.parametrize("mode", CANDIDATES)
def test_checkpoint_contract_accepts_only_simplified_candidates(mode):
    checkpoint = {
        "args": checkpoint_args(mode),
        "selection": {"split": "val", "fixed_threshold": 0.5},
    }
    validate_candidate_checkpoint(checkpoint, mode, "CrackMap", 42)
    wrong = copy.deepcopy(checkpoint)
    wrong["args"].use_ccem = True
    with pytest.raises(ValueError, match="Unexpected active"):
        validate_candidate_checkpoint(wrong, mode, "CrackMap", 42)


@pytest.mark.parametrize("mode", CANDIDATES)
def test_imported_seed42_result_is_complete(mode):
    imported = imported_seed42_record(mode)
    assert imported["dataset"] == "CrackMap"
    assert imported["seed"] == 42
    assert imported["mode"] == mode
    assert imported["selection_split"] == "val"


def test_candidate_promotion_requires_mean_wins_pairing_and_stability():
    rows = [
        row("CrackMap", seed, 0.82, 0.79, 0.79, 0.80)
        for seed in (42, 3407, 2026, 1234)
    ]
    decision = candidate_decision(CANDIDATES[0], rows, references())
    assert decision["promoted"]
    assert decision["paired_v3_wins"] == 4

    unstable = copy.deepcopy(rows)
    unstable[0]["mIoU_fixed"] = 0.90
    unstable[1]["mIoU_fixed"] = 0.801
    decision = candidate_decision(CANDIDATES[0], unstable, references())
    assert not decision["promoted"]
    assert not decision["checks"]["mIoU_std_no_worse_than_baseline"]


def test_choose_candidate_prefers_higher_mean_miou_then_f1():
    first_rows = [
        row("CrackMap", seed, 0.820, 0.790, 0.79, 0.80)
        for seed in (42, 3407, 2026, 1234)
    ]
    second_rows = [
        row("CrackMap", seed, 0.821, 0.789, 0.79, 0.80)
        for seed in (42, 3407, 2026, 1234)
    ]
    decisions = [
        candidate_decision(CANDIDATES[0], first_rows, references()),
        candidate_decision(CANDIDATES[1], second_rows, references()),
    ]
    assert choose_candidate(decisions) == CANDIDATES[1]


def test_final_promotion_requires_cross_dataset_contract():
    refs = references()
    rows = [
        row("DeepCrack", 42, 0.861, 0.841),
        row("CamCrack789", 42, 0.859, 0.839),
    ]
    assert final_decision(rows, refs)["promoted"]
    rows[0]["mIoU_fixed"] = 0.84
    assert not final_decision(rows, refs)["promoted"]


def test_metric_contract_is_complete():
    assert METRICS == [
        "mIoU_fixed",
        "F1_fixed",
        "P_fixed",
        "R_fixed",
        "ODS_F1",
        "OIS_F1",
    ]
