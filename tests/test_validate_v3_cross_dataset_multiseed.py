import copy
from types import SimpleNamespace

import pytest

from tools.validate_v3_cross_dataset_multiseed import (
    ALL_METRICS,
    ALL_SEEDS,
    DATASETS,
    JOBS,
    MODELS,
    metric_summary,
    paired_deltas,
    paired_summary,
    seed42_reference_rows,
    validate_checkpoint,
    wait_for_gpu,
)


def checkpoint_args(model, **updates):
    values = {
        "epochs": 50,
        "seed": 3407,
        "nbins": 180,
        "dataset_path": "dataset/DeepCrack",
        "BCELoss_ratio": 0.87,
        "DiceLoss_ratio": 0.13,
        "use_triple_stack_v3": model == "msa_ocv_gbc",
        "triple_stack_v3_mode": "msa_ocv_gbc",
        "use_ccem": False,
        "use_tversky": False,
        "use_boundary_loss": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def metric_row(dataset, model, seed, offset):
    row = {"dataset": dataset, "model": model, "seed": seed}
    for index, metric in enumerate(ALL_METRICS):
        row[metric] = 0.7 + index * 0.001 + offset
    return row


def test_fixed_job_matrix_contains_exactly_eight_new_runs():
    assert len(JOBS) == 8
    assert {job["gpu"] for job in JOBS} == {0, 1, 2}
    assert [sum(job["gpu"] == gpu for job in JOBS) for gpu in (0, 1, 2)] == [2, 3, 3]
    keys = {(job["dataset"], job["model"], job["seed"]) for job in JOBS}
    assert keys == {
        (dataset, model, seed)
        for dataset in DATASETS
        for model in MODELS
        for seed in (3407, 2026)
    }


@pytest.mark.parametrize("model", MODELS)
def test_checkpoint_contract_accepts_baseline_and_v3_only(model):
    checkpoint = {
        "args": checkpoint_args(model),
        "selection": {"split": "val", "fixed_threshold": 0.5},
    }
    validate_checkpoint(checkpoint, model, "DeepCrack", 3407)
    wrong = copy.deepcopy(checkpoint)
    wrong["args"].use_ccem = True
    with pytest.raises(ValueError, match="Unexpected active"):
        validate_checkpoint(wrong, model, "DeepCrack", 3407)


def test_wait_for_gpu_enforces_physical_assignment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(
        "tools.validate_v3_cross_dataset_multiseed.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="7, 0\n"),
    )
    wait_for_gpu(2)
    with pytest.raises(ValueError, match="GPU mismatch"):
        wait_for_gpu(1)


def test_seed42_reference_set_is_complete():
    records = seed42_reference_rows()
    assert set(records) == {(dataset, model, 42) for dataset in DATASETS for model in MODELS}


def test_summary_and_paired_deltas_use_matched_seeds():
    rows = []
    for dataset in DATASETS:
        for seed in ALL_SEEDS:
            rows.append(metric_row(dataset, "baseline", seed, 0.0))
            rows.append(metric_row(dataset, "msa_ocv_gbc", seed, 0.01))
    summaries = metric_summary(rows)
    assert len(summaries) == 4
    assert all(row["n"] == 3 for row in summaries)
    deltas = paired_deltas(rows)
    assert len(deltas) == 6
    for row in deltas:
        assert row["mIoU_fixed"] == pytest.approx(0.01)
        assert row["Component_MAE"] == pytest.approx(-0.01)
    paired = paired_summary(deltas)
    assert len(paired) == len(DATASETS) * len(ALL_METRICS)
    assert all(row["wins"] == 3 for row in paired if row["metric"] == "mIoU_fixed")
