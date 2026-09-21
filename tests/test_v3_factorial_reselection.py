from types import SimpleNamespace

import pytest

from tools.reselect_v3_factorial import (
    FACTOR_NAMES,
    MODE_ORDER,
    component_bits,
    factorial_effect,
    factorial_jobs,
    select_best,
    validate_factorial_checkpoint,
)


def test_manifest_has_exact_factorial_and_balanced_gpu_assignment():
    jobs = factorial_jobs()
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 9
    assert tuple(job["mode"] for job in jobs) == MODE_ORDER
    assert [sum(job["gpu"] == gpu for job in jobs) for gpu in range(3)] == [3, 3, 3]


def test_component_mapping_covers_all_eight_factorial_cells():
    cells = set()
    for mode in MODE_ORDER[1:]:
        bits = component_bits(mode)
        assert bits["wrapper"] == 1
        cells.add(tuple(bits[name] for name in FACTOR_NAMES))
    assert cells == {
        (msa, ocv, gbc)
        for msa in (0, 1)
        for ocv in (0, 1)
        for gbc in (0, 1)
    }
    assert component_bits("baseline") == {"wrapper": 0, "MSA": 0, "OCV": 0, "GBC": 0}
    with pytest.raises(ValueError):
        component_bits("msa_bad_gbc")


def checkpoint_args(**updates):
    values = dict(
        epochs=50,
        seed=42,
        nbins=180,
        dataset_path="dataset/CrackMap",
        BCELoss_ratio=0.87,
        DiceLoss_ratio=0.13,
        use_triple_stack_v3=True,
        triple_stack_v3_mode="msa_ocv_gbc",
        use_ccem=False,
        use_tversky=False,
        use_boundary_loss=False,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_checkpoint_protocol_accepts_v3_and_baseline_only():
    v3_job = dict(job_id="v3", mode="msa_ocv_gbc")
    validate_factorial_checkpoint({"args": checkpoint_args(), "epoch": 49}, v3_job, 49)

    baseline_job = dict(job_id="baseline", mode="baseline")
    baseline = checkpoint_args(use_triple_stack_v3=False)
    validate_factorial_checkpoint({"args": baseline, "epoch": 49}, baseline_job, 49)

    bad = checkpoint_args(use_tversky=True)
    with pytest.raises(ValueError, match="Unexpected active flags"):
        validate_factorial_checkpoint({"args": bad, "epoch": 49}, v3_job, 49)
    wrong_mode = checkpoint_args(triple_stack_v3_mode="id_ocv_gbc")
    with pytest.raises(ValueError, match="Wrong V3 mode"):
        validate_factorial_checkpoint({"args": wrong_mode, "epoch": 49}, v3_job, 49)


def test_selection_uses_miou_then_f1_then_earliest_epoch():
    rows = [
        {"mIoU": 0.8, "F1": 0.7, "epoch": 1},
        {"mIoU": 0.8, "F1": 0.71, "epoch": 8},
        {"mIoU": 0.8, "F1": 0.71, "epoch": 4},
    ]
    assert select_best(rows)["epoch"] == 4


def test_factorial_effects_recover_additive_main_effects():
    rows = []
    for msa in (0, 1):
        for ocv in (0, 1):
            for gbc in (0, 1):
                rows.append({
                    "wrapper": 1,
                    "MSA": msa,
                    "OCV": ocv,
                    "GBC": gbc,
                    "mIoU_fixed": 0.5 + 0.03 * msa + 0.02 * ocv - 0.01 * gbc,
                })
    assert factorial_effect(rows, ("MSA",), "mIoU_fixed") == pytest.approx(0.03)
    assert factorial_effect(rows, ("OCV",), "mIoU_fixed") == pytest.approx(0.02)
    assert factorial_effect(rows, ("GBC",), "mIoU_fixed") == pytest.approx(-0.01)
    assert factorial_effect(rows, ("MSA", "OCV"), "mIoU_fixed") == pytest.approx(0.0)
    assert factorial_effect(rows, FACTOR_NAMES, "mIoU_fixed") == pytest.approx(0.0)
