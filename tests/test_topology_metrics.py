import numpy as np
import pytest

from tools.eval_topology import (
    boundary_metrics,
    cldice,
    component_count,
    endpoint_count,
    evaluate_topology,
    morphological_skeleton,
)


def horizontal_line(row=8, start=3, stop=13):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[row, start:stop] = 1
    return mask


def test_identical_line_has_perfect_topology_and_boundary():
    mask = horizontal_line()
    assert cldice(mask, mask) == pytest.approx(1.0)
    assert boundary_metrics(mask, mask, tolerance_px=2)["F1"] == pytest.approx(1.0)
    assert component_count(mask) == 1
    assert endpoint_count(mask) == 2


def test_broken_line_reduces_cldice_and_increases_component_error():
    target = horizontal_line()
    prediction = target.copy()
    prediction[8, 7:9] = 0
    assert cldice(prediction, target) < 1.0
    assert component_count(prediction) == 2
    assert abs(component_count(prediction) - component_count(target)) == 1
    assert endpoint_count(prediction) > endpoint_count(target)


def test_boundary_tolerance_accepts_small_shift_but_not_large_shift():
    target = horizontal_line(row=6)
    close = horizontal_line(row=7)
    far = horizontal_line(row=12)
    close_f1 = boundary_metrics(close, target, tolerance_px=2)["F1"]
    far_f1 = boundary_metrics(far, target, tolerance_px=2)["F1"]
    assert close_f1 == pytest.approx(1.0)
    assert far_f1 == pytest.approx(0.0)


def test_empty_masks_are_perfect_and_one_sided_empty_is_zero():
    empty = np.zeros((16, 16), dtype=np.uint8)
    line = horizontal_line()
    assert cldice(empty, empty) == pytest.approx(1.0)
    assert boundary_metrics(empty, empty)["F1"] == pytest.approx(1.0)
    assert cldice(empty, line) == pytest.approx(0.0)
    assert boundary_metrics(empty, line)["F1"] == pytest.approx(0.0)


def test_skeleton_and_dataset_aggregation_are_finite():
    thick = np.zeros((16, 16), dtype=np.uint8)
    thick[5:10, 3:13] = 1
    skeleton = morphological_skeleton(thick)
    assert skeleton.sum() > 0
    pairs = [
        (thick.astype(np.float32), thick),
        (horizontal_line().astype(np.float32), horizontal_line()),
    ]
    metrics = evaluate_topology(pairs, threshold=0.5, boundary_tolerance_px=2)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["clDice"] == pytest.approx(1.0)
    assert metrics["Boundary_F1"] == pytest.approx(1.0)
    assert metrics["Component_MAE"] == pytest.approx(0.0)
