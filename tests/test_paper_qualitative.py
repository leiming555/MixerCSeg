import numpy as np
import pytest

from tools.make_paper_qualitative import (
    describe_mask, error_overlay, resize_binary, select_representatives,
)


def test_describe_mask_reports_geometry_without_predictions():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    mask[7:9, 7:9] = 255
    row = describe_mask(mask, "sample")
    assert row["name"] == "sample"
    assert row["foreground_ratio"] == pytest.approx(0.13)
    assert row["components"] == 2
    assert row["fragmentation"] > 0
    assert row["boundary_complexity"] > 0


def test_representatives_are_deterministic_and_unique():
    rows = [
        {
            "name": f"sample-{index}",
            "foreground_ratio": ratio,
            "components": components,
            "fragmentation": fragmentation,
            "boundary_complexity": complexity,
        }
        for index, (ratio, components, fragmentation, complexity) in enumerate(
            [
                (0.01, 1, 0.1, 1.0),
                (0.03, 7, 1.4, 2.0),
                (0.05, 2, 0.3, 8.0),
                (0.20, 1, 0.1, 1.2),
                (0.10, 3, 0.4, 3.0),
            ]
        )
    ]
    selected = select_representatives(rows)
    assert [row["name"] for row in selected] == [
        "sample-0", "sample-1", "sample-2", "sample-3",
    ]
    assert len({row["name"] for row in selected}) == 4


def test_error_overlay_color_semantics():
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    label = np.array([[1, 0], [1, 0]], dtype=bool)
    overlay = error_overlay(prediction, label)
    assert tuple(overlay[0, 0]) == (45, 180, 75)
    assert tuple(overlay[0, 1]) == (225, 55, 55)
    assert tuple(overlay[1, 0]) == (45, 125, 235)
    assert tuple(overlay[1, 1]) == (0, 0, 0)
    with pytest.raises(ValueError, match="shapes differ"):
        error_overlay(np.zeros((2, 2)), np.zeros((3, 3)))


def test_binary_resize_matches_nearest_neighbor_pipeline():
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    resized = resize_binary(mask, (4, 4))
    assert resized.shape == (4, 4)
    assert resized.sum() == 8
    assert resized[0, 0] == 0 and resized[0, 3] == 1
