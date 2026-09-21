from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.make_paper_qualitative_v2 import add_rois, choose_gt_roi, crop_to_roi


def test_gt_roi_selects_densest_window_deterministically():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[7:11, 8:12] = 1
    assert choose_gt_roi(mask, crop_size=4) == (8, 7, 12, 11)
    assert choose_gt_roi(mask, crop_size=4) == (8, 7, 12, 11)


def test_gt_roi_ties_use_top_left_window():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[7:9, 7:9] = 1
    assert choose_gt_roi(mask, crop_size=2) == (1, 1, 3, 3)


def test_gt_roi_rejects_invalid_masks():
    with pytest.raises(ValueError, match="empty"):
        choose_gt_roi(np.zeros((5, 5), dtype=np.uint8), crop_size=3)
    with pytest.raises(ValueError, match="two-dimensional"):
        choose_gt_roi(np.zeros((2, 2, 3), dtype=np.uint8), crop_size=2)
    with pytest.raises(ValueError, match="positive"):
        choose_gt_roi(np.ones((5, 5), dtype=np.uint8), crop_size=0)


def test_crop_to_roi_preserves_identical_coordinates_across_panels():
    image = np.arange(100).reshape(10, 10)
    crop = crop_to_roi(image, (2, 3, 7, 9))
    assert crop.shape == (6, 5)
    assert crop[0, 0] == image[3, 2]
    assert crop[-1, -1] == image[8, 6]
    with pytest.raises(ValueError, match="outside"):
        crop_to_roi(image, (-1, 0, 5, 5))


def test_add_rois_uses_evaluation_labels_only(tmp_path):
    baseline_dir = Path(tmp_path)
    label = np.zeros((16, 16), dtype=np.uint8)
    label[9:13, 10:14] = 255
    assert cv2.imwrite(str(baseline_dir / "case_lab.png"), label)
    selected = [{
        "role": "sparse",
        "name": "case",
        "foreground_ratio": 0.1,
        "components": 1,
        "fragmentation": 0.1,
        "boundary_complexity": 1.0,
    }]
    rows = add_rois(selected, baseline_dir, crop_size=4)
    assert len(rows) == 1
    assert (rows[0]["roi_x0"], rows[0]["roi_y0"], rows[0]["roi_x1"], rows[0]["roi_y1"]) == (
        10, 9, 14, 13
    )
    assert rows[0]["roi_foreground"] == 16
    assert rows[0]["roi_rule"] == "max_gt_foreground_then_centroid_4px"
