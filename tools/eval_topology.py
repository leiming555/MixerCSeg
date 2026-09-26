"""Offline topology and boundary metrics for saved binary segmentation maps."""

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_standard import load_pairs


TOPOLOGY_FIELDS = [
    "method",
    "result_dir",
    "threshold",
    "boundary_tolerance_px",
    "n_images",
    "clDice",
    "Boundary_F1",
    "Boundary_Precision",
    "Boundary_Recall",
    "Component_MAE",
    "Endpoint_MAE",
]


def binary_mask(mask):
    return (np.asarray(mask) > 0).astype(np.uint8)


def morphological_skeleton(mask):
    work = binary_mask(mask) * 255
    skeleton = np.zeros_like(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded
    return (skeleton > 0).astype(np.uint8)


def safe_overlap(numerator, denominator, both_empty):
    if denominator:
        return numerator / denominator
    return 1.0 if both_empty else 0.0


def cldice(prediction, target):
    prediction = binary_mask(prediction)
    target = binary_mask(target)
    if not prediction.any() and not target.any():
        return 1.0
    pred_skeleton = morphological_skeleton(prediction)
    target_skeleton = morphological_skeleton(target)
    topology_precision = safe_overlap(
        int(np.sum(pred_skeleton & target)),
        int(np.sum(pred_skeleton)),
        not target.any(),
    )
    topology_recall = safe_overlap(
        int(np.sum(target_skeleton & prediction)),
        int(np.sum(target_skeleton)),
        not prediction.any(),
    )
    denominator = topology_precision + topology_recall
    return 0.0 if denominator == 0 else 2.0 * topology_precision * topology_recall / denominator


def mask_boundary(mask):
    mask = binary_mask(mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & (1 - eroded)


def boundary_metrics(prediction, target, tolerance_px=2):
    pred_boundary = mask_boundary(prediction)
    target_boundary = mask_boundary(target)
    pred_count = int(np.sum(pred_boundary))
    target_count = int(np.sum(target_boundary))
    if pred_count == 0 and target_count == 0:
        return {"F1": 1.0, "Precision": 1.0, "Recall": 1.0}
    size = tolerance_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated_pred = cv2.dilate(pred_boundary, kernel)
    dilated_target = cv2.dilate(target_boundary, kernel)
    matched_pred = int(np.sum(pred_boundary & dilated_target))
    matched_target = int(np.sum(target_boundary & dilated_pred))
    precision = matched_pred / pred_count if pred_count else 0.0
    recall = matched_target / target_count if target_count else 0.0
    denominator = precision + recall
    f1 = 0.0 if denominator == 0 else 2.0 * precision * recall / denominator
    return {"F1": f1, "Precision": precision, "Recall": recall}


def component_count(mask):
    count, _ = cv2.connectedComponents(binary_mask(mask), connectivity=8)
    return count - 1


def endpoint_count(mask):
    skeleton = morphological_skeleton(mask)
    neighborhood = cv2.filter2D(
        skeleton,
        cv2.CV_16S,
        np.ones((3, 3), dtype=np.int16),
        borderType=cv2.BORDER_CONSTANT,
    )
    neighbors = neighborhood - skeleton
    return int(np.sum((skeleton == 1) & (neighbors == 1)))


def evaluate_topology(pairs, threshold=0.5, boundary_tolerance_px=2):
    rows = []
    for probability, target in pairs:
        prediction = (probability > threshold).astype(np.uint8)
        target = binary_mask(target)
        boundary = boundary_metrics(prediction, target, boundary_tolerance_px)
        rows.append(
            {
                "clDice": cldice(prediction, target),
                "Boundary_F1": boundary["F1"],
                "Boundary_Precision": boundary["Precision"],
                "Boundary_Recall": boundary["Recall"],
                "Component_MAE": abs(component_count(prediction) - component_count(target)),
                "Endpoint_MAE": abs(endpoint_count(prediction) - endpoint_count(target)),
            }
        )
    if not rows:
        raise ValueError("No prediction/target pairs")
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "clDice",
            "Boundary_F1",
            "Boundary_Precision",
            "Boundary_Recall",
            "Component_MAE",
            "Endpoint_MAE",
        )
    }


def evaluate_result_dir(result_dir, method="", threshold=0.5, boundary_tolerance_px=2):
    result_dir = Path(result_dir).resolve()
    pairs = load_pairs(str(result_dir))
    metrics = evaluate_topology(pairs, threshold, boundary_tolerance_px)
    return {
        "method": method or result_dir.name,
        "result_dir": str(result_dir),
        "threshold": threshold,
        "boundary_tolerance_px": boundary_tolerance_px,
        "n_images": len(pairs),
        **metrics,
    }


def write_csv(path, row):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOPOLOGY_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary_tolerance_px", type=int, default=2)
    parser.add_argument("--save_csv", default="")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if args.boundary_tolerance_px < 0:
        parser.error("--boundary_tolerance_px must be non-negative")
    row = evaluate_result_dir(
        args.result_dir,
        args.method,
        args.threshold,
        args.boundary_tolerance_px,
    )
    print("Topology Evaluation")
    for field in TOPOLOGY_FIELDS:
        print(f"{field}: {row[field]}")
    if args.save_csv:
        write_csv(args.save_csv, row)


if __name__ == "__main__":
    main()
