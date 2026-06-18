import argparse
import csv
import glob
import os
from pathlib import Path

import cv2
import numpy as np


CSV_FIELDS = [
    "method",
    "result_dir",
    "mIoU_fixed",
    "F1_fixed",
    "P_fixed",
    "R_fixed",
    "ODS_thr",
    "ODS_F1",
    "P_ODS",
    "R_ODS",
    "mIoU_ODS",
    "best_mIoU_thr",
    "best_mIoU",
    "OIS_F1",
]


def parse_thresholds(value):
    if value:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [i / 100.0 for i in range(100)]


def read_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def load_pairs(result_dir):
    lab_paths = sorted(glob.glob(os.path.join(result_dir, "*_lab.png")))
    if not lab_paths:
        raise FileNotFoundError(f"No *_lab.png files found in {result_dir}")

    pairs = []
    missing = []
    for lab_path in lab_paths:
        pre_path = lab_path.replace("_lab.png", "_pre.png")
        if not os.path.exists(pre_path):
            missing.append(pre_path)
            continue
        pred = read_gray(pre_path).astype(np.float32) / 255.0
        gt = (read_gray(lab_path) > 127).astype(np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {pre_path} {pred.shape} vs {lab_path} {gt.shape}")
        pairs.append((pred, gt))

    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing[:20]))
    if not pairs:
        raise FileNotFoundError(f"No valid *_pre.png/*_lab.png pairs found in {result_dir}")
    return pairs


def confusion_at_threshold(pairs, threshold):
    tp = fp = fn = tn = 0
    per_image = []
    for pred, gt in pairs:
        pred_bin = (pred > threshold).astype(np.uint8)
        cur_tp = int(np.sum((pred_bin == 1) & (gt == 1)))
        cur_fp = int(np.sum((pred_bin == 1) & (gt == 0)))
        cur_fn = int(np.sum((pred_bin == 0) & (gt == 1)))
        cur_tn = int(np.sum((pred_bin == 0) & (gt == 0)))
        tp += cur_tp
        fp += cur_fp
        fn += cur_fn
        tn += cur_tn
        per_image.append((cur_tp, cur_fp, cur_fn, cur_tn))
    return tp, fp, fn, tn, per_image


def metrics_from_counts(tp, fp, fn, tn):
    precision = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)

    crack_den = tp + fp + fn
    bg_den = tn + fp + fn
    iou_crack = 0.0 if crack_den == 0 else tp / crack_den
    iou_bg = 0.0 if bg_den == 0 else tn / bg_den
    miou = (iou_crack + iou_bg) / 2.0
    return {
        "mIoU": miou,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
    }


def eval_threshold(pairs, threshold):
    tp, fp, fn, tn, _ = confusion_at_threshold(pairs, threshold)
    return metrics_from_counts(tp, fp, fn, tn)


def eval_ods_and_best_miou(pairs, thresholds):
    rows = []
    for threshold in thresholds:
        metrics = eval_threshold(pairs, threshold)
        rows.append((threshold, metrics))

    ods_thr, ods_metrics = max(rows, key=lambda item: (item[1]["F1"], item[1]["mIoU"]))
    best_miou_thr, best_miou_metrics = max(rows, key=lambda item: (item[1]["mIoU"], item[1]["F1"]))
    return ods_thr, ods_metrics, best_miou_thr, best_miou_metrics


def eval_ois(pairs, thresholds):
    best_f1_list = []
    for pred, gt in pairs:
        single_pair = [(pred, gt)]
        best = max((eval_threshold(single_pair, threshold)["F1"] for threshold in thresholds), default=0.0)
        best_f1_list.append(best)
    return float(np.mean(np.array(best_f1_list))) if best_f1_list else 0.0


def infer_method(result_dir):
    path = Path(result_dir)
    if path.name.startswith("results_") and path.parent.name:
        return path.parent.name
    return path.name


def save_summary_csv(path, row):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def format_float(value):
    return f"{value:.6f}"


def print_summary(row):
    print("Standard Evaluation")
    print(f"method: {row['method']}")
    print(f"result_dir: {row['result_dir']}")
    print("")
    print("Fixed Threshold")
    print(f"mIoU@fixed: {format_float(row['mIoU_fixed'])}")
    print(f"F1@fixed: {format_float(row['F1_fixed'])}")
    print(f"Precision@fixed: {format_float(row['P_fixed'])}")
    print(f"Recall@fixed: {format_float(row['R_fixed'])}")
    print("")
    print("ODS")
    print(f"ODS_threshold: {format_float(row['ODS_thr'])}")
    print(f"ODS_F1: {format_float(row['ODS_F1'])}")
    print(f"Precision@ODS: {format_float(row['P_ODS'])}")
    print(f"Recall@ODS: {format_float(row['R_ODS'])}")
    print(f"mIoU@ODS: {format_float(row['mIoU_ODS'])}")
    print("")
    print("Best mIoU")
    print(f"best_mIoU_threshold: {format_float(row['best_mIoU_thr'])}")
    print(f"best_mIoU: {format_float(row['best_mIoU'])}")
    print("")
    print("OIS")
    print(f"OIS_F1: {format_float(row['OIS_F1'])}")
    print("")
    print("CSV Row")
    print(",".join(CSV_FIELDS))
    print(",".join(str(row[field]) for field in CSV_FIELDS))


def main():
    parser = argparse.ArgumentParser(description="Standard offline evaluation for MixerCSeg saved predictions")
    parser.add_argument("--result_dir", required=True, help="Directory containing *_pre.png and *_lab.png")
    parser.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated thresholds. Default: 0.00 to 0.99 with step 0.01",
    )
    parser.add_argument("--fixed_threshold", default=0.5, type=float, help="Fixed threshold for fixed metrics")
    parser.add_argument("--save_csv", action=argparse.BooleanOptionalAction, default=True, help="Save summary.csv")
    parser.add_argument("--method", default="", help="Optional method name for CSV output")
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    thresholds = parse_thresholds(args.thresholds)
    pairs = load_pairs(result_dir)

    fixed = eval_threshold(pairs, args.fixed_threshold)
    ods_thr, ods, best_miou_thr, best_miou_metrics = eval_ods_and_best_miou(pairs, thresholds)
    ois_f1 = eval_ois(pairs, thresholds)

    row = {
        "method": args.method or infer_method(result_dir),
        "result_dir": result_dir,
        "mIoU_fixed": fixed["mIoU"],
        "F1_fixed": fixed["F1"],
        "P_fixed": fixed["Precision"],
        "R_fixed": fixed["Recall"],
        "ODS_thr": ods_thr,
        "ODS_F1": ods["F1"],
        "P_ODS": ods["Precision"],
        "R_ODS": ods["Recall"],
        "mIoU_ODS": ods["mIoU"],
        "best_mIoU_thr": best_miou_thr,
        "best_mIoU": best_miou_metrics["mIoU"],
        "OIS_F1": ois_f1,
    }

    print_summary(row)
    if args.save_csv:
        csv_path = os.path.join(result_dir, "summary.csv")
        save_summary_csv(csv_path, row)
        print(f"\nsaved_csv: {csv_path}")


if __name__ == "__main__":
    main()
