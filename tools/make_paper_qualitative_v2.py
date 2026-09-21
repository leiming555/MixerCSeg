"""Create a readable qualitative figure with ground-truth-defined zoom regions."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_paper_qualitative import (  # noqa: E402
    DEFAULT_BASELINE,
    DEFAULT_PROPOSED,
    collect_descriptors,
    error_overlay,
    file_hash,
    read_gray,
    select_representatives,
)


DEFAULT_OUTPUT = ROOT / "paper_msa_ocv_gbc/figures/qualitative_comparison_v2.png"
DEFAULT_PREDICTIONS = ROOT / "paper_msa_ocv_gbc/figures/qualitative_predictions_v2.png"
DEFAULT_ERRORS = ROOT / "paper_msa_ocv_gbc/figures/qualitative_errors_v2.png"
DEFAULT_MANIFEST = ROOT / "paper_msa_ocv_gbc/figures/qualitative_selection_v2.tsv"


def choose_gt_roi(mask, crop_size=256):
    """Return the densest GT window, breaking ties toward the GT centroid."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Ground-truth mask must be two-dimensional")
    if not binary.any():
        raise ValueError("Cannot select an ROI from an empty ground-truth mask")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    height, width = binary.shape
    crop_height = min(int(crop_size), height)
    crop_width = min(int(crop_size), width)
    integral = cv2.integral(binary.astype(np.uint8), sdepth=cv2.CV_64F)
    scores = (
        integral[crop_height:, crop_width:]
        - integral[:-crop_height, crop_width:]
        - integral[crop_height:, :-crop_width]
        + integral[:-crop_height, :-crop_width]
    )
    candidates_y, candidates_x = np.nonzero(scores == scores.max())
    foreground_y, foreground_x = np.nonzero(binary)
    centroid_y = float(foreground_y.mean())
    centroid_x = float(foreground_x.mean())
    center_y = candidates_y + (crop_height - 1) / 2.0
    center_x = candidates_x + (crop_width - 1) / 2.0
    distance = (center_y - centroid_y) ** 2 + (center_x - centroid_x) ** 2
    best = int(np.lexsort((candidates_x, candidates_y, distance))[0])
    y0 = int(candidates_y[best])
    x0 = int(candidates_x[best])
    return int(x0), int(y0), int(x0 + crop_width), int(y0 + crop_height)


def crop_to_roi(array, roi):
    x0, y0, x1, y1 = (int(value) for value in roi)
    image = np.asarray(array)
    if image.ndim not in (2, 3):
        raise ValueError("ROI input must be a two- or three-dimensional image")
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError(f"ROI {roi} is outside image shape {image.shape}")
    return image[y0:y1, x0:x1]


def add_rois(selected, baseline_dir, crop_size=256):
    rows = []
    for row in selected:
        label = read_gray(Path(baseline_dir) / f"{row['name']}_lab.png") > 0
        x0, y0, x1, y1 = choose_gt_roi(label, crop_size=crop_size)
        rows.append({
            **row,
            "roi_x0": x0,
            "roi_y0": y0,
            "roi_x1": x1,
            "roi_y1": y1,
            "roi_foreground": int(label[y0:y1, x0:x1].sum()),
            "roi_rule": (
                f"max_gt_foreground_then_centroid_"
                f"{min(crop_size, label.shape[0], label.shape[1])}px"
            ),
        })
    return rows


def draw_roi(image, roi):
    marked = np.asarray(image).copy()
    x0, y0, x1, y1 = roi
    thickness = max(2, int(round(min(marked.shape[:2]) / 170)))
    cv2.rectangle(marked, (x0, y0), (x1 - 1, y1 - 1), (250, 184, 30), thickness)
    return marked


def load_case(dataset_root, baseline_dir, proposed_dir, row):
    name = row["name"]
    image_path = Path(dataset_root) / "test_img" / f"{name}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable test image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    baseline = read_gray(Path(baseline_dir) / f"{name}_pre.png") > 127
    proposed = read_gray(Path(proposed_dir) / f"{name}_pre.png") > 127
    label = read_gray(Path(baseline_dir) / f"{name}_lab.png") > 0
    if baseline.shape != label.shape or proposed.shape != label.shape:
        raise ValueError(f"Prediction shape differs from label: {name}")
    if image.shape[:2] != label.shape:
        image = cv2.resize(
            image, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    roi = tuple(row[key] for key in ("roi_x0", "roi_y0", "roi_x1", "roi_y1"))
    return image, label, baseline, proposed, roi


def style_axis(axis, is_binary):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#5F6B76")
        spine.set_linewidth(0.65)
    if is_binary:
        axis.images[-1].set_clim(0, 1)


def set_row_label(axis, row):
    axis.set_ylabel(
        f"{row['role'].title()}\n{row['name']}",
        fontsize=9,
        fontweight="semibold",
        rotation=0,
        ha="right",
        va="center",
        labelpad=10,
    )


def render_figure(dataset_root, baseline_dir, proposed_dir, selected, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10,
    })
    columns = (
        "Input + GT-only ROI",
        "Ground truth",
        "Baseline",
        "MSA-OCV-GBC",
        "Baseline errors",
        "Proposed errors",
    )
    figure, axes = plt.subplots(
        len(selected), len(columns), figsize=(13.8, 8.5), squeeze=False
    )

    for row_index, row in enumerate(selected):
        image, label, baseline, proposed, roi = load_case(
            dataset_root, baseline_dir, proposed_dir, row
        )
        panels = (
            draw_roi(image, roi),
            crop_to_roi(label, roi),
            crop_to_roi(baseline, roi),
            crop_to_roi(proposed, roi),
            crop_to_roi(error_overlay(baseline, label), roi),
            crop_to_roi(error_overlay(proposed, label), roi),
        )
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            is_binary = panel.ndim == 2
            axis.imshow(
                panel,
                cmap="gray" if is_binary else None,
                vmin=0 if is_binary else None,
                vmax=1 if is_binary else None,
                interpolation="nearest" if is_binary else "bilinear",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            style_axis(axis, is_binary)
            if row_index == 0:
                axis.set_title(columns[column_index], pad=6, fontweight="semibold")

        set_row_label(axes[row_index, 0], row)

    legend = (
        Patch(facecolor="#2DB44B", edgecolor="none", label="True positive"),
        Patch(facecolor="#E13737", edgecolor="none", label="False positive"),
        Patch(facecolor="#2D7DEB", edgecolor="none", label="False negative"),
    )
    figure.legend(
        handles=legend,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.004),
        frameon=False,
        ncol=3,
        columnspacing=1.3,
        handlelength=1.0,
        fontsize=9,
    )
    figure.text(
        0.105,
        0.018,
        "Samples and zoom regions are selected from ground truth only.",
        ha="left",
        va="center",
        fontsize=9,
        color="#3F4850",
    )
    figure.subplots_adjust(
        left=0.105, right=0.995, top=0.945, bottom=0.075, wspace=0.045, hspace=0.09
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=360, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def render_prediction_figure(dataset_root, baseline_dir, proposed_dir, selected, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    columns = ("Input + GT-only ROI", "Ground truth", "Baseline", "MSA-OCV-GBC")
    figure, axes = plt.subplots(
        len(selected), len(columns), figsize=(10.6, 9.2), squeeze=False
    )
    for row_index, row in enumerate(selected):
        image, label, baseline, proposed, roi = load_case(
            dataset_root, baseline_dir, proposed_dir, row
        )
        panels = (
            draw_roi(image, roi),
            crop_to_roi(label, roi),
            crop_to_roi(baseline, roi),
            crop_to_roi(proposed, roi),
        )
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            is_binary = panel.ndim == 2
            axis.imshow(
                panel,
                cmap="gray" if is_binary else None,
                vmin=0 if is_binary else None,
                vmax=1 if is_binary else None,
                interpolation="nearest" if is_binary else "bilinear",
            )
            style_axis(axis, is_binary)
            if row_index == 0:
                axis.set_title(columns[column_index], pad=6, fontweight="semibold")
        set_row_label(axes[row_index, 0], row)

    figure.text(
        0.132,
        0.015,
        "Selection and zoom regions use ground truth only; all predictions use threshold 0.5.",
        ha="left",
        fontsize=9,
        color="#3F4850",
    )
    figure.subplots_adjust(
        left=0.132, right=0.995, top=0.95, bottom=0.055, wspace=0.045, hspace=0.085
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=360, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def render_error_figure(dataset_root, baseline_dir, proposed_dir, selected, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    columns = ("Input crop", "Baseline errors", "Proposed errors")
    figure, axes = plt.subplots(
        len(selected), len(columns), figsize=(8.4, 10.0), squeeze=False
    )
    for row_index, row in enumerate(selected):
        image, label, baseline, proposed, roi = load_case(
            dataset_root, baseline_dir, proposed_dir, row
        )
        panels = (
            crop_to_roi(image, roi),
            crop_to_roi(error_overlay(baseline, label), roi),
            crop_to_roi(error_overlay(proposed, label), roi),
        )
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(panel, interpolation="bilinear" if column_index == 0 else "nearest")
            style_axis(axis, False)
            if row_index == 0:
                axis.set_title(columns[column_index], pad=6, fontweight="semibold")
        set_row_label(axes[row_index, 0], row)

    legend = (
        Patch(facecolor="#2DB44B", edgecolor="none", label="True positive"),
        Patch(facecolor="#E13737", edgecolor="none", label="False positive"),
        Patch(facecolor="#2D7DEB", edgecolor="none", label="False negative"),
    )
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.003),
        frameon=False,
        ncol=3,
        columnspacing=1.4,
        handlelength=1.0,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.16, right=0.995, top=0.955, bottom=0.06, wspace=0.05, hspace=0.085
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=360, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def write_manifest(path, selected, dataset_root, baseline_dir, proposed_dir):
    fields = (
        "role", "name", "foreground_ratio", "components", "fragmentation",
        "boundary_complexity", "roi_x0", "roi_y0", "roi_x1", "roi_y1",
        "roi_foreground", "roi_rule", "image", "native_label", "evaluation_label",
        "baseline_prediction", "proposed_prediction", "native_label_sha256",
        "evaluation_label_sha256", "baseline_sha256", "proposed_sha256",
    )
    output_rows = []
    for row in selected:
        name = row["name"]
        native_label = Path(dataset_root) / "test_lab" / f"{name}.png"
        evaluation_label = Path(baseline_dir) / f"{name}_lab.png"
        baseline = Path(baseline_dir) / f"{name}_pre.png"
        proposed = Path(proposed_dir) / f"{name}_pre.png"
        output_rows.append({
            **row,
            "image": str(Path(dataset_root) / "test_img" / f"{name}.png"),
            "native_label": str(native_label),
            "evaluation_label": str(evaluation_label),
            "baseline_prediction": str(baseline),
            "proposed_prediction": str(proposed),
            "native_label_sha256": file_hash(native_label),
            "evaluation_label_sha256": file_hash(evaluation_label),
            "baseline_sha256": file_hash(baseline),
            "proposed_sha256": file_hash(proposed),
        })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset/CrackMap")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--proposed-dir", type=Path, default=DEFAULT_PROPOSED)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--error-output", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    descriptors = collect_descriptors(args.dataset_root, args.baseline_dir, args.proposed_dir)
    selected = select_representatives(descriptors)
    selected = add_rois(selected, args.baseline_dir, crop_size=args.crop_size)
    render_figure(
        args.dataset_root, args.baseline_dir, args.proposed_dir, selected, args.output
    )
    render_prediction_figure(
        args.dataset_root,
        args.baseline_dir,
        args.proposed_dir,
        selected,
        args.prediction_output,
    )
    render_error_figure(
        args.dataset_root,
        args.baseline_dir,
        args.proposed_dir,
        selected,
        args.error_output,
    )
    write_manifest(
        args.manifest, selected, args.dataset_root, args.baseline_dir, args.proposed_dir
    )
    print(f"figure={args.output}")
    print(f"prediction_figure={args.prediction_output}")
    print(f"error_figure={args.error_output}")
    print(f"manifest={args.manifest}")
    for row in selected:
        print(
            f"selected role={row['role']} name={row['name']} "
            f"roi=({row['roi_x0']},{row['roi_y0']},{row['roi_x1']},{row['roi_y1']}) "
            f"roi_foreground={row['roi_foreground']}"
        )


if __name__ == "__main__":
    main()
