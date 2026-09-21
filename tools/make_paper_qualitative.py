"""Create a traceable qualitative figure from validation-selected predictions."""

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12/reselected"
DEFAULT_BASELINE = EVIDENCE_ROOT / "CrackMap_baseline_seed42/attempt0001/test_probability"
DEFAULT_PROPOSED = EVIDENCE_ROOT / "CrackMap_msa_ocv_gbc_seed42/attempt0001/test_probability"
DEFAULT_OUTPUT = ROOT / "paper_msa_ocv_gbc/figures/qualitative_comparison.png"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_mask(mask, name=""):
    binary = np.asarray(mask) > 0
    area = int(binary.sum())
    if area == 0:
        raise ValueError(f"Empty crack mask: {name}")
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    components = int(sum(stats[index, cv2.CC_STAT_AREA] >= 4 for index in range(1, count)))
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    scale = float(np.sqrt(area))
    return {
        "name": name,
        "foreground_ratio": area / binary.size,
        "components": components,
        "fragmentation": components / scale,
        "boundary_complexity": perimeter / scale,
    }


def select_representatives(descriptors):
    if len(descriptors) < 4:
        raise ValueError("At least four test masks are required")
    rules = (
        ("sparse", lambda row: (row["foreground_ratio"], row["name"])),
        ("fragmented", lambda row: (-row["fragmentation"], row["name"])),
        ("complex boundary", lambda row: (-row["boundary_complexity"], row["name"])),
        ("dense", lambda row: (-row["foreground_ratio"], row["name"])),
    )
    selected = []
    used = set()
    for role, key in rules:
        candidates = sorted(descriptors, key=key)
        choice = next((row for row in candidates if row["name"] not in used), None)
        if choice is None:
            raise ValueError("Could not select four unique representative masks")
        selected.append(dict(choice, role=role))
        used.add(choice["name"])
    return selected


def error_overlay(prediction, label):
    prediction = np.asarray(prediction, dtype=bool)
    label = np.asarray(label, dtype=bool)
    if prediction.shape != label.shape:
        raise ValueError("Prediction and label shapes differ")
    overlay = np.zeros((*label.shape, 3), dtype=np.uint8)
    overlay[prediction & label] = (45, 180, 75)       # true positive: green
    overlay[prediction & ~label] = (225, 55, 55)      # false positive: red
    overlay[~prediction & label] = (45, 125, 235)     # false negative: blue
    return overlay


def prediction_names(directory):
    return {
        path.name[:-len("_pre.png")]
        for path in Path(directory).glob("*_pre.png")
    }


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable grayscale image: {path}")
    return image


def resize_binary(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != tuple(shape):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(shape[1]), int(shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0


def collect_descriptors(dataset_root, baseline_dir, proposed_dir):
    baseline_names = prediction_names(baseline_dir)
    proposed_names = prediction_names(proposed_dir)
    source_names = {path.stem for path in (dataset_root / "test_img").glob("*.png")}
    if baseline_names != proposed_names or baseline_names != source_names:
        raise ValueError(
            "Qualitative inputs are not one-to-one: "
            f"baseline={len(baseline_names)}, proposed={len(proposed_names)}, source={len(source_names)}"
        )
    if len(source_names) != 24:
        raise ValueError(f"Expected 24 CrackMap test samples, found {len(source_names)}")

    rows = []
    for name in sorted(source_names):
        dataset_label = read_gray(dataset_root / "test_lab" / f"{name}.png")
        baseline_label = read_gray(baseline_dir / f"{name}_lab.png")
        proposed_label = read_gray(proposed_dir / f"{name}_lab.png")
        resized_label = resize_binary(dataset_label, baseline_label.shape)
        if baseline_label.shape != proposed_label.shape:
            raise ValueError(f"Saved label shapes differ: {name}")
        saved_label = baseline_label > 0
        if not np.array_equal(saved_label, proposed_label > 0):
            raise ValueError(f"Baseline and proposed labels differ: {name}")
        union = int(np.sum(resized_label | saved_label))
        overlap = int(np.sum(resized_label & saved_label))
        if union == 0 or overlap / union < 0.999:
            raise ValueError(f"Saved label does not match resized dataset label: {name}")
        rows.append(describe_mask(saved_label, name))
    return rows


def render_figure(dataset_root, baseline_dir, proposed_dir, selected, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    columns = (
        "Input", "Ground truth", "Baseline", "MSA-OCV-GBC",
        "Baseline error", "Proposed error",
    )
    figure, axes = plt.subplots(len(selected), len(columns), figsize=(12.8, 8.7))
    for row_index, row in enumerate(selected):
        name = row["name"]
        image = cv2.imread(str(dataset_root / "test_img" / f"{name}.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable test image: {name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        baseline = read_gray(baseline_dir / f"{name}_pre.png") > 127
        proposed = read_gray(proposed_dir / f"{name}_pre.png") > 127
        label = read_gray(baseline_dir / f"{name}_lab.png") > 0
        if image.shape[:2] != label.shape:
            image = cv2.resize(image, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
        if baseline.shape != label.shape or proposed.shape != label.shape:
            raise ValueError(f"Prediction shape differs from label: {name}")

        panels = (
            image,
            label,
            baseline,
            proposed,
            error_overlay(baseline, label),
            error_overlay(proposed, label),
        )
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.45)
            if row_index == 0:
                axis.set_title(columns[column_index], fontsize=9, pad=5)
        axes[row_index, 0].set_ylabel(
            f"{row['role']}\n{name}", fontsize=8, rotation=90, labelpad=7
        )

    figure.text(
        0.5,
        0.012,
        "Error maps: green = true positive, red = false positive, blue = false negative. "
        "Samples are selected from ground-truth morphology only.",
        ha="center",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.055, wspace=0.035, hspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, facecolor="white")
    plt.close(figure)


def write_manifest(path, selected, dataset_root, baseline_dir, proposed_dir):
    fields = (
        "role", "name", "foreground_ratio", "components", "fragmentation",
        "boundary_complexity", "image", "native_label", "evaluation_label",
        "baseline_prediction", "proposed_prediction", "native_label_sha256",
        "evaluation_label_sha256", "baseline_sha256", "proposed_sha256",
    )
    rows = []
    for row in selected:
        name = row["name"]
        native_label = dataset_root / "test_lab" / f"{name}.png"
        evaluation_label = baseline_dir / f"{name}_lab.png"
        baseline = baseline_dir / f"{name}_pre.png"
        proposed = proposed_dir / f"{name}_pre.png"
        rows.append({
            **row,
            "image": str(dataset_root / "test_img" / f"{name}.png"),
            "native_label": str(native_label),
            "evaluation_label": str(evaluation_label),
            "baseline_prediction": str(baseline),
            "proposed_prediction": str(proposed),
            "native_label_sha256": file_hash(native_label),
            "evaluation_label_sha256": file_hash(evaluation_label),
            "baseline_sha256": file_hash(baseline),
            "proposed_sha256": file_hash(proposed),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset/CrackMap")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--proposed-dir", type=Path, default=DEFAULT_PROPOSED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "paper_msa_ocv_gbc/figures/qualitative_selection.tsv",
    )
    args = parser.parse_args()
    descriptors = collect_descriptors(args.dataset_root, args.baseline_dir, args.proposed_dir)
    selected = select_representatives(descriptors)
    render_figure(args.dataset_root, args.baseline_dir, args.proposed_dir, selected, args.output)
    write_manifest(
        args.manifest, selected, args.dataset_root, args.baseline_dir, args.proposed_dir
    )
    print(f"figure={args.output}")
    print(f"manifest={args.manifest}")
    for row in selected:
        print(
            f"selected role={row['role']} name={row['name']} "
            f"ratio={row['foreground_ratio']:.6f} components={row['components']}"
        )


if __name__ == "__main__":
    main()
