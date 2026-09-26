"""Validate, rank, and summarize the TripleStack-v12 CrackMap search."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.experimental_triple_stack_v12 import TRIPLE_STACK_V12_MODES


STANDARD_METRICS = (
    "mIoU_fixed",
    "F1_fixed",
    "P_fixed",
    "R_fixed",
    "ODS_F1",
    "OIS_F1",
)


def assigned_gpu(mode: str) -> int:
    return 1 if TRIPLE_STACK_V12_MODES.index(mode) < 6 else 2


def write_tsv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_checkpoint(checkpoint, mode: str):
    if "args" not in checkpoint:
        raise ValueError(f"Checkpoint has no args: {mode}")
    args = checkpoint["args"]
    expected = {
        "dataset_path": "CrackMap",
        "seed": 42,
        "epochs": 50,
        "nbins": 180,
        "BCELoss_ratio": 0.87,
        "DiceLoss_ratio": 0.13,
        "validation_selection": True,
        "use_triple_stack_v12": True,
        "triple_stack_v12_mode": mode,
    }
    actual = {
        "dataset_path": Path(args.dataset_path).name,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "nbins": int(args.nbins),
        "BCELoss_ratio": float(args.BCELoss_ratio),
        "DiceLoss_ratio": float(args.DiceLoss_ratio),
        "validation_selection": bool(args.validation_selection),
        "use_triple_stack_v12": bool(args.use_triple_stack_v12),
        "triple_stack_v12_mode": args.triple_stack_v12_mode,
    }
    for key, expected_value in expected.items():
        if isinstance(expected_value, float):
            valid = math.isclose(actual[key], expected_value, rel_tol=0.0, abs_tol=1e-12)
        else:
            valid = actual[key] == expected_value
        if not valid:
            raise ValueError(f"Protocol mismatch for {mode}: {key}={actual[key]!r}")
    prohibited = (
        "use_ccem",
        "use_tversky",
        "use_boundary_loss",
        "use_exp_module",
        "use_exp_second_module",
        "use_exp_third_module",
        "use_paper_stack",
        "use_drsgi",
        "use_rsgdi_v2",
        "use_triple_stack_v3",
        "use_triple_stack_v4",
        "use_triple_stack_v5",
        "use_triple_stack_v6",
        "use_triple_stack_v7",
        "use_triple_stack_v8",
        "use_triple_stack_v9",
        "use_triple_stack_v10",
        "use_triple_stack_v11",
    )
    active = [name for name in prohibited if bool(getattr(args, name, False))]
    if active:
        raise ValueError(f"Unexpected active modules for {mode}: {active}")


def completed_run(job_root: Path, mode: str):
    candidates = sorted(
        job_root.glob("*/training_complete.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    errors = []
    for completion_path in candidates:
        try:
            completion = json.loads(completion_path.read_text())
            selection = completion["selection"]
            checkpoint_path = Path(completion["checkpoint"])
            if int(completion["completed_epochs"]) != 50:
                raise ValueError("completed_epochs is not 50")
            if selection.get("split") != "val" or selection.get("fixed_threshold") != 0.5:
                raise ValueError("checkpoint was not selected on val at threshold 0.5")
            if not checkpoint_path.is_file() or checkpoint_path.name != "checkpoint_best_val.pth":
                raise ValueError("checkpoint_best_val.pth is missing")
            if not (completion_path.parent / "checkpoint49.pth").is_file():
                raise ValueError("checkpoint49.pth is missing")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            validate_checkpoint(checkpoint, mode)
            values = [float(selection[name]) for name in ("mIoU", "F1", "Precision", "Recall")]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite validation metric")
            return completion_path, completion
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{completion_path}: {exc}")
    detail = "; ".join(errors) if errors else "no training_complete.json"
    raise ValueError(f"No valid completed V12 run for {mode}: {detail}")


def collect_validation(out_root: Path):
    rows = []
    for mode in TRIPLE_STACK_V12_MODES:
        completion_path, completion = completed_run(out_root / f"v12_{mode}_seed42", mode)
        selection = completion["selection"]
        rows.append({
            "mode": mode,
            "gpu": assigned_gpu(mode),
            "epoch": int(selection["epoch"]),
            "val_mIoU": float(selection["mIoU"]),
            "val_F1": float(selection["F1"]),
            "val_Precision": float(selection["Precision"]),
            "val_Recall": float(selection["Recall"]),
            "checkpoint": completion["checkpoint"],
            "run_dir": str(completion_path.parent),
        })
    rows.sort(key=lambda row: (row["val_mIoU"], row["val_F1"], -row["epoch"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def collect_command(args):
    rows = collect_validation(Path(args.out_root))
    fields = (
        "rank", "mode", "gpu", "epoch", "val_mIoU", "val_F1",
        "val_Precision", "val_Recall", "checkpoint", "run_dir",
    )
    write_tsv(Path(args.validation_tsv), rows, fields)
    write_tsv(Path(args.top_tsv), rows[:args.top_k], fields)
    for row in rows:
        print(
            f"{row['rank']}\t{row['mode']}\t"
            f"val_mIoU={row['val_mIoU']:.6f}\tval_F1={row['val_F1']:.6f}"
        )


def standard_row(summary_path: Path):
    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one standard-evaluation row: {summary_path}")
    return rows[0]


def reference_row(reference_path: Path):
    matches = [
        row for row in read_tsv(reference_path)
        if row.get("job_id") == "CrackMap_msa_ocv_gbc_seed42"
    ]
    if len(matches) != 1:
        raise ValueError("V3 seed42 validation-selected reference is missing or duplicated")
    return matches[0]


def finalize_command(args):
    top_rows = read_tsv(Path(args.top_tsv))
    if len(top_rows) != args.top_k:
        raise ValueError(f"Expected exactly {args.top_k} validation-selected candidates")
    reference = reference_row(Path(args.reference_tsv))
    result_root = Path(args.result_root)
    rows = []
    for validation_row in top_rows:
        mode = validation_row["mode"]
        standard = standard_row(result_root / f"v12_{mode}_seed42" / "summary.csv")
        row = {
            "validation_rank": int(validation_row["rank"]),
            "mode": mode,
            "gpu": int(validation_row["gpu"]),
            "val_mIoU": float(validation_row["val_mIoU"]),
            "val_F1": float(validation_row["val_F1"]),
            "checkpoint": validation_row["checkpoint"],
            "result_dir": standard["result_dir"],
        }
        for name in STANDARD_METRICS:
            row[name] = float(standard[name])
            row[f"delta_{name}"] = row[name] - float(reference[name])
        row["promoted_over_v3_seed42"] = (
            row["mIoU_fixed"] > float(reference["mIoU_fixed"])
            and row["F1_fixed"] > float(reference["F1_fixed"])
            and not (
                row["delta_ODS_F1"] < -0.002
                and row["delta_OIS_F1"] < -0.002
            )
        )
        rows.append(row)
    rows.sort(key=lambda row: (row["mIoU_fixed"], row["F1_fixed"]), reverse=True)
    fields = (
        "validation_rank", "mode", "gpu", "val_mIoU", "val_F1",
        *STANDARD_METRICS,
        *(f"delta_{name}" for name in STANDARD_METRICS),
        "promoted_over_v3_seed42", "checkpoint", "result_dir",
    )
    write_tsv(Path(args.test_tsv), rows, fields)

    comparison = [{
        "rank": 0,
        "model": "TripleStack-v3",
        "mode": "msa_ocv_gbc",
        **{name: float(reference[name]) for name in STANDARD_METRICS},
        **{f"delta_{name}": 0.0 for name in STANDARD_METRICS},
        "promoted_over_v3_seed42": "reference",
    }]
    for rank, row in enumerate(rows, 1):
        comparison.append({
            "rank": rank,
            "model": "TripleStack-v12",
            "mode": row["mode"],
            **{name: row[name] for name in STANDARD_METRICS},
            **{f"delta_{name}": row[f"delta_{name}"] for name in STANDARD_METRICS},
            "promoted_over_v3_seed42": row["promoted_over_v3_seed42"],
        })
    comparison_fields = (
        "rank", "model", "mode", *STANDARD_METRICS,
        *(f"delta_{name}" for name in STANDARD_METRICS),
        "promoted_over_v3_seed42",
    )
    write_tsv(Path(args.comparison_tsv), comparison, comparison_fields)
    best = rows[0]
    Path(args.best_mode).write_text(best["mode"] + "\n")
    print(
        f"BEST_V12_MODE={best['mode']} mIoU={best['mIoU_fixed']:.6f} "
        f"F1={best['F1_fixed']:.6f} promoted={best['promoted_over_v3_seed42']}"
    )


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--out-root", required=True)
    collect.add_argument("--validation-tsv", required=True)
    collect.add_argument("--top-tsv", required=True)
    collect.add_argument("--top-k", type=int, default=2)
    collect.set_defaults(func=collect_command)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--top-tsv", required=True)
    finalize.add_argument("--result-root", required=True)
    finalize.add_argument("--reference-tsv", required=True)
    finalize.add_argument("--test-tsv", required=True)
    finalize.add_argument("--comparison-tsv", required=True)
    finalize.add_argument("--best-mode", required=True)
    finalize.add_argument("--top-k", type=int, default=2)
    finalize.set_defaults(func=finalize_command)
    return parser


def main():
    args = get_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
