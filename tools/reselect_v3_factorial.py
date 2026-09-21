"""Validation-reselect the existing nine-run TripleStack-v3 factorial study."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models import build_MixerCSeg
from tools.paper_audit import (
    METRICS,
    atomic_json,
    audit_dataset,
    checkpoint_signature,
    completed,
    evaluate_test,
    fresh_attempt,
    load_checkpoint,
    log,
    model_args,
    write_tsv,
)
from util.standard_validation import evaluate_validation, make_validation_loader, selection_key


SOURCE = ROOT / "logs/20260808_005610_CrackMap_triple_stack_v3_ablation_gpu1.metrics.tsv"
MODE_ORDER = (
    "baseline",
    "id_id_id",
    "msa_id_id",
    "id_ocv_id",
    "id_id_gbc",
    "msa_ocv_id",
    "msa_id_gbc",
    "id_ocv_gbc",
    "msa_ocv_gbc",
)
FACTOR_NAMES = ("MSA", "OCV", "GBC")


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def component_bits(mode):
    if mode == "baseline":
        return {"wrapper": 0, "MSA": 0, "OCV": 0, "GBC": 0}
    parts = mode.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid V3 factorial mode: {mode}")
    expected = (("id", "msa"), ("id", "ocv"), ("id", "gbc"))
    if any(value not in choices for value, choices in zip(parts, expected)):
        raise ValueError(f"Invalid V3 factorial mode: {mode}")
    return {
        "wrapper": 1,
        "MSA": int(parts[0] == "msa"),
        "OCV": int(parts[1] == "ocv"),
        "GBC": int(parts[2] == "gbc"),
    }


def factorial_jobs(source=SOURCE):
    rows = read_tsv(source)
    by_mode = {}
    for row in rows:
        mode = "baseline" if row["model"] == "baseline" else row["mode"]
        if mode in by_mode:
            raise ValueError(f"Duplicate factorial mode in source: {mode}")
        by_mode[mode] = row
    if set(by_mode) != set(MODE_ORDER):
        missing = sorted(set(MODE_ORDER) - set(by_mode))
        extra = sorted(set(by_mode) - set(MODE_ORDER))
        raise ValueError(f"Expected exactly nine factorial rows; missing={missing}, extra={extra}")

    jobs = []
    for index, mode in enumerate(MODE_ORDER):
        row = by_mode[mode]
        if int(row["seed"]) != 42:
            raise ValueError(f"Unexpected seed for {mode}: {row['seed']}")
        jobs.append({
            "job_id": f"CrackMap_{mode}_seed42",
            "dataset": "CrackMap",
            "model": "baseline" if mode == "baseline" else "triple_stack_v3",
            "mode": mode,
            "seed": 42,
            "gpu": index % 3,
            "checkpoint": row["checkpoint"],
            "source": str(Path(source).resolve()),
            "old_metrics": {name: float(row[name]) for name in METRICS},
            **component_bits(mode),
        })
    return jobs


def validate_factorial_checkpoint(checkpoint, job, epoch=None):
    if "args" not in checkpoint:
        raise ValueError(f"Checkpoint has no args: {job['job_id']}")
    args = checkpoint["args"]
    if epoch is not None and int(checkpoint.get("epoch", -1)) != epoch:
        raise ValueError(f"Wrong checkpoint epoch for {job['job_id']}: expected {epoch}")
    if int(args.epochs) != 50 or int(args.seed) != 42 or int(args.nbins) != 180:
        raise ValueError(f"Training protocol mismatch: {job['job_id']}")
    if Path(args.dataset_path).name != "CrackMap":
        raise ValueError(f"Dataset mismatch: {job['job_id']}")
    if not math.isclose(float(args.BCELoss_ratio), 0.87, abs_tol=1e-12):
        raise ValueError(f"BCE weight mismatch: {job['job_id']}")
    if not math.isclose(float(args.DiceLoss_ratio), 0.13, abs_tol=1e-12):
        raise ValueError(f"Dice weight mismatch: {job['job_id']}")

    active = {
        name for name, value in vars(args).items()
        if name.startswith("use_") and value is True
    }
    expected = set() if job["mode"] == "baseline" else {"use_triple_stack_v3"}
    if active != expected:
        raise ValueError(f"Unexpected active flags for {job['job_id']}: {sorted(active)}")
    if expected and args.triple_stack_v3_mode != job["mode"]:
        raise ValueError(f"Wrong V3 mode for {job['job_id']}: {args.triple_stack_v3_mode}")


def select_best(records):
    if not records:
        raise ValueError("No validation records")
    return max(records, key=lambda row: selection_key(row, row["epoch"]))


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(ROOT / "dataset/CrackMap")
    atomic_json(run / "dataset_audit.json", audit)
    if audit["errors"]:
        raise ValueError("CrackMap dataset audit failed:\n" + "\n".join(audit["errors"]))

    jobs = factorial_jobs()
    for job in jobs:
        parent = (ROOT / job["checkpoint"]).parent
        paths = [parent / f"checkpoint{epoch}.pth" for epoch in range(50)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"Missing checkpoints for {job['job_id']}: {missing}")
        checkpoint = load_checkpoint(paths[-1])
        validate_factorial_checkpoint(checkpoint, job, 49)
        args = model_args(checkpoint, "CrackMap", "cpu")
        model, _ = build_MixerCSeg(args)
        model.load_state_dict(checkpoint["model"], strict=True)
        job["trajectory"] = [checkpoint_signature(path) for path in paths]
        del model, checkpoint
        log(f"PREFLIGHT OK {job['job_id']} gpu={job['gpu']} checkpoints=50")

    manifest = run / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != jobs:
        raise ValueError("Existing manifest differs; use a fresh run directory")
    atomic_json(manifest, jobs)
    log("PREPARE DONE trajectories=9")


def wait_for_gpu(gpu):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu not in (0, 1, 2) or visible != str(gpu):
        raise ValueError(f"Worker GPU mismatch: assigned={gpu}, CUDA_VISIBLE_DEVICES={visible}")
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        try:
            memory, utilization = [int(value.strip()) for value in result.stdout.strip().split(",")]
        except ValueError:
            memory, utilization = 999999, 100
        if result.returncode == 0 and memory < 1000 and utilization < 10:
            log(f"selected GPU {gpu} memory={memory}MiB utilization={utilization}%")
            return
        log(f"waiting GPU {gpu} memory={memory}MiB utilization={utilization}%")
        time.sleep(int(os.environ.get("POLL_SECONDS", "60")))


def reselect(run, job):
    job_root = run / "reselected" / job["job_id"]
    if completed(job_root):
        log(f"RESELECT SKIP {job['job_id']}")
        return

    wait_for_gpu(job["gpu"])
    attempt = fresh_attempt(job_root)
    log(f"RESELECT START {job['job_id']} gpu={job['gpu']}")
    parent = (ROOT / job["checkpoint"]).parent
    checkpoint = load_checkpoint(parent / "checkpoint0.pth")
    args = model_args(checkpoint, "CrackMap", "cuda:0")
    model, _ = build_MixerCSeg(args)
    model.to(args.device)
    loader = make_validation_loader(args)
    validation_rows = []
    start = time.monotonic()

    for epoch in range(50):
        path = parent / f"checkpoint{epoch}.pth"
        if checkpoint_signature(path) != job["trajectory"][epoch]:
            raise ValueError(f"Checkpoint changed since preflight: {path}")
        checkpoint = load_checkpoint(path)
        validate_factorial_checkpoint(checkpoint, job, epoch)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics = evaluate_validation(model, loader, args.device)
        validation_rows.append(dict(metrics, epoch=epoch, checkpoint=str(path)))
        write_tsv(attempt / "validation_epochs.tsv", validation_rows)
        if epoch % 10 == 9:
            log(
                f"RESELECT PROGRESS {job['job_id']} epochs={epoch + 1}/50 "
                f"elapsed={time.monotonic() - start:.1f}s"
            )

    best = select_best(validation_rows)
    atomic_json(attempt / "selection.json", dict(best, split="val", fixed_threshold=0.5))
    checkpoint = load_checkpoint(best["checkpoint"])
    model.load_state_dict(checkpoint["model"], strict=True)
    result_dir = attempt / "test_probability"
    test_metrics = evaluate_test(model, args, result_dir)
    record = {
        key: job[key]
        for key in (
            "job_id", "dataset", "model", "mode", "seed", "gpu",
            "wrapper", "MSA", "OCV", "GBC",
        )
    }
    record.update(
        test_metrics,
        selection_split="val",
        selection_mIoU=best["mIoU"],
        selection_F1=best["F1"],
        epoch=best["epoch"],
        checkpoint=best["checkpoint"],
        result_dir=str(result_dir),
        elapsed_seconds=time.monotonic() - start,
    )
    for name, value in job["old_metrics"].items():
        record[f"old_{name}"] = value
        record[f"reselection_delta_{name}"] = test_metrics[name] - value
    atomic_json(job_root / "completed.json", record)
    del model, checkpoint
    torch.cuda.empty_cache()
    log(
        f"RESELECT DONE {job['job_id']} epoch={best['epoch']} "
        f"mIoU={test_metrics['mIoU_fixed']:.6f} F1={test_metrics['F1_fixed']:.6f}"
    )


def factorial_effect(records, factors, metric):
    factorial = [row for row in records if row["wrapper"] == 1]
    if len(factorial) != 8:
        raise ValueError("Factorial effects require all eight wrapper configurations")
    positive = []
    negative = []
    for row in factorial:
        sign = math.prod(1 if row[factor] else -1 for factor in factors)
        (positive if sign > 0 else negative).append(float(row[metric]))
    return sum(positive) / len(positive) - sum(negative) / len(negative)


def summarize(run):
    jobs = json.loads((run / "manifest.json").read_text())
    records = []
    for job in jobs:
        job_root = run / "reselected" / job["job_id"]
        if not completed(job_root):
            raise ValueError(f"Missing successful re-evaluation: {job['job_id']}")
        records.append(json.loads((job_root / "completed.json").read_text()))
    records.sort(key=lambda row: MODE_ORDER.index(row["mode"]))
    write_tsv(run / "reselected.metrics.tsv", records)

    baseline = next(row for row in records if row["mode"] == "baseline")
    wrapper = next(row for row in records if row["mode"] == "id_id_id")
    full = next(row for row in records if row["mode"] == "msa_ocv_gbc")
    ablation = []
    for source in records:
        row = source.copy()
        for metric in METRICS:
            row[f"delta_baseline_{metric}"] = row[metric] - baseline[metric]
            row[f"delta_wrapper_{metric}"] = row[metric] - wrapper[metric]
            row[f"delta_full_{metric}"] = row[metric] - full[metric]
        ablation.append(row)
    write_tsv(run / "ablation.tsv", ablation)

    effect_rows = []
    effect_sets = (
        ("MSA",), ("OCV",), ("GBC",),
        ("MSA", "OCV"), ("MSA", "GBC"), ("OCV", "GBC"),
        ("MSA", "OCV", "GBC"),
    )
    for factors in effect_sets:
        row = {"effect": ":".join(factors), "order": len(factors)}
        for metric in METRICS:
            row[metric] = factorial_effect(records, factors, metric)
        effect_rows.append(row)
    write_tsv(run / "effects.tsv", effect_rows)

    ranked = sorted(records, key=lambda row: (row["mIoU_fixed"], row["F1_fixed"]), reverse=True)
    lines = [
        "# TripleStack-v3 validation-reselected factorial ablation",
        "",
        "Checkpoint selection uses only the CrackMap validation split. The test split is evaluated "
        "once after selection. Historical test-selected values are retained only for audit.",
        "",
        "## Results",
        "",
        "| Configuration | Epoch | mIoU | F1 | ODS | OIS | Delta mIoU vs baseline | Delta F1 vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ablation:
        lines.append(
            f"| {row['mode']} | {row['epoch']} | {row['mIoU_fixed']:.6f} | "
            f"{row['F1_fixed']:.6f} | {row['ODS_F1']:.6f} | {row['OIS_F1']:.6f} | "
            f"{row['delta_baseline_mIoU_fixed']:+.6f} | "
            f"{row['delta_baseline_F1_fixed']:+.6f} |"
        )
    lines += [
        "",
        "## Factorial effects",
        "",
        "Effects are positive-product minus negative-product group means over the eight wrapper runs.",
        "",
        "| Effect | mIoU | F1 | ODS | OIS |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in effect_rows:
        lines.append(
            f"| {row['effect']} | {row['mIoU_fixed']:+.6f} | {row['F1_fixed']:+.6f} | "
            f"{row['ODS_F1']:+.6f} | {row['OIS_F1']:+.6f} |"
        )
    lines += [
        "",
        f"Best validation-reselected configuration: `{ranked[0]['mode']}` "
        f"(mIoU={ranked[0]['mIoU_fixed']:.6f}, F1={ranked[0]['F1_fixed']:.6f}).",
        f"Full `msa_ocv_gbc` rank: {ranked.index(full) + 1}/9.",
        "",
        "This is a single-seed ablation and does not establish statistical significance. Validation "
        "reselection corrects epoch selection but cannot erase earlier architecture-search exposure "
        "to the test set.",
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    log(
        f"SUMMARY DONE rows=9 best={ranked[0]['mode']} "
        f"full_rank={ranked.index(full) + 1}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "summarize", "dry-run"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "work_dirs/v3_factorial_reselect9_gpu012",
    )
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = args.run_dir.resolve()

    if args.action == "dry-run":
        for job in factorial_jobs():
            print(
                f"DRYRUN RESELECT job={job['job_id']} checkpoints=50 "
                f"gpu={job['gpu']} selection=val test_evaluations=1"
            )
        return
    if args.action == "prepare":
        prepare(run)
        return
    if args.action == "summarize":
        summarize(run)
        return
    if args.gpu is None:
        parser.error("--gpu is required for worker")

    jobs = json.loads((run / "manifest.json").read_text())
    failures = 0
    for job in jobs:
        if job["gpu"] != args.gpu:
            continue
        try:
            reselect(run, job)
        except Exception:
            failures += 1
            log(f"FAIL action=worker job={job['job_id']}")
            traceback.print_exc()
            break
    log(f"WORKER gpu={args.gpu} failures={failures}")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
