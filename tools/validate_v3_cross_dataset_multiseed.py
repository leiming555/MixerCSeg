"""Paired three-seed DeepCrack/CamCrack789 validation for baseline and V3."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models import build_MixerCSeg
from tools.eval_topology import evaluate_result_dir
from tools.paper_audit import (
    METRICS,
    atomic_json,
    audit_dataset,
    evaluate_test,
    fresh_attempt,
    load_checkpoint,
    log,
    model_args,
    read_tsv,
    write_tsv,
)


REFERENCE_RUN = ROOT / "work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12"
DATASETS = ("DeepCrack", "CamCrack789")
MODELS = ("baseline", "msa_ocv_gbc")
ALL_SEEDS = (42, 3407, 2026)
NEW_SEEDS = (3407, 2026)
TOPOLOGY_METRICS = (
    "clDice",
    "Boundary_F1",
    "Boundary_Precision",
    "Boundary_Recall",
    "Component_MAE",
    "Endpoint_MAE",
)
HIGHER_IS_BETTER = set(METRICS) | {
    "clDice",
    "Boundary_F1",
    "Boundary_Precision",
    "Boundary_Recall",
}
ALL_METRICS = tuple(METRICS) + TOPOLOGY_METRICS

# The mapping balances two slower CamCrack789 pairs against four DeepCrack jobs.
JOBS = (
    {"dataset": "CamCrack789", "model": "baseline", "seed": 3407, "gpu": 0},
    {"dataset": "CamCrack789", "model": "msa_ocv_gbc", "seed": 3407, "gpu": 0},
    {"dataset": "CamCrack789", "model": "baseline", "seed": 2026, "gpu": 1},
    {"dataset": "DeepCrack", "model": "baseline", "seed": 3407, "gpu": 1},
    {"dataset": "DeepCrack", "model": "msa_ocv_gbc", "seed": 3407, "gpu": 1},
    {"dataset": "CamCrack789", "model": "msa_ocv_gbc", "seed": 2026, "gpu": 2},
    {"dataset": "DeepCrack", "model": "baseline", "seed": 2026, "gpu": 2},
    {"dataset": "DeepCrack", "model": "msa_ocv_gbc", "seed": 2026, "gpu": 2},
)


def job_id(dataset, model, seed):
    return f"{dataset}_{model}_seed{seed}"


def absolute_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def checkpoint_signature(path):
    path = absolute_path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def seed42_reference_rows():
    rows = read_tsv(REFERENCE_RUN / "reselected.metrics.tsv")
    expected = {(dataset, model, 42) for dataset in DATASETS for model in MODELS}
    records = {}
    for row in rows:
        key = (row["dataset"], row["model"], int(row["seed"]))
        if key not in expected:
            continue
        records[key] = {
            "job_id": job_id(*key),
            "dataset": key[0],
            "model": key[1],
            "seed": key[2],
            "gpu": int(row["gpu"]),
            **{metric: float(row[metric]) for metric in METRICS},
            "checkpoint": str(absolute_path(row["checkpoint"])),
            "result_dir": str(absolute_path(row["result_dir"])),
            "selection_split": "val",
            "epoch": int(row["epoch"]),
            "stage": "imported_seed42",
        }
    if set(records) != expected:
        raise ValueError(f"Incomplete seed42 references: missing={expected - set(records)}")
    return records


def validate_checkpoint(checkpoint, model, dataset, seed, require_selection=True):
    args = checkpoint["args"]
    if int(args.epochs) != 50 or int(args.seed) != seed or int(args.nbins) != 180:
        raise ValueError("Cross-dataset training protocol mismatch")
    if Path(args.dataset_path).name != dataset:
        raise ValueError("Cross-dataset checkpoint dataset mismatch")
    if not math.isclose(float(args.BCELoss_ratio), 0.87, abs_tol=1e-12) or not math.isclose(
        float(args.DiceLoss_ratio), 0.13, abs_tol=1e-12
    ):
        raise ValueError("Cross-dataset loss weights mismatch")
    active = {
        name
        for name, value in vars(args).items()
        if name.startswith("use_") and value is True
    }
    expected = set() if model == "baseline" else {"use_triple_stack_v3"}
    if active != expected:
        raise ValueError(f"Unexpected active model flags: {active}")
    if model == "msa_ocv_gbc" and args.triple_stack_v3_mode != model:
        raise ValueError("Wrong TripleStack-v3 mode")
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if require_selection:
        selection = checkpoint.get("selection", {})
        if selection.get("split") != "val" or selection.get("fixed_threshold") != 0.5:
            raise ValueError("Checkpoint was not validation-selected at threshold 0.5")


def validate_result_dir(path):
    path = absolute_path(path).resolve()
    if not (path / "summary.csv").is_file():
        raise ValueError(f"Missing standard evaluation: {path}")
    labs = list(path.glob("*_lab.png"))
    predictions = list(path.glob("*_pre.png"))
    if not labs or len(labs) != len(predictions):
        raise ValueError(f"Incomplete probability maps: {path}")
    return path


def wait_for_gpu(gpu):
    if gpu not in (0, 1, 2) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError(f"Worker GPU mismatch: physical={gpu}")
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
            memory, utilization = [
                int(value.strip()) for value in result.stdout.strip().split(",")
            ]
        except ValueError:
            memory, utilization = 999999, 100
        if result.returncode == 0 and memory < 1000 and utilization < 10:
            log(f"selected GPU {gpu} memory={memory}MiB utilization={utilization}%")
            return
        log(f"waiting GPU {gpu} memory={memory} utilization={utilization}")
        time.sleep(int(os.environ.get("POLL_SECONDS", "60")))


def topology_metrics(record):
    result = evaluate_result_dir(
        validate_result_dir(record["result_dir"]),
        method=record["job_id"],
        threshold=0.5,
        boundary_tolerance_px=2,
    )
    return {metric: float(result[metric]) for metric in TOPOLOGY_METRICS}


def job_complete(job_root):
    marker = job_root / "completed.json"
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
        result_dir = validate_result_dir(record["result_dir"])
        checkpoint = Path(record["checkpoint"])
        if (
            not checkpoint.is_file()
            or not (checkpoint.parent / "checkpoint49.pth").is_file()
            or record.get("selection_split") != "val"
        ):
            return False
        with (result_dir / "summary.csv").open(newline="") as handle:
            saved = next(csv.DictReader(handle))
        return all(
            math.isfinite(float(record[metric]))
            and (
                metric in TOPOLOGY_METRICS
                or math.isclose(float(record[metric]), float(saved[metric]), abs_tol=1e-12)
            )
            for metric in ALL_METRICS
        )
    except (KeyError, ValueError, OSError, StopIteration):
        return False


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    audits = [audit_dataset(ROOT / "dataset" / dataset) for dataset in DATASETS]
    atomic_json(run / "dataset_audit.json", audits)
    errors = [error for audit in audits for error in audit["errors"]]
    if errors:
        raise ValueError("Dataset audit failed:\n" + "\n".join(errors))

    references = seed42_reference_rows()
    for key, record in references.items():
        checkpoint = load_checkpoint(record["checkpoint"])
        validate_checkpoint(checkpoint, key[1], key[0], key[2], require_selection=False)
        args = model_args(checkpoint, key[0], "cpu")
        model, _ = build_MixerCSeg(args)
        model.load_state_dict(checkpoint["model"], strict=True)
        if not (Path(record["checkpoint"]).parent / "checkpoint49.pth").is_file():
            raise ValueError(f"Incomplete seed42 trajectory: {record['job_id']}")
        validate_result_dir(record["result_dir"])
        del model, checkpoint

    manifest = {
        "protocol": {
            "epochs": 50,
            "nbins": 180,
            "BCELoss_ratio": 0.87,
            "DiceLoss_ratio": 0.13,
            "selection_split": "val",
            "test_evaluations_after_selection": 1,
            "topology_threshold": 0.5,
            "boundary_tolerance_px": 2,
        },
        "jobs": [dict(job, job_id=job_id(job["dataset"], job["model"], job["seed"])) for job in JOBS],
        "imported_seed42": list(references.values()),
        "reference_file": checkpoint_signature(REFERENCE_RUN / "reselected.metrics.tsv"),
    }
    manifest_path = run / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new run directory")
    atomic_json(manifest_path, manifest)
    log("PREPARE DONE new_jobs=8 imported_seed42=4 datasets=2")


def train_job(run, job):
    dataset = job["dataset"]
    model_name = job["model"]
    seed = int(job["seed"])
    gpu = int(job["gpu"])
    identifier = job_id(dataset, model_name, seed)
    job_root = run / "trained" / identifier
    if job_complete(job_root):
        log(f"SKIP {identifier}")
        return
    wait_for_gpu(gpu)

    ready = sorted(job_root.glob("attempt*/weights/*/training_complete.json"))
    if ready:
        marker = ready[-1]
        attempt = marker.parents[2]
        completion = json.loads(marker.read_text())
        checkpoint_path = absolute_path(completion["checkpoint"])
        log(f"RESUME EVAL {identifier} checkpoint={checkpoint_path}")
    else:
        attempt = fresh_attempt(job_root)
        command = [
            sys.executable,
            "-u",
            str(ROOT / "main.py"),
            "--dataset_path",
            f"dataset/{dataset}",
            "--nbins",
            "180",
            "--seed",
            str(seed),
            "--epochs",
            "50",
            "--BCELoss_ratio",
            "0.87",
            "--DiceLoss_ratio",
            "0.13",
            "--output_dir",
            str(attempt / "weights"),
            "--validation_selection",
        ]
        if model_name == "msa_ocv_gbc":
            command.extend(
                ["--use_triple_stack_v3", "--triple_stack_v3_mode", "msa_ocv_gbc"]
            )
        log(f"TRAIN START dataset={dataset} model={model_name} seed={seed} gpu={gpu}")
        with (attempt / "train.log").open("w") as handle:
            subprocess.run(
                command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT
            )
        markers = list((attempt / "weights").glob("*/training_complete.json"))
        if len(markers) != 1:
            raise ValueError("Expected exactly one successful training marker")
        marker = markers[0]
        completion = json.loads(marker.read_text())
        checkpoint_path = absolute_path(completion["checkpoint"])

    if completion.get("completed_epochs") != 50 or not (marker.parent / "checkpoint49.pth").is_file():
        raise ValueError("Training is not complete")
    checkpoint = load_checkpoint(checkpoint_path)
    validate_checkpoint(checkpoint, model_name, dataset, seed)
    if completion.get("selection") != checkpoint.get("selection"):
        raise ValueError("Training completion and selected checkpoint disagree")

    args = model_args(checkpoint, dataset, "cuda:0")
    model, _ = build_MixerCSeg(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    evaluation = fresh_attempt(attempt / "evaluations") / "test_probability"
    pixel = evaluate_test(model, args, evaluation)
    record = {
        "job_id": identifier,
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "gpu": gpu,
        **pixel,
        "checkpoint": str(checkpoint_path),
        "result_dir": str(evaluation),
        "selection_split": "val",
        "epoch": int(checkpoint["epoch"]),
        "stage": "trained",
    }
    record.update(topology_metrics(record))
    atomic_json(job_root / "completed.json", record)
    del model, checkpoint
    torch.cuda.empty_cache()
    log(
        f"TRAIN DONE dataset={dataset} model={model_name} seed={seed} "
        f"mIoU={pixel['mIoU_fixed']:.6f} F1={pixel['F1_fixed']:.6f} "
        f"clDice={record['clDice']:.6f} BF1={record['Boundary_F1']:.6f}"
    )


def worker(run, gpu):
    failures = 0
    for job in JOBS:
        if job["gpu"] != gpu:
            continue
        try:
            train_job(run, job)
        except Exception:
            failures += 1
            traceback.print_exc()
            break
    log(f"WORKER gpu={gpu} failures={failures}")
    return failures


def collect_rows(run):
    references = seed42_reference_rows()
    rows = []
    for key in sorted(references):
        record = references[key]
        record.update(topology_metrics(record))
        rows.append(record)
    for job in JOBS:
        root = run / "trained" / job_id(job["dataset"], job["model"], job["seed"])
        if not job_complete(root):
            raise ValueError(f"Missing completed result: {root}")
        rows.append(json.loads((root / "completed.json").read_text()))
    keys = {(row["dataset"], row["model"], int(row["seed"])) for row in rows}
    expected = {(dataset, model, seed) for dataset in DATASETS for model in MODELS for seed in ALL_SEEDS}
    if keys != expected or len(rows) != len(expected):
        raise ValueError(f"Incomplete result set: missing={expected - keys}, extra={keys - expected}")
    return sorted(rows, key=lambda row: (row["dataset"], row["model"], int(row["seed"])))


def metric_summary(rows):
    output = []
    for dataset in DATASETS:
        for model_name in MODELS:
            group = [row for row in rows if row["dataset"] == dataset and row["model"] == model_name]
            summary = {"dataset": dataset, "model": model_name, "n": len(group)}
            for metric in ALL_METRICS:
                values = [float(row[metric]) for row in group]
                summary[metric + "_mean"] = statistics.mean(values)
                summary[metric + "_std"] = statistics.stdev(values)
            output.append(summary)
    return output


def paired_deltas(rows):
    lookup = {(row["dataset"], row["model"], int(row["seed"])): row for row in rows}
    output = []
    for dataset in DATASETS:
        for seed in ALL_SEEDS:
            baseline = lookup[(dataset, "baseline", seed)]
            model = lookup[(dataset, "msa_ocv_gbc", seed)]
            output.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    **{
                        metric: (
                            float(model[metric]) - float(baseline[metric])
                            if metric in HIGHER_IS_BETTER
                            else float(baseline[metric]) - float(model[metric])
                        )
                        for metric in ALL_METRICS
                    },
                }
            )
    return output


def paired_summary(delta_rows):
    output = []
    t_critical_df2 = 4.302652729911275
    for dataset in DATASETS:
        group = [row for row in delta_rows if row["dataset"] == dataset]
        for metric in ALL_METRICS:
            values = [float(row[metric]) for row in group]
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            half_width = t_critical_df2 * std / math.sqrt(len(values))
            output.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n": len(values),
                    "delta_direction": "positive_is_better",
                    "delta_mean": mean,
                    "delta_std": std,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "wins": sum(value > 0 for value in values),
                }
            )
    return output


def write_report(run, summaries, paired):
    summary_lookup = {(row["dataset"], row["model"]): row for row in summaries}
    delta_lookup = {(row["dataset"], row["metric"]): row for row in paired}
    lines = [
        "# V3 cross-dataset paired three-seed validation",
        "",
        "All checkpoints were selected on validation at threshold 0.5. Test was evaluated once after selection. "
        "Topology metrics use the same fixed-threshold probability maps; Boundary F1 tolerance is two pixels.",
        "",
        "Positive paired deltas indicate improvement. The three-pair confidence intervals are descriptive only.",
        "",
        "| Dataset | Model | mIoU | F1 | ODS | OIS | clDice | Boundary F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for model_name in MODELS:
            row = summary_lookup[(dataset, model_name)]
            lines.append(
                f"| {dataset} | {model_name} | {row['mIoU_fixed_mean']:.6f} +/- {row['mIoU_fixed_std']:.6f} | "
                f"{row['F1_fixed_mean']:.6f} +/- {row['F1_fixed_std']:.6f} | "
                f"{row['ODS_F1_mean']:.6f} | {row['OIS_F1_mean']:.6f} | "
                f"{row['clDice_mean']:.6f} | {row['Boundary_F1_mean']:.6f} |"
            )
    lines += [
        "",
        "## Paired V3 minus baseline",
        "",
        "| Dataset | Metric | Mean delta | 95% descriptive CI | Wins |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for metric in ("mIoU_fixed", "F1_fixed", "ODS_F1", "OIS_F1", "clDice", "Boundary_F1"):
            row = delta_lookup[(dataset, metric)]
            lines.append(
                f"| {dataset} | {metric} | {row['delta_mean']:+.6f} | "
                f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['wins']}/3 |"
            )
    (run / "report.md").write_text("\n".join(lines) + "\n")


def collect(run):
    rows = collect_rows(run)
    summaries = metric_summary(rows)
    deltas = paired_deltas(rows)
    paired = paired_summary(deltas)
    write_tsv(run / "per_run.tsv", rows)
    write_tsv(run / "three_seed_summary.tsv", summaries)
    write_tsv(run / "paired_deltas.tsv", deltas)
    write_tsv(run / "paired_summary.tsv", paired)
    write_report(run, summaries, paired)
    atomic_json(run / "completed.json", {"status": "complete", "records": len(rows), "new_trainings": len(JOBS)})
    log("CROSS DATASET MULTISEED COMPLETE records=12 new_trainings=8 failures=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "worker", "collect", "dry-run"])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "work_dirs/v3_cross_dataset_multiseed_gpu012",
    )
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    run = args.run_dir.resolve()
    torch.set_num_threads(2)

    if args.action == "dry-run":
        for job in JOBS:
            print(
                f"DRYRUN dataset={job['dataset']} model={job['model']} "
                f"seed={job['seed']} gpu={job['gpu']} epochs=50"
            )
        print("DRYRUN imported dataset=DeepCrack,CamCrack789 models=baseline,msa_ocv_gbc seed=42")
    elif args.action == "prepare":
        prepare(run)
    elif args.action == "worker":
        if args.gpu is None:
            parser.error("worker requires --gpu")
        sys.exit(bool(worker(run, args.gpu)))
    elif args.action == "collect":
        collect(run)


if __name__ == "__main__":
    main()
