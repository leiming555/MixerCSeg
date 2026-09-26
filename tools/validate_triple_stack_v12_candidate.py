"""Staged validation for the frozen TripleStack-v12 wfe_cpa_bcs candidate."""

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
from tools.paper_audit import (
    METRICS,
    atomic_json,
    audit_dataset,
    completed,
    evaluate_test,
    fresh_attempt,
    load_checkpoint,
    log,
    model_args,
    read_tsv,
    write_tsv,
)


MODE = "wfe_cpa_bcs"
REFERENCE_RUN = ROOT / "work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12"
SEED42_TEST_TSV = (
    ROOT / "logs/20260921_225136_TripleStackV12_full12_valselect_gpu12.test.tsv"
)
STAGE1_JOBS = [
    {"dataset": "CrackMap", "seed": 3407, "gpu": 0},
    {"dataset": "CrackMap", "seed": 2026, "gpu": 1},
    {"dataset": "CrackMap", "seed": 1234, "gpu": 2},
]
STAGE2_JOBS = [
    {"dataset": "DeepCrack", "seed": 42, "gpu": 0},
    {"dataset": "CamCrack789", "seed": 42, "gpu": 1},
]


def job_id(dataset, seed):
    return f"{dataset}_triple_stack_v12_{MODE}_seed{seed}"


def absolute_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def checkpoint_signature(path):
    path = absolute_path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def reference_records():
    rows = read_tsv(REFERENCE_RUN / "reselected.metrics.tsv")
    expected = {
        ("CrackMap", model, seed)
        for model in ("baseline", "msa_ocv_gbc")
        for seed in (42, 3407, 2026, 1234)
    }
    expected.update(
        (dataset, model, 42)
        for dataset in ("DeepCrack", "CamCrack789")
        for model in ("baseline", "msa_ocv_gbc")
    )
    records = {}
    for row in rows:
        key = (row["dataset"], row["model"], int(row["seed"]))
        if key in expected:
            records[key] = {**row, **{metric: float(row[metric]) for metric in METRICS}}
    if set(records) != expected:
        raise ValueError(f"Incomplete baseline/V3 references: missing={expected - set(records)}")
    return records


def validate_candidate_checkpoint(checkpoint, dataset, seed):
    args = checkpoint["args"]
    if int(args.epochs) != 50 or int(args.seed) != seed or int(args.nbins) != 180:
        raise ValueError("Candidate training protocol mismatch")
    if Path(args.dataset_path).name != dataset:
        raise ValueError("Candidate dataset mismatch")
    if not math.isclose(args.BCELoss_ratio, 0.87) or not math.isclose(
        args.DiceLoss_ratio, 0.13
    ):
        raise ValueError("Candidate loss weights mismatch")
    active = {
        name
        for name, value in vars(args).items()
        if name.startswith("use_") and value is True
    }
    if active != {"use_triple_stack_v12"}:
        raise ValueError(f"Unexpected active candidate flags: {active}")
    if args.triple_stack_v12_mode != MODE:
        raise ValueError("Wrong TripleStack-v12 mode")
    selection = checkpoint.get("selection", {})
    if selection.get("split") != "val" or selection.get("fixed_threshold") != 0.5:
        raise ValueError("Candidate checkpoint was not selected on validation at threshold 0.5")


def seed42_record():
    rows = [row for row in read_tsv(SEED42_TEST_TSV) if row.get("mode") == MODE]
    if len(rows) != 1:
        raise ValueError(f"Expected one imported seed42 row for {MODE}, found {len(rows)}")
    source = rows[0]
    checkpoint_path = absolute_path(source["checkpoint"])
    result_dir = absolute_path(source["result_dir"])
    completion_path = checkpoint_path.parent / "training_complete.json"
    final_checkpoint = checkpoint_path.parent / "checkpoint49.pth"
    summary_path = result_dir / "summary.csv"
    if not completion_path.is_file() or not final_checkpoint.is_file() or not summary_path.is_file():
        raise ValueError("Imported seed42 artifacts are incomplete")
    completion = json.loads(completion_path.read_text())
    checkpoint = load_checkpoint(checkpoint_path)
    validate_candidate_checkpoint(checkpoint, "CrackMap", 42)
    if completion.get("completed_epochs") != 50 or completion.get("selection") != checkpoint["selection"]:
        raise ValueError("Imported seed42 completion metadata mismatch")
    with summary_path.open(newline="") as handle:
        summary = next(csv.DictReader(handle))
    metrics = {metric: float(source[metric]) for metric in METRICS}
    if any(
        not math.isclose(metrics[metric], float(summary[metric]), abs_tol=1e-12)
        for metric in METRICS
    ):
        raise ValueError("Imported seed42 metrics do not match summary.csv")
    return {
        "job_id": job_id("CrackMap", 42),
        "dataset": "CrackMap",
        "model": MODE,
        "seed": 42,
        "gpu": int(source["gpu"]),
        **metrics,
        "checkpoint": str(checkpoint_path),
        "result_dir": str(result_dir),
        "selection_split": "val",
        "epoch": int(checkpoint["epoch"]),
        "stage": "imported_seed42",
    }


def wait_for_gpu(gpu):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu not in (0, 1, 2) or visible != str(gpu):
        raise ValueError(f"Worker GPU mismatch: physical={gpu} CUDA_VISIBLE_DEVICES={visible}")
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


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    audits = [
        audit_dataset(ROOT / "dataset" / name)
        for name in ("CrackMap", "DeepCrack", "CamCrack789")
    ]
    atomic_json(run / "dataset_audit.json", audits)
    errors = [error for report in audits for error in report["errors"]]
    if errors:
        raise ValueError("Dataset audit failed:\n" + "\n".join(errors))

    references = reference_records()
    seed42 = seed42_record()
    checkpoint_path = Path(seed42["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path)
    args = model_args(checkpoint, "CrackMap", "cpu")
    model, _ = build_MixerCSeg(args)
    model.load_state_dict(checkpoint["model"], strict=True)

    manifest = {
        "mode": MODE,
        "stage1": [dict(item, job_id=job_id(item["dataset"], item["seed"])) for item in STAGE1_JOBS],
        "stage2": [
            dict(item, job_id=job_id(item["dataset"], item["seed"]), conditional=True)
            for item in STAGE2_JOBS
        ],
        "seed42": seed42,
        "seed42_checkpoint_signature": checkpoint_signature(checkpoint_path),
        "reference_rows": len(references),
        "reference_file": checkpoint_signature(REFERENCE_RUN / "reselected.metrics.tsv"),
    }
    manifest_path = run / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new run directory")
    atomic_json(manifest_path, manifest)
    del model, checkpoint
    log("PREPARE DONE stage1=3 stage2_conditional=2 references=12")


def train_candidate(run, stage, dataset, seed, gpu):
    identifier = job_id(dataset, seed)
    job_root = run / stage / identifier
    if completed(job_root):
        log(f"SKIP {stage} {identifier}")
        return
    wait_for_gpu(gpu)

    ready = sorted(job_root.glob("attempt*/weights/*/training_complete.json"))
    if ready:
        marker = ready[-1]
        attempt = marker.parents[2]
        completion = json.loads(marker.read_text())
        checkpoint_path = absolute_path(completion["checkpoint"])
        log(f"RESUME EVAL {stage} {identifier} checkpoint={checkpoint_path}")
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
            "--use_triple_stack_v12",
            "--triple_stack_v12_mode",
            MODE,
            "--validation_selection",
        ]
        log(f"TRAIN START stage={stage} dataset={dataset} seed={seed} gpu={gpu}")
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
    validate_candidate_checkpoint(checkpoint, dataset, seed)
    if completion.get("selection") != checkpoint.get("selection"):
        raise ValueError("Training completion and selected checkpoint disagree")

    args = model_args(checkpoint, dataset, "cuda:0")
    model, _ = build_MixerCSeg(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    evaluation = fresh_attempt(attempt / "evaluations") / "test_probability"
    metrics = evaluate_test(model, args, evaluation)
    record = {
        "job_id": identifier,
        "dataset": dataset,
        "model": MODE,
        "seed": seed,
        "gpu": gpu,
        **metrics,
        "checkpoint": str(checkpoint_path),
        "result_dir": str(evaluation),
        "selection_split": "val",
        "epoch": checkpoint["epoch"],
        "stage": stage,
    }
    atomic_json(job_root / "completed.json", record)
    del model, checkpoint
    torch.cuda.empty_cache()
    log(
        f"TRAIN DONE stage={stage} dataset={dataset} seed={seed} "
        f"mIoU={metrics['mIoU_fixed']:.6f} F1={metrics['F1_fixed']:.6f}"
    )


def collect_stage(run, stage, jobs):
    rows = []
    for job in jobs:
        root = run / stage / job_id(job["dataset"], job["seed"])
        if not completed(root):
            raise ValueError(f"Missing completed result: {root}")
        rows.append(json.loads((root / "completed.json").read_text()))
    return rows


def metric_summary(model, rows):
    result = {"model": model, "n": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[metric + "_mean"] = statistics.mean(values)
        result[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def stage1_decision(candidate_rows, references):
    candidate = metric_summary(MODE, candidate_rows)
    baseline_rows = [
        references[("CrackMap", "baseline", int(row["seed"]))] for row in candidate_rows
    ]
    v3_rows = [
        references[("CrackMap", "msa_ocv_gbc", int(row["seed"]))]
        for row in candidate_rows
    ]
    baseline = metric_summary("baseline", baseline_rows)
    v3 = metric_summary("msa_ocv_gbc", v3_rows)
    wins = sum(
        float(row["mIoU_fixed"]) > float(reference["mIoU_fixed"])
        and float(row["F1_fixed"]) > float(reference["F1_fixed"])
        for row, reference in zip(candidate_rows, v3_rows)
    )
    checks = {
        "candidate_mIoU_gt_v3": candidate["mIoU_fixed_mean"] > v3["mIoU_fixed_mean"],
        "candidate_F1_gt_v3": candidate["F1_fixed_mean"] > v3["F1_fixed_mean"],
        "candidate_mIoU_gt_baseline": candidate["mIoU_fixed_mean"]
        > baseline["mIoU_fixed_mean"],
        "candidate_F1_gt_baseline": candidate["F1_fixed_mean"] > baseline["F1_fixed_mean"],
        "paired_v3_wins_at_least_3": wins >= 3,
        "ods_ois_not_both_below_v3_by_0_002": not (
            candidate["ODS_F1_mean"] < v3["ODS_F1_mean"] - 0.002
            and candidate["OIS_F1_mean"] < v3["OIS_F1_mean"] - 0.002
        ),
    }
    promoted = all(checks.values())
    return {
        "status": "stage1_promoted" if promoted else "stage1_rejected",
        "promoted": promoted,
        "checks": checks,
        "paired_v3_wins": wins,
        "candidate": candidate,
        "baseline": baseline,
        "v3": v3,
    }


def summarize_stage1(run):
    references = reference_records()
    rows = [seed42_record()] + collect_stage(run, "stage1", STAGE1_JOBS)
    rows.sort(key=lambda row: int(row["seed"]))
    write_tsv(run / "stage1_candidate_4seed.tsv", rows)
    decision = stage1_decision(rows, references)
    write_tsv(
        run / "stage1_summary.tsv",
        [decision[name] for name in ("baseline", "v3", "candidate")],
    )
    deltas = []
    for row in rows:
        for model in ("baseline", "msa_ocv_gbc"):
            reference = references[("CrackMap", model, int(row["seed"]))]
            deltas.append(
                {
                    "seed": row["seed"],
                    "reference": model,
                    **{
                        metric: float(row[metric]) - float(reference[metric])
                        for metric in METRICS
                    },
                }
            )
    write_tsv(run / "stage1_paired_deltas.tsv", deltas)
    atomic_json(run / "stage1_decision.json", decision)
    log(f"{decision['status']} paired_v3_wins={decision['paired_v3_wins']}/4")
    return decision["promoted"]


def final_decision(candidate_rows, references):
    comparisons = []
    for row in candidate_rows:
        dataset = row["dataset"]
        baseline = references[(dataset, "baseline", 42)]
        v3 = references[(dataset, "msa_ocv_gbc", 42)]
        checks = {
            "mIoU_gt_baseline": float(row["mIoU_fixed"]) > baseline["mIoU_fixed"],
            "F1_gt_baseline": float(row["F1_fixed"]) > baseline["F1_fixed"],
            "mIoU_within_v3_0_002": float(row["mIoU_fixed"])
            >= v3["mIoU_fixed"] - 0.002,
            "F1_within_v3_0_002": float(row["F1_fixed"]) >= v3["F1_fixed"] - 0.002,
        }
        comparisons.append(
            {
                "dataset": dataset,
                "candidate": row,
                "baseline": baseline,
                "v3": v3,
                "checks": checks,
            }
        )
    one_strict_v3_win = any(
        float(item["candidate"]["mIoU_fixed"]) > item["v3"]["mIoU_fixed"]
        and float(item["candidate"]["F1_fixed"]) > item["v3"]["F1_fixed"]
        for item in comparisons
    )
    promoted = all(all(item["checks"].values()) for item in comparisons) and one_strict_v3_win
    return {
        "status": "final_promoted" if promoted else "final_rejected",
        "promoted": promoted,
        "one_dataset_strictly_beats_v3": one_strict_v3_win,
        "comparisons": comparisons,
    }


def summarize_final(run):
    references = reference_records()
    rows = collect_stage(run, "stage2", STAGE2_JOBS)
    write_tsv(run / "cross_dataset_candidate.tsv", rows)
    decision = final_decision(rows, references)
    atomic_json(run / "final_decision.json", decision)
    comparison_rows = []
    for item in decision["comparisons"]:
        for model, row in (
            (MODE, item["candidate"]),
            ("baseline", item["baseline"]),
            ("msa_ocv_gbc", item["v3"]),
        ):
            comparison_rows.append(
                {
                    "dataset": item["dataset"],
                    "model": model,
                    **{metric: row[metric] for metric in METRICS},
                }
            )
    write_tsv(run / "cross_dataset_comparison.tsv", comparison_rows)
    log(decision["status"])
    return decision["promoted"]


def write_report(run, terminal_status):
    stage1 = json.loads((run / "stage1_decision.json").read_text())
    lines = [
        "# TripleStack-v12 wfe_cpa_bcs staged validation",
        "",
        f"Terminal status: `{terminal_status}`",
        "",
        "Every checkpoint was selected on validation; test was evaluated once after selection.",
        "",
        "## CrackMap four-seed summary",
        "",
        "| Model | mIoU mean +/- SD | F1 mean +/- SD | ODS mean | OIS mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("baseline", "v3", "candidate"):
        row = stage1[key]
        lines.append(
            f"| {row['model']} | {row['mIoU_fixed_mean']:.6f} +/- "
            f"{row['mIoU_fixed_std']:.6f} | {row['F1_fixed_mean']:.6f} +/- "
            f"{row['F1_fixed_std']:.6f} | {row['ODS_F1_mean']:.6f} | "
            f"{row['OIS_F1_mean']:.6f} |"
        )
    lines += [
        "",
        f"Paired wins over V3: {stage1['paired_v3_wins']}/4.",
        "",
        "Four seeds are robustness evidence, not a significance claim. The architecture was selected "
        "after prior CrackMap experiments, so this is not an untouched external test.",
    ]
    final_path = run / "final_decision.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
        lines += ["", "## Cross-dataset decision", "", f"Status: `{final['status']}`."]
        for item in final["comparisons"]:
            row = item["candidate"]
            lines.append(
                f"- {item['dataset']}: mIoU={float(row['mIoU_fixed']):.6f}, "
                f"F1={float(row['F1_fixed']):.6f}"
            )
    (run / "report.md").write_text("\n".join(lines) + "\n")


def run_profile(run, gpu):
    wait_for_gpu(gpu)
    output = run / "profile.log"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "tools/profile_model.py"),
        "--dataset_path",
        "dataset/CrackMap",
        "--nbins",
        "180",
        "--use_triple_stack_v12",
        "--triple_stack_v12_mode",
        MODE,
    ]
    with output.open("w") as handle:
        subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
    log(f"PROFILE DONE gpu={gpu} output={output}")


def worker(run, stage, gpu):
    jobs = STAGE1_JOBS if stage == "stage1" else STAGE2_JOBS
    failures = 0
    for job in jobs:
        if job["gpu"] != gpu:
            continue
        try:
            train_candidate(run, stage, job["dataset"], job["seed"], gpu)
        except Exception:
            failures += 1
            traceback.print_exc()
            break
    log(f"WORKER stage={stage} gpu={gpu} failures={failures}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["prepare", "worker", "stage1-summary", "final-summary", "profile", "dry-run"],
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "work_dirs/triple_stack_v12_wfe_cpa_bcs_staged_gpu012",
    )
    parser.add_argument("--stage", choices=["stage1", "stage2"])
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    run = args.run_dir.resolve()
    torch.set_num_threads(2)

    if args.action == "dry-run":
        for job in STAGE1_JOBS:
            print(
                f"DRYRUN stage1 dataset={job['dataset']} seed={job['seed']} "
                f"gpu={job['gpu']} epochs=50"
            )
        for job in STAGE2_JOBS:
            print(
                f"DRYRUN conditional stage2 dataset={job['dataset']} seed={job['seed']} "
                f"gpu={job['gpu']} epochs=50"
            )
    elif args.action == "prepare":
        prepare(run)
    elif args.action == "worker":
        if args.stage is None or args.gpu is None:
            parser.error("worker requires --stage and --gpu")
        sys.exit(bool(worker(run, args.stage, args.gpu)))
    elif args.action == "stage1-summary":
        promoted = summarize_stage1(run)
        write_report(run, "stage1_promoted" if promoted else "stage1_rejected")
        sys.exit(0 if promoted else 3)
    elif args.action == "final-summary":
        promoted = summarize_final(run)
        write_report(run, "final_promoted" if promoted else "final_rejected")
        sys.exit(0 if promoted else 4)
    elif args.action == "profile":
        if args.gpu is None:
            parser.error("profile requires --gpu")
        run_profile(run, args.gpu)


if __name__ == "__main__":
    main()
