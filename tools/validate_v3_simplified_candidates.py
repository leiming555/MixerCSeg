"""Staged validation for the two strongest simplified TripleStack-v3 candidates."""

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


CANDIDATES = ("id_ocv_gbc", "msa_id_gbc")
REFERENCE_RUN = ROOT / "work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12"
FACTORIAL_TSV = ROOT / "work_dirs/20260919_173408_v3_factorial_reselect9_gpu012/ablation.tsv"
STAGE1_JOBS = [
    {"mode": mode, "dataset": "CrackMap", "seed": seed, "gpu": gpu}
    for seed, gpu in ((3407, 0), (2026, 1), (1234, 2))
    for mode in CANDIDATES
]
STAGE2_DATASETS = (("DeepCrack", 0), ("CamCrack789", 1))


def job_id(mode, dataset, seed):
    return f"{dataset}_triple_stack_v3_{mode}_seed{seed}"


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
        for dataset, _ in STAGE2_DATASETS
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


def validate_candidate_checkpoint(checkpoint, mode, dataset, seed, require_selection=True):
    args = checkpoint["args"]
    if int(args.epochs) != 50 or int(args.seed) != seed or int(args.nbins) != 180:
        raise ValueError("Candidate training protocol mismatch")
    if Path(args.dataset_path).name != dataset:
        raise ValueError("Candidate dataset mismatch")
    if not math.isclose(float(args.BCELoss_ratio), 0.87, abs_tol=1e-12) or not math.isclose(
        float(args.DiceLoss_ratio), 0.13, abs_tol=1e-12
    ):
        raise ValueError("Candidate loss weights mismatch")
    active = {
        name
        for name, value in vars(args).items()
        if name.startswith("use_") and value is True
    }
    if active != {"use_triple_stack_v3"}:
        raise ValueError(f"Unexpected active candidate flags: {active}")
    if args.triple_stack_v3_mode != mode or mode not in CANDIDATES:
        raise ValueError("Wrong simplified TripleStack-v3 mode")
    if require_selection:
        selection = checkpoint.get("selection", {})
        if selection.get("split") != "val" or selection.get("fixed_threshold") != 0.5:
            raise ValueError("Candidate checkpoint was not validation-selected at threshold 0.5")


def imported_seed42_record(mode):
    rows = [row for row in read_tsv(FACTORIAL_TSV) if row.get("mode") == mode]
    if len(rows) != 1:
        raise ValueError(f"Expected one imported seed42 row for {mode}, found {len(rows)}")
    source = rows[0]
    checkpoint_path = absolute_path(source["checkpoint"])
    result_dir = absolute_path(source["result_dir"])
    summary_path = result_dir / "summary.csv"
    if not checkpoint_path.is_file() or not (checkpoint_path.parent / "checkpoint49.pth").is_file():
        raise ValueError(f"Imported seed42 trajectory is incomplete for {mode}")
    if not summary_path.is_file():
        raise ValueError(f"Imported seed42 standard evaluation is missing for {mode}")
    checkpoint = load_checkpoint(checkpoint_path)
    validate_candidate_checkpoint(checkpoint, mode, "CrackMap", 42, require_selection=False)
    with summary_path.open(newline="") as handle:
        summary = next(csv.DictReader(handle))
    metrics = {metric: float(source[metric]) for metric in METRICS}
    if any(
        not math.isclose(metrics[metric], float(summary[metric]), abs_tol=1e-12)
        for metric in METRICS
    ):
        raise ValueError(f"Imported seed42 metrics do not match summary.csv for {mode}")
    return {
        "job_id": job_id(mode, "CrackMap", 42),
        "dataset": "CrackMap",
        "model": mode,
        "mode": mode,
        "seed": 42,
        "gpu": int(source["gpu"]),
        **metrics,
        "checkpoint": str(checkpoint_path),
        "result_dir": str(result_dir),
        "selection_split": "val",
        "epoch": int(source["epoch"]),
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


def stage2_jobs(run):
    selection = run / "selected_mode.txt"
    if not selection.is_file():
        raise ValueError("No promoted simplified V3 mode was selected")
    mode = selection.read_text().strip()
    if mode not in CANDIDATES:
        raise ValueError(f"Invalid selected mode: {mode}")
    return [
        {"mode": mode, "dataset": dataset, "seed": 42, "gpu": gpu}
        for dataset, gpu in STAGE2_DATASETS
    ]


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
    imported = {mode: imported_seed42_record(mode) for mode in CANDIDATES}
    for mode, record in imported.items():
        checkpoint = load_checkpoint(record["checkpoint"])
        args = model_args(checkpoint, "CrackMap", "cpu")
        model, _ = build_MixerCSeg(args)
        model.load_state_dict(checkpoint["model"], strict=True)
        del model, checkpoint

    manifest = {
        "candidates": list(CANDIDATES),
        "stage1": [dict(job, job_id=job_id(job["mode"], job["dataset"], job["seed"])) for job in STAGE1_JOBS],
        "stage2": [
            {"dataset": dataset, "seed": 42, "gpu": gpu, "conditional": True}
            for dataset, gpu in STAGE2_DATASETS
        ],
        "imported_seed42": imported,
        "imported_checkpoint_signatures": {
            mode: checkpoint_signature(record["checkpoint"])
            for mode, record in imported.items()
        },
        "reference_rows": len(references),
        "reference_file": checkpoint_signature(REFERENCE_RUN / "reselected.metrics.tsv"),
    }
    manifest_path = run / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new run directory")
    atomic_json(manifest_path, manifest)
    log("PREPARE DONE stage1=6 stage2_conditional=2 candidates=2 references=12")


def train_candidate(run, stage, mode, dataset, seed, gpu):
    identifier = job_id(mode, dataset, seed)
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
            "--use_triple_stack_v3",
            "--triple_stack_v3_mode",
            mode,
            "--validation_selection",
        ]
        log(
            f"TRAIN START stage={stage} mode={mode} dataset={dataset} "
            f"seed={seed} gpu={gpu}"
        )
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
    validate_candidate_checkpoint(checkpoint, mode, dataset, seed)
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
        "model": mode,
        "mode": mode,
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
        f"TRAIN DONE stage={stage} mode={mode} dataset={dataset} seed={seed} "
        f"mIoU={metrics['mIoU_fixed']:.6f} F1={metrics['F1_fixed']:.6f}"
    )


def collect_stage1(run, mode):
    rows = [imported_seed42_record(mode)]
    for job in STAGE1_JOBS:
        if job["mode"] != mode:
            continue
        root = run / "stage1" / job_id(mode, job["dataset"], job["seed"])
        if not completed(root):
            raise ValueError(f"Missing completed result: {root}")
        rows.append(json.loads((root / "completed.json").read_text()))
    return sorted(rows, key=lambda row: int(row["seed"]))


def collect_stage2(run):
    rows = []
    for job in stage2_jobs(run):
        root = run / "stage2" / job_id(job["mode"], job["dataset"], job["seed"])
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


def candidate_decision(mode, rows, references):
    candidate = metric_summary(mode, rows)
    baseline_rows = [
        references[("CrackMap", "baseline", int(row["seed"]))] for row in rows
    ]
    v3_rows = [
        references[("CrackMap", "msa_ocv_gbc", int(row["seed"]))] for row in rows
    ]
    baseline = metric_summary("baseline", baseline_rows)
    v3 = metric_summary("msa_ocv_gbc", v3_rows)
    wins = sum(
        float(row["mIoU_fixed"]) > float(reference["mIoU_fixed"])
        and float(row["F1_fixed"]) > float(reference["F1_fixed"])
        for row, reference in zip(rows, v3_rows)
    )
    checks = {
        "candidate_mIoU_gt_v3": candidate["mIoU_fixed_mean"] > v3["mIoU_fixed_mean"],
        "candidate_F1_gt_v3": candidate["F1_fixed_mean"] > v3["F1_fixed_mean"],
        "candidate_mIoU_gt_baseline": candidate["mIoU_fixed_mean"]
        > baseline["mIoU_fixed_mean"],
        "candidate_F1_gt_baseline": candidate["F1_fixed_mean"] > baseline["F1_fixed_mean"],
        "paired_v3_wins_at_least_3": wins >= 3,
        "mIoU_std_no_worse_than_baseline": candidate["mIoU_fixed_std"]
        <= baseline["mIoU_fixed_std"],
        "F1_std_no_worse_than_baseline": candidate["F1_fixed_std"]
        <= baseline["F1_fixed_std"],
        "ods_ois_not_both_below_v3_by_0_002": not (
            candidate["ODS_F1_mean"] < v3["ODS_F1_mean"] - 0.002
            and candidate["OIS_F1_mean"] < v3["OIS_F1_mean"] - 0.002
        ),
    }
    promoted = all(checks.values())
    return {
        "mode": mode,
        "status": "candidate_promoted" if promoted else "candidate_rejected",
        "promoted": promoted,
        "checks": checks,
        "paired_v3_wins": wins,
        "candidate": candidate,
        "baseline": baseline,
        "v3": v3,
    }


def choose_candidate(decisions):
    promoted = [decision for decision in decisions if decision["promoted"]]
    if not promoted:
        return None
    return max(
        promoted,
        key=lambda decision: (
            decision["candidate"]["mIoU_fixed_mean"],
            decision["candidate"]["F1_fixed_mean"],
            -decision["candidate"]["mIoU_fixed_std"],
            -decision["candidate"]["F1_fixed_std"],
        ),
    )["mode"]


def summarize_stage1(run):
    references = reference_records()
    decisions = []
    all_rows = []
    delta_rows = []
    for mode in CANDIDATES:
        rows = collect_stage1(run, mode)
        all_rows.extend(rows)
        decision = candidate_decision(mode, rows, references)
        decisions.append(decision)
        for row in rows:
            for reference_name in ("baseline", "msa_ocv_gbc"):
                reference = references[("CrackMap", reference_name, int(row["seed"]))]
                delta_rows.append(
                    {
                        "mode": mode,
                        "seed": row["seed"],
                        "reference": reference_name,
                        **{
                            metric: float(row[metric]) - float(reference[metric])
                            for metric in METRICS
                        },
                    }
                )
    write_tsv(run / "stage1_candidate_4seed.tsv", all_rows)
    write_tsv(run / "stage1_paired_deltas.tsv", delta_rows)
    write_tsv(
        run / "stage1_summary.tsv",
        [decision[name] for decision in decisions for name in ("candidate",)],
    )
    selected = choose_candidate(decisions)
    outcome = {
        "status": "stage1_promoted" if selected else "stage1_rejected",
        "promoted": selected is not None,
        "selected_mode": selected,
        "decisions": decisions,
    }
    atomic_json(run / "stage1_decision.json", outcome)
    if selected:
        (run / "selected_mode.txt").write_text(selected + "\n")
    log(
        f"{outcome['status']} selected={selected or 'none'} "
        + " ".join(
            f"{decision['mode']}_wins={decision['paired_v3_wins']}/4"
            for decision in decisions
        )
    )
    return selected is not None


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
    rows = collect_stage2(run)
    write_tsv(run / "cross_dataset_candidate.tsv", rows)
    decision = final_decision(rows, references)
    atomic_json(run / "final_decision.json", decision)
    comparison_rows = []
    for item in decision["comparisons"]:
        for model, row in (
            (item["candidate"]["mode"], item["candidate"]),
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
        "# Simplified TripleStack-v3 candidate validation",
        "",
        f"Terminal status: `{terminal_status}`",
        "",
        "Every new checkpoint was selected on validation; test was evaluated once after selection.",
        "",
        "## CrackMap four-seed candidates",
        "",
        "| Mode | mIoU mean +/- SD | F1 mean +/- SD | Paired wins vs V3 | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for decision in stage1["decisions"]:
        row = decision["candidate"]
        lines.append(
            f"| {decision['mode']} | {row['mIoU_fixed_mean']:.6f} +/- "
            f"{row['mIoU_fixed_std']:.6f} | {row['F1_fixed_mean']:.6f} +/- "
            f"{row['F1_fixed_std']:.6f} | {decision['paired_v3_wins']}/4 | "
            f"{decision['status']} |"
        )
    lines += [
        "",
        f"Selected mode: `{stage1['selected_mode'] or 'none'}`.",
        "",
        "Four seeds are robustness evidence, not a significance claim. These candidates were chosen "
        "after the seed42 factorial study, so the results are not an untouched external test.",
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
    mode_path = run / "selected_mode.txt"
    modes = [mode_path.read_text().strip()] if mode_path.is_file() else list(CANDIDATES)
    for mode in modes:
        wait_for_gpu(gpu)
        output = run / f"profile_{mode}.log"
        command = [
            sys.executable,
            "-u",
            str(ROOT / "tools/profile_model.py"),
            "--dataset_path",
            "dataset/CrackMap",
            "--nbins",
            "180",
            "--use_triple_stack_v3",
            "--triple_stack_v3_mode",
            mode,
        ]
        with output.open("w") as handle:
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
        log(f"PROFILE DONE mode={mode} gpu={gpu} output={output}")


def worker(run, stage, gpu):
    jobs = STAGE1_JOBS if stage == "stage1" else stage2_jobs(run)
    failures = 0
    for job in jobs:
        if job["gpu"] != gpu:
            continue
        try:
            train_candidate(
                run,
                stage,
                job["mode"],
                job["dataset"],
                job["seed"],
                gpu,
            )
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
        default=ROOT / "work_dirs/v3_simplified_candidates_staged_gpu012",
    )
    parser.add_argument("--stage", choices=["stage1", "stage2"])
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    run = args.run_dir.resolve()
    torch.set_num_threads(2)

    if args.action == "dry-run":
        for job in STAGE1_JOBS:
            print(
                f"DRYRUN stage1 mode={job['mode']} dataset={job['dataset']} "
                f"seed={job['seed']} gpu={job['gpu']} epochs=50"
            )
        for dataset, gpu in STAGE2_DATASETS:
            print(
                f"DRYRUN conditional stage2 mode=<winner> dataset={dataset} "
                f"seed=42 gpu={gpu} epochs=50"
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
