"""Evaluate topology metrics after the simplified V3 candidate queue finishes."""

import argparse
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_topology import evaluate_result_dir
from tools.paper_audit import atomic_json, completed, read_tsv, write_tsv


REFERENCE_RUN = ROOT / "work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12"
TOPOLOGY_METRICS = (
    "clDice",
    "Boundary_F1",
    "Boundary_Precision",
    "Boundary_Recall",
    "Component_MAE",
    "Endpoint_MAE",
)
HIGHER_IS_BETTER = {"clDice", "Boundary_F1", "Boundary_Precision", "Boundary_Recall"}
CRACKMAP_SEEDS = (42, 3407, 2026, 1234)
CANDIDATES = ("id_ocv_gbc", "msa_id_gbc")


def reference_rows():
    rows = read_tsv(REFERENCE_RUN / "reselected.metrics.tsv")
    keep = []
    for row in rows:
        dataset = row["dataset"]
        model = row["model"]
        seed = int(row["seed"])
        if model not in ("baseline", "msa_ocv_gbc"):
            continue
        if dataset == "CrackMap" and seed in CRACKMAP_SEEDS:
            keep.append(row)
        elif dataset in ("DeepCrack", "CamCrack789") and seed == 42:
            keep.append(row)
    expected = 12
    if len(keep) != expected:
        raise ValueError(f"Expected {expected} reference rows, found {len(keep)}")
    return keep


def candidate_rows(candidate_run):
    stage1_path = candidate_run / "stage1_candidate_4seed.tsv"
    if not stage1_path.is_file():
        raise ValueError(f"Missing candidate results: {stage1_path}")
    rows = read_tsv(stage1_path)
    keys = {(row["model"], int(row["seed"])) for row in rows}
    expected = {(mode, seed) for mode in CANDIDATES for seed in CRACKMAP_SEEDS}
    if keys != expected:
        raise ValueError(f"Incomplete candidate rows: missing={expected - keys}, extra={keys - expected}")
    cross_path = candidate_run / "cross_dataset_candidate.tsv"
    if cross_path.is_file():
        rows.extend(read_tsv(cross_path))
    return rows


def validate_result_record(row):
    result_dir = Path(row["result_dir"])
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    if not (result_dir / "summary.csv").is_file():
        raise ValueError(f"Missing standard evaluation: {result_dir}")
    lab_count = len(list(result_dir.glob("*_lab.png")))
    pre_count = len(list(result_dir.glob("*_pre.png")))
    if not lab_count or lab_count != pre_count:
        raise ValueError(f"Incomplete probability maps: {result_dir}")
    return result_dir


def collect_records(candidate_run):
    combined = []
    for row in reference_rows() + candidate_rows(candidate_run):
        combined.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "result_dir": str(validate_result_record(row)),
            }
        )
    keys = [(row["dataset"], row["model"], row["seed"]) for row in combined]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate topology-evaluation records")
    return combined


def evaluate_records(records):
    rows = []
    for index, record in enumerate(records, 1):
        metrics = evaluate_result_dir(
            record["result_dir"],
            method=f"{record['dataset']}_{record['model']}_seed{record['seed']}",
            threshold=0.5,
            boundary_tolerance_px=2,
        )
        rows.append({**record, **metrics})
        print(
            f"TOPOLOGY DONE {index}/{len(records)} dataset={record['dataset']} "
            f"model={record['model']} seed={record['seed']} "
            f"clDice={metrics['clDice']:.6f} BF1={metrics['Boundary_F1']:.6f}",
            flush=True,
        )
    return rows


def summarize_crackmap(rows):
    groups = []
    models = ("baseline", "msa_ocv_gbc") + CANDIDATES
    for model in models:
        group = [row for row in rows if row["dataset"] == "CrackMap" and row["model"] == model]
        if len(group) != 4:
            raise ValueError(f"Expected four CrackMap rows for {model}, found {len(group)}")
        summary = {"model": model, "n": len(group)}
        for metric in TOPOLOGY_METRICS:
            values = [float(row[metric]) for row in group]
            summary[metric + "_mean"] = statistics.mean(values)
            summary[metric + "_std"] = statistics.stdev(values)
        groups.append(summary)
    return groups


def paired_deltas(rows):
    lookup = {
        (row["dataset"], row["model"], int(row["seed"])): row
        for row in rows
    }
    output = []
    for model in ("msa_ocv_gbc",) + CANDIDATES:
        for seed in CRACKMAP_SEEDS:
            row = lookup[("CrackMap", model, seed)]
            baseline = lookup[("CrackMap", "baseline", seed)]
            output.append(
                {
                    "model": model,
                    "seed": seed,
                    **{
                        metric: (
                            float(row[metric]) - float(baseline[metric])
                            if metric in HIGHER_IS_BETTER
                            else float(baseline[metric]) - float(row[metric])
                        )
                        for metric in TOPOLOGY_METRICS
                    },
                }
            )
    return output


def write_report(run, summaries, rows, candidate_run):
    selected_path = candidate_run / "selected_mode.txt"
    selected = selected_path.read_text().strip() if selected_path.is_file() else "none"
    lines = [
        "# Topology and boundary metric audit",
        "",
        "All metrics use the validation-selected test probability maps at threshold 0.5. "
        "Boundary F1 uses a two-pixel tolerance. Positive paired deltas always mean better than baseline.",
        "",
        f"Simplified-candidate decision: `{selected}`.",
        "",
        "## CrackMap four-seed summary",
        "",
        "| Model | clDice | Boundary F1 | Component MAE | Endpoint MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {summary['clDice_mean']:.6f} +/- "
            f"{summary['clDice_std']:.6f} | {summary['Boundary_F1_mean']:.6f} +/- "
            f"{summary['Boundary_F1_std']:.6f} | {summary['Component_MAE_mean']:.3f} | "
            f"{summary['Endpoint_MAE_mean']:.3f} |"
        )
    cross = [row for row in rows if row["dataset"] != "CrackMap"]
    if cross:
        lines += [
            "",
            "## Cross-dataset topology metrics",
            "",
            "| Dataset | Model | clDice | Boundary F1 | Component MAE | Endpoint MAE |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in sorted(cross, key=lambda item: (item["dataset"], item["model"])):
            lines.append(
                f"| {row['dataset']} | {row['model']} | {row['clDice']:.6f} | "
                f"{row['Boundary_F1']:.6f} | {row['Component_MAE']:.3f} | "
                f"{row['Endpoint_MAE']:.3f} |"
            )
    (run / "report.md").write_text("\n".join(lines) + "\n")


def collect(run, candidate_run):
    run.mkdir(parents=True, exist_ok=True)
    candidate_run = candidate_run.resolve()
    stage1_decision = candidate_run / "stage1_decision.json"
    if not stage1_decision.is_file():
        raise ValueError("Simplified-candidate queue has not produced a decision")
    decision = json.loads(stage1_decision.read_text())
    if decision.get("status") not in ("stage1_promoted", "stage1_rejected"):
        raise ValueError("Invalid simplified-candidate decision")
    records = collect_records(candidate_run)
    manifest = {
        "candidate_run": str(candidate_run),
        "candidate_status": decision["status"],
        "selected_mode": decision.get("selected_mode"),
        "threshold": 0.5,
        "boundary_tolerance_px": 2,
        "records": records,
    }
    atomic_json(run / "manifest.json", manifest)
    rows = evaluate_records(records)
    summaries = summarize_crackmap(rows)
    deltas = paired_deltas(rows)
    write_tsv(run / "per_run.tsv", rows)
    write_tsv(run / "crackmap_4seed_summary.tsv", summaries)
    write_tsv(run / "paired_deltas.tsv", deltas)
    write_tsv(run / "cross_dataset.tsv", [row for row in rows if row["dataset"] != "CrackMap"])
    write_report(run, summaries, rows, candidate_run)
    atomic_json(run / "completed.json", {"status": "complete", "records": len(rows)})
    print(f"TOPOLOGY AUDIT COMPLETE records={len(rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "dry-run"])
    parser.add_argument("--run-dir", type=Path, default=ROOT / "work_dirs/topology_metrics")
    parser.add_argument("--candidate-run", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "dry-run":
        print("DRYRUN topology fixed_threshold=0.5 boundary_tolerance_px=2")
        print("DRYRUN CrackMap models=baseline,msa_ocv_gbc,id_ocv_gbc,msa_id_gbc seeds=4")
        print("DRYRUN cross_dataset references=DeepCrack,CamCrack789 candidate=conditional")
    else:
        collect(args.run_dir.resolve(), args.candidate_run.resolve())


if __name__ == "__main__":
    main()
