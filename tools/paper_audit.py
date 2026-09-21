"""Evidence inventory, validation reselection, and the fixed V7 removal queue."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from datasets import create_dataset
from datasets.image_folder import make_dataset
from main import get_args_parser
from models import build_MixerCSeg
from tools.infer_probability import apply_checkpoint_args, run_inference
from util.standard_validation import evaluate_validation, make_validation_loader, selection_key


METRICS = ["mIoU_fixed", "F1_fixed", "P_fixed", "R_fixed", "ODS_F1", "OIS_F1"]
REMOVALS = ["id_ctv_dgb", "wma_id_dgb", "wma_ctv_id", "id_id_id"]
SOURCES = [
    ("v4", "logs/20260806_220017_CrackMap_triple_stack_v4_search12_validate8_gpu0.metrics.tsv"),
    ("v7", "logs/20260906_145204_MixerCSeg_pending_gpu0_resume.v7.metrics.tsv"),
    ("core", "logs/20260908_221618_TripleStackV7_wma_ctv_dgb_core5_gpu2.metrics.tsv"),
    ("DeepCrack", "logs/20260906_145204_MixerCSeg_pending_gpu0_resume.deepcrack.metrics.tsv"),
    ("CamCrack789", "logs/20260906_145204_MixerCSeg_pending_gpu0_resume.camcrack789.metrics.tsv"),
]


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_checkpoint(path):
    return torch.load(str(path), map_location="cpu")


def checkpoint_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def audit_dataset(root):
    """Audit native decoded images, not resized crops or mask hashes alone."""
    root = Path(root)
    seen = {}
    rows = []
    errors = []
    for split in ("train", "val", "test"):
        images = [Path(p) for p in make_dataset(str(root / f"{split}_img"))]
        if not images:
            errors.append(f"{split}: empty image split")
        paired = set()
        stems = set()
        for path in images:
            label_name = ("target-" + path.stem.split("-")[-1] + ".png"
                          if root.name == "CamCrack789" else path.name.split(".")[0] + ".png")
            label = root / f"{split}_lab" / label_name
            if path.stem in stems or label in paired:
                errors.append(f"{split}: ambiguous sample or mask {path}")
            stems.add(path.stem)
            paired.add(label)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(label), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                errors.append(f"Unreadable pair: {path}, {label}")
                continue
            if image.shape[:2] != mask.shape:
                errors.append(f"Native shape mismatch: {path}")
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                errors.append(f"Cross-split duplicate: {seen[digest][1]} and {path}")
            seen.setdefault(digest, (split, str(path)))
            rows.append({"split": split, "image": str(path), "label": str(label),
                         "image_hash": digest,
                         "label_hash": hashlib.sha256(mask.tobytes()).hexdigest()})
        labels = {Path(p) for p in make_dataset(str(root / f"{split}_lab"))}
        if labels != paired:
            errors.append(f"{split}: unpaired labels={sorted(str(p) for p in labels - paired)}")
    return {"dataset": root.name, "samples": rows, "errors": errors}


def core_jobs():
    jobs = []
    for kind, source in SOURCES:
        for row in read_tsv(ROOT / source):
            dataset = row.get("dataset", "CrackMap")
            if kind == "v4":
                if row["model"] not in ("baseline", "v3"):
                    continue
                model = "baseline" if row["model"] == "baseline" else "msa_ocv_gbc"
            elif kind == "v7":
                if row["mode"] != "wma_ctv_dgb":
                    continue
                model = row["mode"]
            elif kind == "core":
                model = row["model"]
            else:
                dataset, model = kind, row["job"]
            seed = int(row["seed"])
            jobs.append({"job_id": f"{dataset}_{model}_seed{seed}", "dataset": dataset,
                         "model": model, "seed": seed, "checkpoint": row["checkpoint"],
                         "source": source, "old_metrics": {m: float(row[m]) for m in METRICS}})
    ids = [job["job_id"] for job in jobs]
    if len(jobs) != 18 or len(set(ids)) != 18:
        raise ValueError("Expected 18 unique, explicitly sourced training trajectories")
    return jobs


def validate_checkpoint(checkpoint, job, epoch=None):
    args = checkpoint["args"]
    if epoch is not None and checkpoint.get("epoch") != epoch:
        raise ValueError(f"Wrong checkpoint epoch for {job['job_id']}: expected {epoch}")
    if int(args.epochs) != 50 or int(args.seed) != job["seed"] or int(args.nbins) != 180:
        raise ValueError(f"Training protocol mismatch: {job['job_id']}")
    if Path(args.dataset_path).name != job["dataset"]:
        raise ValueError(f"Dataset mismatch: {job['job_id']}")
    if not math.isclose(args.BCELoss_ratio, 0.87) or not math.isclose(args.DiceLoss_ratio, 0.13):
        raise ValueError(f"Loss weights mismatch: {job['job_id']}")
    active = {name for name, value in vars(args).items()
              if name.startswith("use_") and value is True}
    expected = ({"use_triple_stack_v3"} if job["model"] == "msa_ocv_gbc" else
                set() if job["model"] == "baseline" else {"use_triple_stack_v7"})
    if active != expected:
        raise ValueError(f"Unexpected active flags for {job['job_id']}: {active}")
    if expected:
        version = "v3" if job["model"] == "msa_ocv_gbc" else "v7"
        if getattr(args, f"triple_stack_{version}_mode") != job["model"]:
            raise ValueError("Wrong model mode in checkpoint")


def model_args(checkpoint, dataset, device):
    args = get_args_parser().parse_args([])
    args.use_checkpoint_args = True
    apply_checkpoint_args(args, checkpoint)
    args.dataset_path = str(ROOT / "dataset" / dataset)
    args.device = torch.device(device)
    args.num_threads = 0
    args.batch_size_test = 1
    args.serial_batches = True
    return args


def family_name(name):
    for family in ["triple_stack_v10", "triple_stack_v9", "triple_stack_v8", "triple_stack_v7",
                   "triple_stack_v6", "triple_stack_v5", "triple_stack_v4", "triple_stack_v3",
                   "rsgdi_v2", "drsgi", "paper", "third", "second", "exp", "edrm"]:
        if family in name:
            return family
    return "baseline" if "ccem_baseline" in name else "ccem" if "ccem_" in name else "unknown"


def inventory(run):
    records = []
    cached = {}
    for source in sorted((ROOT / "logs").glob("*.metrics.tsv")):
        with source.open(newline="") as handle:
            raw = list(csv.reader(handle, delimiter="\t"))
        if not raw:
            continue
        if "checkpoint" not in raw[0]:
            for values in raw:
                records.append({"source": str(source.relative_to(ROOT)), "family": family_name(source.name),
                                "protocol": "unresolved_header_or_checkpoint", "status": "unverified",
                                "raw_record": json.dumps(values)})
            continue
        for row in (dict(zip(raw[0], values)) for values in raw[1:]):
            path = ROOT / row["checkpoint"]
            metadata = {}
            status = "missing_checkpoint"
            if path.is_file():
                if str(path) not in cached:
                    try:
                        checkpoint = load_checkpoint(path)
                        a = vars(checkpoint["args"])
                        cached[str(path)] = {key: a.get(key, "") for key in
                                            ("seed", "epochs", "BCELoss_ratio", "DiceLoss_ratio")}
                        cached[str(path)]["dataset"] = Path(a.get("dataset_path", "")).name
                        cached[str(path)]["family"] = family_name(path.parent.name)
                        cached[str(path)]["architecture"] = path.parent.name.split("_seed", 1)[-1]
                        cached[str(path)]["flags"] = json.dumps({k: v for k, v in a.items()
                                                               if k.startswith("use_")})
                        del checkpoint
                    except Exception as exc:
                        cached[str(path)] = {"metadata_error": str(exc)}
                metadata = cached[str(path)]
                total = metadata.get("epochs")
                status = ("epoch_artifacts_complete" if isinstance(total, int) and
                          (path.parent / f"checkpoint{total - 1}.pth").is_file()
                          else "incomplete_or_unverified")
            common = dict(row, **{k: v for k, v in metadata.items() if k not in row})
            common.update(source=str(source.relative_to(ROOT)),
                          family=metadata.get("family", family_name(path.parent.name)),
                          status=status, selection_split="test")
            for protocol, mkey, fkey in (("standard_png_fixed_0.5", "mIoU_fixed", "F1_fixed"),
                                        ("legacy_normalized_logits", "legacy_mIoU", "legacy_F1")):
                if row.get(mkey) and row.get(fkey):
                    records.append(dict(common, protocol=protocol, mIoU=row[mkey], F1=row[fkey]))
    sources = set((ROOT / "logs").glob("*.outer.log")) | set((ROOT / "logs").glob("*ccem*.log"))
    pattern = re.compile(r"Best metrics \| experiment -> (\S+).*?mIoU -> ([\d.eE+-]+).*?F1 -> ([\d.eE+-]+)")
    for source in sorted(sources):
        weights = {}
        pending = None
        with source.open(errors="replace") as handle:
            for line in handle:
                for key in ("BCELoss_ratio", "DiceLoss_ratio", "epochs"):
                    match = re.search(rf"args: {key} -> ([\d.]+)", line)
                    if match:
                        weights[key] = match.group(1)
                match = pattern.search(line)
                if match:
                    name, miou, f1 = match.groups()
                    seed = re.search(r"_seed(\d+)", name)
                    pending = dict(source=str(source.relative_to(ROOT)), family=family_name(name),
                                   model=name, dataset=name.split("_seed")[0],
                                   seed=seed.group(1) if seed else "", **weights,
                                   protocol="legacy_normalized_logits", selection_split="test",
                                   mIoU=miou, F1=f1, status="summary_present_completion_unverified")
                    records.append(pending)
                if pending is not None and "Process time " in line:
                    pending["status"] = "training_completed_log"
                    pending = None
    write_tsv(run / "inventory.tsv", records)
    (run / "inventory_notes.md").write_text(
        "# Historical experiment inventory\n\n"
        "Rows are evidence records, not counts of independent runs. Repeated checkpoint paths "
        "in merged/resumed TSVs must not be counted twice. Missing protocol metadata is unknown, "
        "not inferred. epoch_artifacts_complete verifies the final epoch artifact, not final "
        "evaluation success. Legacy normalized-logit scores and standard sigmoid scores are "
        "different protocols. Historical test-selected results remain exploratory.\n")
    summarize_inventory(run, records)
    log(f"INVENTORY records={len(records)}")


def summarize_inventory(run, records):
    groups = {}
    for row in records:
        if not row.get("mIoU") or not row.get("F1"):
            continue
        key = tuple(str(row.get(field, "unknown")) for field in
                    ("dataset", "family", "seed", "epochs", "BCELoss_ratio", "DiceLoss_ratio",
                     "protocol", "selection_split", "flags"))
        try:
            score = (float(row["mIoU"]), float(row["F1"]))
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in score):
            continue
        if key not in groups or score > groups[key][0]:
            groups[key] = (score, row)
    best = [entry[1] for entry in groups.values()]
    write_tsv(run / "historical_best_by_protocol.tsv", best)
    lines = ["# Historical experiment families", "",
             "Exploratory best observed scores, grouped by protocol, seed and available loss metadata. "
             "Unknown metadata is NOT a fair-comparison protocol. Incomplete/unverified runs are labelled. "
             "Rows are not independent runs and must not be used as sample counts.", "",
             "| Dataset | Family | Architecture | Seed | Epochs | BCE/Dice | Protocol | mIoU | F1 | Status |",
             "|---|---|---|---:|---:|---|---|---:|---:|---|"]
    for row in best:
        lines.append("| " + str(row.get("dataset", "unknown")) + " | "
                     + str(row.get("family", "unknown")) + " | "
                     + str(row.get("architecture", row.get("model", "unknown"))) + " | "
                     + " | ".join(str(row.get(key, "unknown")) for key in ("seed", "epochs")) + " | "
                     + f"{row.get('BCELoss_ratio', 'unknown')}/{row.get('DiceLoss_ratio', 'unknown')}"
                     + f" | {row['protocol']} | {float(row['mIoU']):.6f} | {float(row['F1']):.6f}"
                     + f" | {row['status']} |")
    (run / "historical_summary.md").write_text("\n".join(lines) + "\n")


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    audits = [audit_dataset(ROOT / "dataset" / name)
              for name in ("CrackMap", "DeepCrack", "CamCrack789")]
    atomic_json(run / "dataset_audit.json", audits)
    errors = [error for audit in audits for error in audit["errors"]]
    if errors:
        raise ValueError("Dataset audit failed:\n" + "\n".join(errors))
    jobs = core_jobs()
    for index, job in enumerate(jobs):
        job["gpu"] = 1 + index % 2
        parent = (ROOT / job["checkpoint"]).parent
        missing = [i for i in range(50) if not (parent / f"checkpoint{i}.pth").is_file()]
        if missing:
            raise ValueError(f"Missing epochs for {job['job_id']}: {missing}")
        checkpoint = load_checkpoint(parent / "checkpoint49.pth")
        validate_checkpoint(checkpoint, job, 49)
        args = model_args(checkpoint, job["dataset"], "cpu")
        model, _ = build_MixerCSeg(args)
        model.load_state_dict(checkpoint["model"], strict=True)
        job["trajectory"] = [checkpoint_signature(parent / f"checkpoint{i}.pth") for i in range(50)]
        del model, checkpoint
        log(f"PREFLIGHT OK {job['job_id']}")
    manifest = run / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != jobs:
        raise ValueError("Existing manifest differs; use a fresh run directory")
    atomic_json(manifest, jobs)
    inventory(run)
    log("PREPARE DONE trajectories=18")


def wait_for_gpu(gpu):
    if gpu not in (1, 2) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("Worker must use its assigned physical GPU 1 or 2 only")
    while True:
        result = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True)
        try:
            memory, utilization = [int(value.strip()) for value in result.stdout.strip().split(",")]
        except ValueError:
            memory, utilization = 999999, 100
        if result.returncode == 0 and memory < 1000 and utilization < 10:
            log(f"selected GPU {gpu} memory={memory}MiB utilization={utilization}%")
            return
        log(f"waiting GPU {gpu} memory={memory} utilization={utilization}")
        time.sleep(int(os.environ.get("POLL_SECONDS", "60")))


def evaluate_test(model, args, destination):
    destination.mkdir(parents=True, exist_ok=False)
    test_args = copy.copy(args)
    test_args.phase = "test"
    test_args.batch_size = test_args.batch_size_test
    test_args.serial_batches = True
    loader = create_dataset(test_args)
    run_inference(test_args, model, loader, test_args.device, destination)
    if any(len(list(destination.glob(pattern))) != len(loader.dataset)
           for pattern in ("*_pre.png", "*_lab.png")):
        raise ValueError("Incomplete probability-map output")
    with (destination / "evaluation.log").open("w") as handle:
        subprocess.run([sys.executable, str(ROOT / "tools/eval_standard.py"),
                        "--result_dir", str(destination)], check=True, stdout=handle,
                       stderr=subprocess.STDOUT)
    with (destination / "summary.csv").open(newline="") as handle:
        result = next(csv.DictReader(handle))
    metrics = {m: float(result[m]) for m in METRICS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite test metric")
    return metrics


def fresh_attempt(job_root):
    job_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        path = job_root / f"attempt{index:04d}"
        if not path.exists():
            path.mkdir()
            return path
    raise RuntimeError("Too many attempts")


def completed(job_root):
    path = job_root / "completed.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
        summary = Path(record["result_dir"]) / "summary.csv"
        if not Path(record["checkpoint"]).is_file() or not summary.is_file():
            return False
        with summary.open(newline="") as handle:
            saved = next(csv.DictReader(handle))
        return (record.get("selection_split") == "val"
                and all(math.isfinite(float(record[m]))
                        and math.isclose(float(record[m]), float(saved[m]), abs_tol=1e-12)
                        for m in METRICS))
    except (KeyError, ValueError, OSError, StopIteration):
        return False


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
    args = model_args(checkpoint, job["dataset"], "cuda:0")
    model, _ = build_MixerCSeg(args)
    model.to(args.device)
    loader = make_validation_loader(args)
    rows = []
    best = None
    start = time.monotonic()
    for epoch in range(50):
        path = parent / f"checkpoint{epoch}.pth"
        if checkpoint_signature(path) != job["trajectory"][epoch]:
            raise ValueError(f"Checkpoint changed since manifest: {path}")
        checkpoint = load_checkpoint(path)
        validate_checkpoint(checkpoint, job, epoch)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics = evaluate_validation(model, loader, args.device)
        record = dict(metrics, epoch=epoch, checkpoint=str(path))
        rows.append(record)
        if best is None or selection_key(metrics, epoch) > selection_key(best, best["epoch"]):
            best = record
        write_tsv(attempt / "validation_epochs.tsv", rows)
        if epoch % 10 == 9:
            log(f"RESELECT PROGRESS {job['job_id']} epochs={epoch + 1}/50 elapsed={time.monotonic()-start:.1f}s")
    atomic_json(attempt / "selection.json", dict(best, split="val", fixed_threshold=0.5))
    checkpoint = load_checkpoint(best["checkpoint"])
    model.load_state_dict(checkpoint["model"], strict=True)
    result_dir = attempt / "test_probability"
    metrics = evaluate_test(model, args, result_dir)
    record = {k: job[k] for k in ("job_id", "dataset", "model", "seed", "gpu")}
    record.update(metrics, selection_split="val", epoch=best["epoch"],
                  checkpoint=best["checkpoint"], result_dir=str(result_dir),
                  elapsed_seconds=time.monotonic() - start)
    for name, value in job["old_metrics"].items():
        record["old_" + name] = value
        record["reselection_delta_" + name] = metrics[name] - value
    atomic_json(job_root / "completed.json", record)
    del model, checkpoint
    torch.cuda.empty_cache()
    log(f"RESELECT DONE {job['job_id']} epoch={best['epoch']} mIoU={metrics['mIoU_fixed']:.6f}")


def train_removal(run, mode, gpu):
    job_id = f"CrackMap_{mode}_seed42"
    job_root = run / "removals" / job_id
    if completed(job_root):
        log(f"TRAIN SKIP {job_id}")
        return
    wait_for_gpu(gpu)
    # Preserve interrupted directories. Only a successful training marker permits evaluation-only recovery.
    ready = sorted(job_root.glob("attempt*/weights/*/training_complete.json"))
    if ready:
        marker = ready[-1]
        attempt = marker.parents[2]
        completion = json.loads(marker.read_text())
        checkpoint_path = Path(completion["checkpoint"])
    else:
        attempt = fresh_attempt(job_root)
        command = [sys.executable, "-u", str(ROOT / "main.py"),
                   "--dataset_path", "dataset/CrackMap", "--nbins", "180", "--seed", "42",
                   "--epochs", "50", "--BCELoss_ratio", "0.87", "--DiceLoss_ratio", "0.13",
                   "--output_dir", str(attempt / "weights"), "--use_triple_stack_v7",
                   "--triple_stack_v7_mode", mode, "--validation_selection"]
        log(f"TRAIN START {job_id} gpu={gpu}")
        with (attempt / "train.log").open("w") as handle:
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
        markers = list((attempt / "weights").glob("*/training_complete.json"))
        if len(markers) != 1:
            raise ValueError("Expected exactly one successful training marker")
        marker = markers[0]
        completion = json.loads(marker.read_text())
        checkpoint_path = Path(completion["checkpoint"])
    if completion["completed_epochs"] != 50 or not (marker.parent / "checkpoint49.pth").is_file():
        raise ValueError("Training is not complete")
    job = {"job_id": job_id, "dataset": "CrackMap", "model": mode, "seed": 42, "gpu": gpu}
    checkpoint = load_checkpoint(checkpoint_path)
    validate_checkpoint(checkpoint, job)
    if checkpoint.get("selection", {}).get("split") != "val":
        raise ValueError("Removal checkpoint was not validation-selected")
    args = model_args(checkpoint, "CrackMap", "cuda:0")
    model, _ = build_MixerCSeg(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    evaluation = fresh_attempt(attempt / "evaluations") / "test_probability"
    metrics = evaluate_test(model, args, evaluation)
    record = dict(job, **metrics, checkpoint=str(checkpoint_path), result_dir=str(evaluation),
                  selection_split="val", epoch=checkpoint["epoch"])
    atomic_json(job_root / "completed.json", record)
    del model, checkpoint
    torch.cuda.empty_cache()
    log(f"TRAIN DONE {job_id} mIoU={metrics['mIoU_fixed']:.6f}")


def summarize(run, require_removals=False):
    jobs = json.loads((run / "manifest.json").read_text())
    records = []
    for job in jobs:
        root = run / "reselected" / job["job_id"]
        if not completed(root):
            raise ValueError(f"Missing successful re-evaluation: {job['job_id']}")
        records.append(json.loads((root / "completed.json").read_text()))
    write_tsv(run / "reselected.metrics.tsv", records)
    rows = []
    crack = [row for row in records if row["dataset"] == "CrackMap"]
    for model in ("baseline", "msa_ocv_gbc", "wma_ctv_dgb"):
        group = [r for r in crack if r["model"] == model]
        row = {"model": model, "n": len(group)}
        for metric in METRICS:
            values = [r[metric] for r in group]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values)
        rows.append(row)
    write_tsv(run / "crackmap_4seed.tsv", rows)
    deltas = []
    for row in records:
        if row["model"] != "wma_ctv_dgb":
            continue
        for reference in ("baseline", "msa_ocv_gbc"):
            base = next(r for r in records if r["dataset"] == row["dataset"]
                        and r["seed"] == row["seed"] and r["model"] == reference)
            deltas.append(dict(dataset=row["dataset"], seed=row["seed"], reference=reference,
                               **{m: row[m] - base[m] for m in METRICS}))
    write_tsv(run / "paired_deltas.tsv", deltas)
    write_tsv(run / "cross_dataset.tsv", [r for r in records if r["dataset"] != "CrackMap"])
    ablation = [r.copy() for r in crack if r["seed"] == 42 and r["model"] != "msa_ocv_gbc"]
    for mode in REMOVALS:
        root = run / "removals" / f"CrackMap_{mode}_seed42"
        if completed(root):
            ablation.append(json.loads((root / "completed.json").read_text()))
        elif require_removals:
            raise ValueError(f"Missing successful removal experiment: {mode}")
    base = next(r for r in ablation if r["model"] == "baseline")
    full = next(r for r in ablation if r["model"] == "wma_ctv_dgb")
    for row in ablation:
        for m in METRICS:
            row["delta_baseline_" + m] = row[m] - base[m]
            row["delta_full_" + m] = row[m] - full[m]
    write_tsv(run / "removal_ablation.tsv", ablation)
    lines = ["# V7 validation-selected evidence", "",
             "Historical test-selected results are exploratory. Validation reselection corrects "
             "epoch selection, not prior architecture-search bias. No claim of independent untouched testing.", "",
             "## CrackMap: four seeds", "", "| Model | mIoU mean +/- sample SD | F1 mean +/- sample SD |",
             "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['mIoU_fixed_mean']:.6f} +/- {row['mIoU_fixed_std']:.6f} | "
                     f"{row['F1_fixed_mean']:.6f} +/- {row['F1_fixed_std']:.6f} |")
    lines += ["", "## Removal ablation: seed42 only", "",
              "| Model | mIoU | F1 | Delta mIoU vs full | Delta F1 vs full |", "|---|---:|---:|---:|---:|"]
    for row in ablation:
        lines.append(f"| {row['model']} | {row['mIoU_fixed']:.6f} | {row['F1_fixed']:.6f} | "
                     f"{row['delta_full_mIoU_fixed']:+.6f} | {row['delta_full_F1_fixed']:+.6f} |")
    lines += ["", "id_id_id retains the wrapper and is NOT the original baseline. "
              "Removal results are single-seed preliminary evidence, not significance tests. "
              "Old V3-to-V7 replacement ablations are a different experiment and must be reported separately.",
              "", "Old and new metrics and their differences are in reselected.metrics.tsv. "
              "Historical protocols and unresolved evidence are in inventory.tsv. "
              "A decrease or a superior removal configuration must be reported, not filtered out."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    log(f"SUMMARY DONE core={len(records)} ablation_rows={len(ablation)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "reselect", "removals", "summarize", "dry-run"])
    parser.add_argument("--run-dir", type=Path, default=ROOT / "work_dirs/paper_v7_val_reselect_drop4")
    parser.add_argument("--gpu", type=int, choices=[1, 2])
    parser.add_argument("--require-removals", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = args.run_dir.resolve()
    if args.action == "dry-run":
        for index, job in enumerate(core_jobs()):
            print(f"DRYRUN RESELECT {job['job_id']} checkpoints=50 gpu={1+index%2}")
        for index, mode in enumerate(REMOVALS):
            print(f"DRYRUN TRAIN mode={mode} epochs=50 gpu={1+index%2} selection=val")
    elif args.action == "prepare":
        prepare(run)
    elif args.action == "summarize":
        summarize(run, args.require_removals)
    else:
        if args.gpu is None:
            parser.error("--gpu is required for workers")
        failures = 0
        jobs = (json.loads((run / "manifest.json").read_text()) if args.action == "reselect"
                else [{"mode": mode, "gpu": 1 + i % 2} for i, mode in enumerate(REMOVALS)])
        for job in jobs:
            if job["gpu"] != args.gpu:
                continue
            try:
                if args.action == "reselect":
                    reselect(run, job)
                else:
                    train_removal(run, job["mode"], args.gpu)
            except Exception:
                failures += 1
                log(f"FAIL action={args.action} job={job}")
                traceback.print_exc()
                # Exit the worker on failure: do not retain an exception's GPU tensors while waiting.
                break
        log(f"WORKER action={args.action} gpu={args.gpu} failures={failures}")
        sys.exit(bool(failures))


if __name__ == "__main__":
    main()
