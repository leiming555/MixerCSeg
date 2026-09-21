#!/usr/bin/env python3
"""Check that the manuscript agrees with the final, traceable experiment files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = ROOT / "paper_msa_ocv_gbc/msa_ocv_gbc.tex"
CORE_METRICS = (
    ROOT
    / "work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12"
    / "reselected.metrics.tsv"
)
DATASET_AUDIT = CORE_METRICS.parent / "dataset_audit.json"
FACTORIAL = (
    ROOT
    / "work_dirs/20260919_173408_v3_factorial_reselect9_gpu012"
    / "ablation.tsv"
)
PROFILE_LOG = ROOT / "logs/20260806_220017_CrackMap_triple_stack_v4_search12_validate8_gpu0.profile.log"
QUALITATIVE_MANIFEST = ROOT / "paper_msa_ocv_gbc/figures/qualitative_selection_v2.tsv"

DATASET_SPLITS = {
    "CrackMap": {"train": 84, "val": 12, "test": 24},
    "DeepCrack": {"train": 368, "val": 53, "test": 106},
    "CamCrack789": {"train": 553, "val": 79, "test": 157},
}
METRICS = ("mIoU_fixed", "F1_fixed", "ODS_F1", "OIS_F1")
T_CRITICAL_DF3_95 = 3.182446305284263


@dataclass
class AuditResult:
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def checked(self, message: str) -> None:
        self.checks.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _table_for_label(tex: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    label_pos = tex.find(marker)
    if label_pos < 0:
        raise ValueError(f"Missing manuscript table label: {label}")
    start = tex.rfind(r"\begin{table}", 0, label_pos)
    end = tex.find(r"\end{table}", label_pos)
    if start < 0 or end < 0:
        raise ValueError(f"Malformed manuscript table: {label}")
    return tex[start : end + len(r"\end{table}")]


def _table_row(table: str, prefix: str) -> str:
    for line in table.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    raise ValueError(f"Missing row starting with {prefix!r}")


def _decimal_values(line: str, places: int = 4) -> list[float]:
    return [
        float(value)
        for value in re.findall(rf"[+-]?\d+\.\d{{{places}}}", line)
    ]


def _four_decimal_values(line: str) -> list[float]:
    return _decimal_values(line, places=4)


def _expect_values(
    result: AuditResult,
    table: str,
    prefix: str,
    expected: Iterable[float],
    context: str,
    places: int = 4,
) -> None:
    try:
        actual = _decimal_values(_table_row(table, prefix), places=places)
    except ValueError as error:
        result.fail(f"{context}: {error}")
        return
    rounded = [round(float(value), places) for value in expected]
    if actual != rounded:
        result.fail(f"{context}: expected {rounded}, found {actual}")


def _check_evidence_files(result: AuditResult) -> bool:
    paths = (CORE_METRICS, DATASET_AUDIT, FACTORIAL, PROFILE_LOG, QUALITATIVE_MANIFEST)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            result.fail(f"Missing evidence file: {path}")
        return False
    result.checked("all manuscript evidence files exist")
    return True


def _check_dataset_splits(result: AuditResult, tex: str) -> None:
    try:
        table = _table_for_label(tex, "tab:datasets")
    except ValueError as error:
        result.fail(str(error))
        return

    for dataset, expected in DATASET_SPLITS.items():
        counts = {}
        for split, expected_count in expected.items():
            image_dir = ROOT / "dataset" / dataset / f"{split}_img"
            label_dir = ROOT / "dataset" / dataset / f"{split}_lab"
            image_count = sum(path.is_file() for path in image_dir.iterdir()) if image_dir.is_dir() else -1
            label_count = sum(path.is_file() for path in label_dir.iterdir()) if label_dir.is_dir() else -1
            counts[split] = image_count
            if image_count != expected_count or label_count != expected_count:
                result.fail(
                    f"{dataset}/{split}: expected {expected_count} image-label pairs, "
                    f"found images={image_count}, labels={label_count}"
                )
        row = _table_row(table, dataset)
        actual = [int(value) for value in re.findall(r"\b\d+\b", row)]
        expected_row = [counts["train"], counts["val"], counts["test"], sum(counts.values())]
        if actual != expected_row:
            result.fail(f"dataset table {dataset}: expected {expected_row}, found {actual}")

    audit = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    audit_errors = [error for item in audit for error in item.get("errors", [])]
    if audit_errors:
        result.fail(f"dataset integrity audit contains {len(audit_errors)} errors")
    else:
        result.checked("dataset counts and cross-split integrity audit agree with the manuscript")


def _crackmap_rows(rows: list[dict[str, str]], model: str) -> dict[int, dict[str, str]]:
    return {
        int(row["seed"]): row
        for row in rows
        if row["dataset"] == "CrackMap" and row["model"] == model
    }


def _check_seed_and_summary_tables(result: AuditResult, tex: str, rows: list[dict[str, str]]) -> None:
    baseline = _crackmap_rows(rows, "baseline")
    proposed = _crackmap_rows(rows, "msa_ocv_gbc")
    seeds = (42, 3407, 2026, 1234)
    if set(baseline) != set(seeds) or set(proposed) != set(seeds):
        result.fail("core evidence does not contain four paired CrackMap seeds")
        return

    try:
        seed_table = _table_for_label(tex, "tab:seed_results")
        summary_table = _table_for_label(tex, "tab:summary_results")
    except ValueError as error:
        result.fail(str(error))
        return

    lines = seed_table.splitlines()
    for seed in seeds:
        matches = [line.strip() for line in lines if line.strip().startswith(f"{seed} &")]
        if len(matches) != 2:
            result.fail(f"seed table: expected two rows for seed {seed}, found {len(matches)}")
            continue
        by_model = {"Baseline": baseline[seed], "Proposed": proposed[seed]}
        for model_name, source in by_model.items():
            line = next((line for line in matches if f"& {model_name} &" in line), None)
            if line is None:
                result.fail(f"seed table: missing {model_name} row for seed {seed}")
                continue
            actual = _four_decimal_values(line)
            expected = [round(float(source[name]), 4) for name in METRICS]
            if actual != expected:
                result.fail(
                    f"seed table {seed}/{model_name}: expected {expected}, found {actual}"
                )

    for metric, label in zip(METRICS, ("mIoU@0.5", "F1@0.5", "ODS-F1", "OIS-F1")):
        base_values = [float(baseline[seed][metric]) for seed in seeds]
        model_values = [float(proposed[seed][metric]) for seed in seeds]
        deltas = [model - base for model, base in zip(model_values, base_values)]
        delta_mean = statistics.mean(deltas)
        half_width = T_CRITICAL_DF3_95 * statistics.stdev(deltas) / math.sqrt(len(deltas))
        expected = (
            statistics.mean(base_values),
            statistics.stdev(base_values),
            statistics.mean(model_values),
            statistics.stdev(model_values),
            delta_mean,
            delta_mean - half_width,
            delta_mean + half_width,
        )
        _expect_values(result, summary_table, label, expected, f"summary table {label}")

    result.checked("paired seed rows, means, sample deviations, gains, and intervals were recomputed")


def _check_cross_dataset_table(result: AuditResult, tex: str, rows: list[dict[str, str]]) -> None:
    try:
        table = _table_for_label(tex, "tab:cross_dataset")
    except ValueError as error:
        result.fail(str(error))
        return

    crack = {
        model: _crackmap_rows(rows, model)
        for model in ("baseline", "msa_ocv_gbc")
    }
    model_names = {"baseline": "Baseline", "msa_ocv_gbc": "Proposed"}
    expected_rows: dict[tuple[str, str], list[float]] = {}
    for model, seed_rows in crack.items():
        expected_rows[("CrackMap", model_names[model])] = [
            statistics.mean(float(row[metric]) for row in seed_rows.values())
            for metric in METRICS
        ]
    for dataset in ("DeepCrack", "CamCrack789"):
        for model in ("baseline", "msa_ocv_gbc"):
            source = next(
                row for row in rows if row["dataset"] == dataset and row["model"] == model
            )
            expected_rows[(dataset, model_names[model])] = [float(source[metric]) for metric in METRICS]

    for (dataset, model_name), expected in expected_rows.items():
        prefix = f"{dataset} & {model_name}"
        _expect_values(result, table, prefix, expected, f"cross-dataset table {dataset}/{model_name}")
    result.checked("cross-dataset table agrees with validation-reselected evidence")


def _check_ablation_table(result: AuditResult, tex: str) -> None:
    try:
        table = _table_for_label(tex, "tab:ablation")
    except ValueError as error:
        result.fail(str(error))
        return
    rows = _read_tsv(FACTORIAL)
    by_mode = {row["mode"]: row for row in rows}
    labels = {
        "Original baseline": "baseline",
        "Identity control": "id_id_id",
        "MSA only": "msa_id_id",
        "OCV only": "id_ocv_id",
        "GBC only": "id_id_gbc",
        "MSA + OCV": "msa_ocv_id",
        "MSA + GBC": "msa_id_gbc",
        "OCV + GBC": "id_ocv_gbc",
        "Full model": "msa_ocv_gbc",
    }
    for label, mode in labels.items():
        source = by_mode.get(mode)
        if source is None:
            result.fail(f"ablation evidence is missing mode {mode}")
            continue
        _expect_values(
            result,
            table,
            label,
            [float(source[metric]) for metric in METRICS],
            f"ablation table {label}",
        )
    result.checked("all nine factorial-ablation rows agree with their source metrics")


def _profile_rows() -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    for block in PROFILE_LOG.read_text(encoding="utf-8").split("================ MixerCSeg Profile ================"):
        model_match = re.search(r"^Model\s*:\s*(\S+)", block, flags=re.MULTILINE)
        if not model_match:
            continue
        values = []
        for label in ("Params (M)", "FLOPs (GFLOPs)", "Model Size (MB)", "FPS", "Latency (ms/img)", "GPU Memory (MB)"):
            match = re.search(rf"^{re.escape(label)}\s*:\s*([0-9.]+)", block, flags=re.MULTILINE)
            if not match:
                raise ValueError(f"Profile block {model_match.group(1)} is missing {label}")
            values.append(float(match.group(1)))
        rows[model_match.group(1)] = values
    return rows


def _check_complexity_table(result: AuditResult, tex: str) -> None:
    try:
        table = _table_for_label(tex, "tab:complexity")
        profiles = _profile_rows()
    except ValueError as error:
        result.fail(str(error))
        return
    baseline = profiles.get("baseline")
    proposed = profiles.get("baseline_triple_stack_v3_msa_ocv_gbc")
    if baseline is None or proposed is None:
        result.fail("profile log lacks baseline or msa_ocv_gbc measurements")
        return
    _expect_values(
        result, table, "Baseline", baseline, "complexity table baseline", places=3
    )
    _expect_values(
        result,
        table,
        r"Proposed \method",
        proposed,
        "complexity table proposed",
        places=3,
    )

    relative_line = _table_row(table, "Relative change")
    actual = [float(value) for value in re.findall(r"[+-]\d+\.\d", relative_line)]
    expected = [round((new - old) / old * 100.0, 1) for old, new in zip(baseline, proposed)]
    if actual != expected:
        result.fail(f"complexity relative changes: expected {expected}, found {actual}")
    result.checked("complexity measurements and relative changes agree with the profile log")


def _check_figure_references(result: AuditResult, tex: str, paper_dir: Path) -> None:
    references = []
    references.extend(re.findall(r"\\input\{([^}]+)\}", tex))
    references.extend(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex))
    for reference in references:
        path = paper_dir / reference
        if path.suffix:
            candidates = (path,)
        else:
            candidates = (path.with_suffix(".tex"), path.with_suffix(".pdf"), path.with_suffix(".png"))
        if not any(candidate.is_file() for candidate in candidates):
            result.fail(f"missing manuscript asset: {reference}")
    stale = (
        "overall_architecture_visual",
        "gab_visual.tex",
        "module_details.tex",
        "overall_architecture_v2.tex",
        "gab_pipeline_v2.tex",
        "module_details_v2.tex",
        "qualitative_comparison.png",
    )
    for name in stale:
        if name in tex:
            result.fail(f"manuscript still references superseded asset: {name}")
    if len(_read_tsv(QUALITATIVE_MANIFEST)) != 4:
        result.fail("qualitative manifest must contain exactly four selected samples")
    result.checked(
        f"all {len(references)} manuscript assets exist and current v3 structure figures are referenced"
    )


def _check_required_protocol_text(result: AuditResult, tex: str) -> None:
    required = {
        "validation-only checkpoint selection": "validation split at threshold 0.5",
        "loss protocol": "0.87 for BCE and 0.13 for Dice",
        "direction bins": "all three datasets use 180 direction bins",
        "no augmentation": "No random crop, flip, color, or geometric augmentation",
        "DeepCrack local count": "527 valid image--mask pairs",
        "architecture-search limitation": "test-set metrics were repeatedly inspected",
    }
    for label, text in required.items():
        if text not in tex:
            result.fail(f"manuscript is missing required protocol statement: {label}")
    result.checked("selection, loss, preprocessing, and limitation statements are explicit")


def _check_placeholders(result: AuditResult, tex: str, strict: bool) -> None:
    hard_placeholders = re.findall(r"\b(?:TBD|TODO|FIXME)\b|\?\?", tex, flags=re.IGNORECASE)
    if hard_placeholders:
        result.fail(f"unresolved hard placeholders found: {sorted(set(hard_placeholders))}")
    review_placeholders = (
        "Journal information omitted for review",
        "Author information omitted for review",
        "Institution information omitted for review",
    )
    present = [placeholder for placeholder in review_placeholders if placeholder in tex]
    if present:
        message = "review-mode metadata still requires target-journal/author replacement: " + "; ".join(present)
        if strict:
            result.fail(message)
        else:
            result.warn(message)


def audit_manuscript(
    root: Path = ROOT,
    tex_path: Path | None = None,
    strict_placeholders: bool = False,
) -> AuditResult:
    """Run the submission audit without modifying source files."""
    if root.resolve() != ROOT.resolve():
        raise ValueError("This audit currently requires the MixerCSeg project root")
    result = AuditResult()
    tex_path = tex_path or DEFAULT_TEX
    if not tex_path.is_file():
        result.fail(f"Missing manuscript source: {tex_path}")
        return result
    tex = tex_path.read_text(encoding="utf-8")
    if not _check_evidence_files(result):
        return result

    core_rows = _read_tsv(CORE_METRICS)
    _check_dataset_splits(result, tex)
    _check_seed_and_summary_tables(result, tex, core_rows)
    _check_cross_dataset_table(result, tex, core_rows)
    _check_ablation_table(result, tex)
    _check_complexity_table(result, tex)
    _check_figure_references(result, tex, tex_path.parent)
    _check_required_protocol_text(result, tex)
    _check_placeholders(result, tex, strict_placeholders)
    return result


def _print_result(result: AuditResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"manuscript consistency audit: {status}")
    for check in result.checks:
        print(f"  OK: {check}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")
    for error in result.errors:
        print(f"  ERROR: {error}")
    print(
        f"summary: checks={len(result.checks)} warnings={len(result.warnings)} "
        f"errors={len(result.errors)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument(
        "--strict-placeholders",
        action="store_true",
        help="Treat anonymous-review author, affiliation, and journal placeholders as errors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_manuscript(tex_path=args.tex, strict_placeholders=args.strict_placeholders)
    _print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
