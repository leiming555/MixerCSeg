from tools.check_paper_consistency import DEFAULT_TEX, ROOT, audit_manuscript


def test_current_manuscript_matches_traceable_evidence():
    result = audit_manuscript()
    assert result.errors == []
    assert len(result.checks) >= 8


def test_metric_drift_is_detected(tmp_path):
    source = DEFAULT_TEX.read_text(encoding="utf-8")
    changed = source.replace(
        "42 & Baseline & -- & 0.7870",
        "42 & Baseline & -- & 0.7871",
        1,
    )
    manuscript = tmp_path / "manuscript.tex"
    manuscript.write_text(changed, encoding="utf-8")
    result = audit_manuscript(tex_path=manuscript)
    assert any("seed table 42/Baseline" in error for error in result.errors)


def test_missing_figure_is_detected():
    source = DEFAULT_TEX.read_text(encoding="utf-8")
    changed = source.replace("figures/overall_architecture_v3.tex", "figures/missing_v3.tex", 1)
    manuscript = ROOT / "paper_msa_ocv_gbc" / "temporary_missing_figure_test.tex"
    try:
        manuscript.write_text(changed, encoding="utf-8")
        result = audit_manuscript(tex_path=manuscript)
    finally:
        manuscript.unlink(missing_ok=True)
    assert any("missing manuscript asset" in error for error in result.errors)


def test_review_metadata_is_warning_unless_strict():
    normal = audit_manuscript()
    strict = audit_manuscript(strict_placeholders=True)
    assert normal.ok and any("review-mode metadata" in warning for warning in normal.warnings)
    assert any("review-mode metadata" in error for error in strict.errors)
