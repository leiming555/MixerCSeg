from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_msa_ocv_gbc"
MANUSCRIPT = PAPER / "msa_ocv_gbc.tex"
STRUCTURE_FIGURES = (
    "overall_architecture_v3.tex",
    "gab_pipeline_v3.tex",
    "msa_detail_v3.tex",
    "ocv_detail_v3.tex",
    "gbc_detail_v3.tex",
)


def test_manuscript_uses_only_v3_structure_figures_without_resizebox():
    source = MANUSCRIPT.read_text(encoding="utf-8")
    for name in STRUCTURE_FIGURES:
        assert f"figures/{name}" in source
        assert f"resizebox{{\\textwidth}}{{!}}{{\\input{{figures/{name}}}" not in source
    for old in ("overall_architecture_v2.tex", "gab_pipeline_v2.tex", "module_details_v2.tex"):
        assert old not in source


def test_v3_figures_keep_readable_type_and_explicit_dimensions():
    for name in STRUCTURE_FIGURES:
        source = (PAPER / "figures" / name).read_text(encoding="utf-8")
        assert "\\begin{tikzpicture}" in source
        assert "font=\\sffamily\\footnotesize" in source
        assert "\\scriptsize" not in source
        assert "\\tiny" not in source
        assert "\\resizebox" not in source
        assert "x=1cm" in source and "y=1cm" in source


def test_module_details_are_split_and_routes_are_orthogonal():
    source = MANUSCRIPT.read_text(encoding="utf-8")
    for label in ("fig:msa_detail", "fig:ocv_detail", "fig:gbc_detail"):
        assert f"\\label{{{label}}}" in source
    for name in STRUCTURE_FIGURES:
        figure = (PAPER / "figures" / name).read_text(encoding="utf-8")
        assert "overlay" not in figure
    for name in ("msa_detail_v3.tex", "ocv_detail_v3.tex", "gbc_detail_v3.tex"):
        figure = (PAPER / "figures" / name).read_text(encoding="utf-8")
        assert "|-" in figure or "-|" in figure
