# Submission Readiness

## Current Status

The experiment package is complete for the current MSA--OCV--GBC manuscript. No training queue is active. The manuscript uses validation-only checkpoint selection, reports all four paired CrackMap seeds, preserves the negative DeepCrack result, and includes the complete seed-42 factorial ablation.

The quantitative content is ready within the reviewed scope. The remaining blockers are submission metadata and venue-specific formatting, not additional model training.

## Verified Evidence

- Four paired CrackMap seeds: baseline versus `msa_ocv_gbc`.
- Validation-selected single-seed transfer on DeepCrack and CamCrack789.
- Complete `2 x 2 x 2` MSA/OCV/GBC factorial study plus the original baseline.
- Parameter, FLOP, latency, FPS, model-size, and GPU-memory measurements.
- Deterministic ground-truth-only qualitative sample and crop selection.
- Dataset pairing and cross-split duplicate audit with zero reported errors.
- PDF compilation without undefined references, overfull boxes, or missing figures.

## Required Before Submission

- Select the target journal or conference and replace the review-mode journal field.
- Insert the real author names, affiliations, corresponding-author email, and ORCID identifiers as required by the venue.
- Confirm whether the venue is single-blind or double-blind before removing anonymous metadata.
- Add the funding statement, acknowledgements, author-contribution statement, and any required ethics declaration.
- Prepare an anonymized code/evidence archive for review, or insert a permanent repository URL if anonymity is not required.
- Confirm dataset redistribution rights before including images, labels, checkpoints, or probability maps in an archive.
- Apply the target venue's page layout, reference style, highlights, graphical-abstract, and supplementary-file rules.
- Run the strict metadata audit immediately before upload.

## Reproducibility Command

From the project root:

```bash
/home/lm/miniconda3/envs/MixerCSeg/bin/python tools/check_paper_consistency.py
```

The normal audit allows the three intentional anonymous-review placeholders and reports them as warnings. The final upload check should use:

```bash
/home/lm/miniconda3/envs/MixerCSeg/bin/python tools/check_paper_consistency.py --strict-placeholders
```

The strict command must pass only after journal, author, and affiliation information has been finalized.

## Evidence Locations

- Core re-evaluation: `work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12/`
- Final factorial study: `work_dirs/20260919_173408_v3_factorial_reselect9_gpu012/`
- Complexity profile: `logs/20260806_220017_CrackMap_triple_stack_v4_search12_validate8_gpu0.profile.log`
- Qualitative manifest: `paper_msa_ocv_gbc/figures/qualitative_selection_v2.tsv`
- Manuscript source: `paper_msa_ocv_gbc/msa_ocv_gbc.tex`
