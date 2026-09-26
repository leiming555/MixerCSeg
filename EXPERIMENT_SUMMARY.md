# MixerCSeg Experiment Summary

Last updated: 2026-09-26

## 1. Current conclusion

The current paper model is TripleStack-v3 `msa_ocv_gbc`:

```text
TransMixer -> MSA -> HoGEdgeGateConv (DEGConv) -> OCV -> GBC
```

It remains the strongest candidate supported by validation-selected, paired multi-seed
evidence. Later structures sometimes produced higher exploratory or single-seed scores,
but none has replaced V3 after the fair four-seed checks.

The final training protocol used for paper evidence is:

```text
dataset-dependent fixed train/val/test splits
50 epochs, nbins=180
BCE/Dice weights = 0.87/0.13
checkpoint selection = validation mIoU at threshold 0.5
test evaluation = once after checkpoint selection
CCEM/Tversky/Boundary Loss = disabled
```

Historical results selected by repeatedly inspecting the test set are retained only as
exploratory evidence. Validation reselection corrects epoch selection, but it does not
remove architecture-selection bias from earlier searches.

## 2. Main CrackMap result

Four paired seeds are `42`, `3407`, `2026`, and `1234`.

| Model | mIoU mean +/- SD | F1 mean +/- SD | ODS mean | OIS mean | Decision |
|---|---:|---:|---:|---:|---|
| Baseline MixerCSeg | 0.787300 +/- 0.021224 | 0.743796 +/- 0.032468 | 0.748387 | 0.754513 | Reference |
| V3 `msa_ocv_gbc` | **0.809307 +/- 0.009342** | **0.776300 +/- 0.012978** | **0.790710** | **0.798730** | Current paper model |
| V7 `wma_ctv_dgb` | 0.787006 +/- 0.011648 | 0.744783 +/- 0.016567 | - | - | Rejected after validation reselection |
| V12 `wfe_cpa_bcs` | 0.751457 +/- 0.072320 | 0.686425 +/- 0.113145 | 0.748324 | 0.753352 | Rejected, 1/4 paired wins over V3 |
| V3 `id_ocv_gbc` | 0.788104 +/- 0.034420 | 0.745076 +/- 0.049787 | 0.768341 | 0.772860 | Rejected, 2/4 paired wins over V3 |
| V3 `msa_id_gbc` | 0.807927 +/- 0.005479 | 0.774316 +/- 0.007739 | 0.783947 | 0.790478 | Rejected, 2/4 paired wins over V3 |

Relative to the paired baseline, V3 improves mean mIoU by `+0.022007` and mean F1
by `+0.032504`. The standard deviation is also lower for both metrics. With only four
pairs, these results should be described as repeated-seed robustness evidence rather
than statistical significance.

## 3. Boundary and topology evidence

All topology metrics use the validation-selected test probability maps at threshold
`0.5`. Boundary F1 uses a two-pixel tolerance.

| Model | clDice | Boundary F1 | Component-count MAE | Endpoint-count MAE |
|---|---:|---:|---:|---:|
| Baseline | 0.851603 +/- 0.039896 | 0.468365 +/- 0.041033 | 2.333 | 43.854 |
| V3 `msa_ocv_gbc` | **0.863435 +/- 0.019984** | **0.532918 +/- 0.036884** | **1.625** | **41.240** |
| `id_ocv_gbc` | 0.858643 +/- 0.024295 | 0.463416 +/- 0.130089 | 2.667 | 45.583 |
| `msa_id_gbc` | **0.865733 +/- 0.019058** | 0.529730 +/- 0.015972 | 1.875 | 44.146 |

V3 improves the four-seed mean Boundary F1 by `+0.064554`, clDice by `+0.011832`,
component-count MAE by `0.708`, and endpoint-count MAE by `2.615`. The paired
Boundary F1 change is positive for 3/4 seeds, while clDice is positive for only 1/4;
therefore the clDice mean must not be presented as a consistent per-seed gain.

These metrics directly support the paper's boundary-preservation and connectivity
claims better than pixel metrics alone.

## 4. Cross-dataset evidence

The completed seed-42 validation-selected comparison is mixed:

| Dataset | Model | mIoU | F1 | clDice | Boundary F1 |
|---|---|---:|---:|---:|---:|
| DeepCrack | Baseline | **0.913619** | **0.909659** | **0.923607** | **0.901751** |
| DeepCrack | V3 | 0.911203 | 0.907004 | 0.912010 | 0.891032 |
| CamCrack789 | Baseline | 0.847694 | 0.827765 | 0.900836 | 0.873834 |
| CamCrack789 | V3 | **0.857028** | **0.839694** | **0.902801** | **0.891349** |

V3 improves CamCrack789 but is slightly below the baseline on DeepCrack. The paper
must describe this as dataset-dependent transfer, not universal cross-dataset
improvement.

An eight-training paired extension is currently running under systemd unit
`mixercseg-v3-cross-20260926_170134.service`. It adds baseline and V3 seeds `3407`
and `2026` on both datasets, then combines them with seed `42` to produce three-seed
means, sample standard deviations, paired deltas, descriptive confidence intervals,
and topology metrics.

## 5. What the later searches established

- CCEM, Tversky, scale-adaptive fusion, PaperStack, and early stacked modules were
  useful architecture exploration, but most early numbers used different losses or
  test-observed selection and are not directly comparable with the final protocol.
- V7 `wma_ctv_dgb` looked strong under the old exploratory protocol. Validation-only
  reselection reduced its four-seed mean to `0.787006/0.744783`, so it is not the
  final paper model.
- V8, V9, and V10 did not provide a robust replacement for V7 during their original
  searches and were not promoted to the final evidence set.
- V12 `wfe_cpa_bcs` improved over V3 at seed 42 in its first comparison, but the
  four-seed run collapsed to `0.751457/0.686425`; this is a clear example of why a
  single-seed winner must not determine the final architecture.
- Removing MSA (`id_ocv_gbc`) or OCV (`msa_id_gbc`) can look competitive for an
  individual seed, but neither simplified candidate passed the four-seed promotion
  rule. The complete V3 remains the defensible choice.

## 6. Paper readiness

The project has enough evidence for a conservative engineering/image-processing SCI
Q4 or EI submission if the claims remain limited:

1. Present V3 as a geometry-aware MixerCSeg extension, not as a universally superior
   crack segmentation model.
2. Use only validation-selected results in the main tables.
3. Report the full factorial and removal results, including configurations that beat
   the complete model for a single seed.
4. Include Boundary F1 and topology metrics alongside mIoU/F1/ODS/OIS.
5. State that the four-seed intervals are descriptive and that architecture search
   previously observed CrackMap test results.
6. Keep the DeepCrack regression visible and update the cross-dataset table after the
   running paired experiment finishes.
7. Add controlled recent-method comparisons before submission when reproducible
   implementations are available.

The largest remaining experimental gap is a fair, same-code comparison against recent
methods. The running cross-dataset experiment addresses the second-largest gap: the
current transfer result is only seed 42.

## 7. Evidence locations

- Validation-selected main audit:
  `work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12/`
- V12 four-seed decision:
  `work_dirs/20260926_134602_triple_stack_v12_wfe_cpa_bcs_staged_gpu012/`
- Simplified V3 candidate decision:
  `work_dirs/20260926_153810_v3_simplified_candidates_staged_gpu012/`
- Boundary/topology audit:
  `work_dirs/20260926_164529_topology_metrics_after_v3_candidates/`
- Running cross-dataset three-seed validation:
  `work_dirs/20260926_170134_v3_cross_dataset_multiseed_gpu012/`
- Paper source:
  `paper_msa_ocv_gbc/msa_ocv_gbc.tex`

## 8. Reproducibility status

- Dataset split audits pass for CrackMap, DeepCrack, and CamCrack789.
- New paper evidence uses validation-only checkpoint selection.
- Probability-map evaluation uses sigmoid outputs and a fixed threshold of 0.5.
- Pixel, boundary, and topology metrics are generated from the same saved probability
  maps.
- Training queues run as user systemd services and preserve interrupted attempts.
- Runtime outputs, checkpoints, probability maps, and timestamped logs are not part of
  the source commits.
