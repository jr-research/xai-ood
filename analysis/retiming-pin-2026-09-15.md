# Re-timing Pin, 2026-09-15

*Written and committed BEFORE the re-timing in `retime_scoring_cost.py` was run, so that its
acceptance rule is a pre-specified analysis plan rather than a description of whatever came out.
The defect under repair is the single unreplicated timing the published run carries.*

## What is being re-measured, and what cannot be

The published per-method scoring cost is the `timing.score` block of
`results/runs/2026-09-10-full-spectrum-shrunk/scores/run_metadata.json`: one wall-clock figure per
scorer over 312,660 rows, thirteen scorers in one sequential process, on the lab workstation at
Python 3.10.20 and numpy 1.26.4. **No repeats, no warm-up discard, no interleaving, no interval.**

**The re-timing runs on a different machine**, a 4-core desktop at Python 3.12.3 and numpy 1.26.4,
with no route to the lab workstation. So the pin below is stated on **ordering and ratios only**.
Absolute microseconds per sample are not expected to reproduce and no rule here requires them to.
The run-to-run spread measured here is the desktop's spread, not the workstation's, and every
report of it must say so.

**The published figures are read out of the manifest by the script at run time**, not copied into it
or into this note, so that a re-derivation cannot silently compare against a stale transcription.

## The rules, decidable before the data

Tiers, defined by the published figures and fixed here:

- **A**, published under 20 microseconds per sample: `pca_residual_class_mean`,
  `pca_residual_class_mean_l2`, `marginal_diagonal`, `marginal_full`,
  `pca_residual_class_mean_whitened`, `pca_residual_all_id`, `pca_residual_all_id_l2`
- **B**, published between 80 and 130: `class_conditional_diagonal`, `class_conditional_full`,
  `rmd`, `rmd_pp`
- **C**, published above 800: `knn_unnormalized`, `knn_normalized`

| | Rule | Verdict if violated |
|---|---|---|
| R1 | `max(median of A) < min(median of B)` and `max(median of B) < min(median of C)` | **STOP.** A finding about the harness, not a result |
| R2 | For all four (marginal Gaussian cell, kNN column) pairs, `median(kNN) / median(marginal) >= 50` | **STOP.** The two-orders-of-magnitude argument fails |
| R3 | Every ordered pair whose **published** ratio is at least 1.5 reproduces the sign of its median difference | **STOP.** A gap the published run resolved has inverted |
| R4 | Pairs whose published ratio is under 1.5 are **not pinned** | Reported with their spreads. These are the pairs n = 1 could not resolve, and a swap among them is the expected outcome of measuring a spread, not a defect |
| R5 | Spearman rank correlation of re-timed medians against published figures | Reported, not pinned |

## The protocol, fixed here

- **Interleaved, cyclically.** In round `r` the thirteen scorers are called in the order rotated by
  `r`, so over any thirteen consecutive rounds each scorer occupies each position exactly once.
  This is the repair for the specific defect the published timing has: drift over the run is
  attributed to whichever scorer happened to be running at the time.
- **Three warm-up rounds, discarded explicitly.** First touch of each fitted array faults pages in
  and the BLAS thread pool is created on the first call, so round 1 is not a sample of steady state.
- **Fifteen kept rounds.** Chosen because the order statistics `x(4)` and `x(12)` of fifteen give a
  distribution-free 96.5% interval for the median, and neither endpoint is an extreme value, so one
  scheduling stall cannot become an interval endpoint. Fewer than thirteen kept rounds would also
  break the position balance above.
- **Median and spread, never a mean.** A scheduling stall is a one-sided contaminant that inflates a
  mean and is indistinguishable from real cost; the median is the statistic that survives it.
  Reported per method: median, `[x(4), x(12)]`, interquartile range, min, max.
- **Block size.** One block of rows is scored per call. The block is drawn once, seeded, from the
  same cached `cls` embeddings the published run scored, and is identical for every method and every
  round. Block size is set by a calibration pass to keep the whole protocol inside twenty minutes
  and is **recorded in the output**; a block-size sweep afterwards reports whether per-sample cost
  depends on it, since the published figure is one call over 312,660 rows and these are not.
- **A data-dependence control.** Two blocks of equal size from different splits are timed, because
  every claim here assumes per-sample cost is a function of row count and dimension and not of the
  values. That assumption is measured rather than asserted.

## Out of scope, deliberately

**Frontier membership is not touched.** It is a joint claim about cost and AUROC, and a method
enters or leaves the frontier on the quality coordinate. Re-timing the cost axis does not rescue
it, and nothing here implies it does.
