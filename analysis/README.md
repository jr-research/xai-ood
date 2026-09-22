# analysis

The scripts that turn a scoring run into the tables and figures the thesis
reports. They were written alongside the write-up rather than as part of the
library, and they sit here because a reported number should be reproducible from
published code rather than described.

Nothing here is imported by `src/xai_ood`, and nothing here imports it. Constants
that both need (the protocol-to-split mapping, the palette) are restated rather
than imported, on purpose: a pin that shares code with the thing it checks is
not a pin. Any drift between the two copies is a bug in this directory.

## Paths

Every script resolves the repository root as the directory above this one, and
takes `XAI_OOD_ROOT` as an override when a run tree lives outside the checkout.
Two scripts need one more thing:

- `derive_spectral_panels.py` needs `XAI_OOD_SPECTRAL` pointing at the output of
  the spectral decomposition, which is a separate one-off harness and is not part
  of this repository. Without it the script refuses and says so.
- The four timing scripts read the embedding cache at `<root>/embeddings-local/`,
  which is not tracked here. See the repository README on the embeddings.

## What each script needs, and what it writes

| Script | Reads | Writes |
|---|---|---|
| `derive_contrasts.py` | `bootstrap_singleton.pkl` from the standard and full-spectrum runs, plus the gated marginals CSV | the degradation table, the factorial and pairwise contrasts, the marginals for both protocols |
| `derive_per_dataset.py` | the score dumps and the replicate objects, both arms | the per-dataset table of record, and the three within-group views built on it |
| `derive_distributions.py` | the full-spectrum score dumps, both arms, and the per-dataset table of record | quantile grids, histogram counts, the tail tables, the raw-score Spearman matrix |
| `derive_method_agreement.py` | one score dump and its replicate object | the ID-percentile agreement matrices, the clustering, the worst-wrong tables |
| `derive_spectral_panels.py` | a spectral decomposition's output and the per-dataset table of record | the two-panel spectral figure input |
| `derive_cost_quality.py` | one replicate object and one run manifest | the cost-quality Pareto frontier and its steps |
| `derive_footprint_and_amortisation.py` | one run manifest and the fitted classes in `src/xai_ood/methods/` | the inference footprint, the extraction cost per image, the amortisation table |
| `retime_scoring_cost.py` | the embedding cache | the re-timed per-method costs, per round, with medians and order statistics |
| `retime_data_dependence_paired.py` | the embedding cache | the paired data-dependence control |
| `retime_tier_a_at_scale.py` | the embedding cache | the cheap tier re-timed at a 25x larger block |
| `make_distribution_figures.py` | what `derive_distributions.py` and `derive_spectral_panels.py` wrote | four thesis figures and one appendix atlas, PDF and PNG |
| `plot_cost_quality.py` | `cost-quality-pareto.csv` | the frontier figure |
| `plot_score_distributions.py` | one score dump | two working figures |

Each script writes into this directory, so several of them are inputs to each
other. The order that resolves every dependency is the order of the table above.

## Two things that are easy to get wrong

**The estimand.** `bootstrap_singleton.pkl` holds per-dataset replicate arrays
and `bootstrap_grouped.pkl` holds pooled ones, and the filenames do not say which
is the headline. The declared headline is the **dataset-averaged** figure, the
unweighted mean of the per-dataset AUROCs, averaged **inside** each replicate.
Reading the pooled arrays instead is not a rounding difference: far-OOD rows are
51 per cent MNIST, and the far-OOD covariance main effect reads +21.84 on the
declared estimand against +25.65 on the pooled one. `derive_contrasts.py` exists
because three figures once went out on the wrong side of that.

**The pins refuse rather than warn.** Most of these scripts rebuild a published
table first and exit non-zero if the rebuild does not match to float64 noise, and
they print how many cells they checked, because a pin that checked none passes
silently. If one refuses, nothing below its output is reportable.

`retiming-pin-2026-09-15.md` is the pre-specified acceptance rule for the three
timing scripts, written and dated before they were run.
