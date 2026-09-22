# xai-ood

Training-free out-of-distribution detection on frozen DINOv2 embeddings: the
code, the pre-specified analysis plan and the analysis scripts behind a
bachelor's thesis.

## What this is

Images are encoded once with a **frozen DINOv2 ViT-B/14** backbone (768-dimensional
CLS token, mean patch token cached alongside it) and never fine-tuned. Every
detector is then a scoring rule over those fixed embeddings, so no classifier
head and no training run sits between the representation and the result.

Fifteen scorer configurations, in three families:

| Family | Configurations |
|---|---|
| Gaussian, a 2x2 factorial of {marginal, class-conditional} x {diagonal, full} | `marginal_diagonal`, `marginal_full`, `class_conditional_diagonal`, `class_conditional_full`, plus `rmd`, `rmd_pp`, `marginal_full_pp`, `class_conditional_full_pp` |
| k-nearest-neighbour distance | `knn_unnormalized`, `knn_normalized` |
| PCA residual, on all in-distribution embeddings and on the per-class means | `pca_residual_all_id`, `pca_residual_class_mean`, their two L2 variants, and `pca_residual_class_mean_whitened` |

The 2x2 is the point of the design: comparing a marginal diagonal score against
a class-conditional full one varies covariance structure and label use at once,
and the factorial separates them into two main effects and an interaction.

Evaluation follows **OpenOOD's standard and full-spectrum protocols** at the
pinned commit below. CIFAR-10 is in-distribution, CIFAR-10-C is the
covariate-shifted in-distribution set (a hand-built imglist over 15 corruptions
at 5 severities), CIFAR-100 and Tiny ImageNet are near-OOD, and MNIST, SVHN,
Texture and Places365 are far-OOD. Uncertainty is a **cluster bootstrap over
the 9,000 source photographs**, so the 76 corrupted rows derived from one
photograph resample together rather than as 76 independent observations.
Intervals are 95% percentile intervals at B = 1,000 and are reported
descriptively: **no significance claim is attached to any comparison**. That is
a branch elected on 2026-09-08, before any result existed, and it is recorded
with its dates in [`docs/analysis-plan.md`](docs/analysis-plan.md).

## The thesis this belongs to

| | |
|---|---|
| Title | *Empirical Analysis of the OOD Robustness for Foundation Model Embeddings* |
| Author | Jonas Reutter |
| Institution | University of Bamberg |
| Year | 2026 |
| Submission date | 2026-09-24 |

This repository is the code attachment that accompanies that submission. It is
not a library anyone is asked to depend on.

## How this code is distributed

> **PLACEHOLDER, not yet decided.** Whether this repository is public, private
> with the examiners invited, or supplied as a git bundle alongside the thesis
> has not been settled. Replace this block with the route actually taken, and
> with the URL or filename a reader should use, before the thesis is handed in.

## Layout

```
src/xai_ood/       the library: scorers, metrics, the cluster bootstrap, the schema
scripts/           the invocation record of Phase 1, and the Phase 2 scoring driver
analysis/          the scripts that turn a scoring run into the reported tables and figures
docs/              the pre-specified analysis plan
data_manifests/    the CIFAR-10-C index map, committed so the cs-ID tree is rebuildable
env-lockfiles/     the second conda environment, and the two pins used during install
logs/              the extraction runs as they happened, successes and failures alike
tests/             617 tests over the library and the driver
```

## Setup

There are **two** conda environments, and they are deliberately separate.

| Environment | File | What it is for |
|---|---|---|
| `openood` | `env-lockfiles/openood-env-2026-07-31.yml` | the OpenOOD reproduction only |
| `xai-ood` | `environment.yml` | embedding extraction, the scorers, the analysis |

```bash
conda env create -f env-lockfiles/openood-env-2026-07-31.yml   # creates openood
conda env create -f environment.yml                            # creates xai-ood
conda activate xai-ood
pip install -e .
```

Each file declares its own `name:`, so neither call needs `-n` and neither can
clobber the other. They differ in more than the name: `xai-ood` carries
`faiss-gpu`, `pyarrow` and `pytest`, `openood` carries `faiss-cpu` and the
bundled NVIDIA CUDA wheels.

Both sit on the same stack: Python 3.10.20, PyTorch 2.13.0, NumPy 1.26.4,
OpenCV 5.0.0.93, scikit-learn 1.7.2. **The version of record for each
environment is its own YAML file**, which is a full `conda env export`.
`env-lockfiles/constraints.txt` is **not** the pin for all of that: it is two
lines, `numpy==1.26.4` and `opencv-python==5.0.0.93`, held as an installer
constraint because those two were what resolved badly on the workstation.

One honest limit on the environment files. OpenOOD is an **editable install
from a local checkout**, which `conda env export` flattens to a bare version
string, so replaying either YAML on a fresh machine does not reconstruct the
OpenOOD that ran. The provenance of record is the manifest written beside each
artefact, which records the OpenOOD commit explicitly.

`src/xai_ood` itself imports **numpy and pandas only**. scipy and scikit-learn
appear in the test suite as independent cross-checks, never as dependencies, and
`tests/test_import_hygiene.py` enforces that.

## Reproducing a reported number

### The anchor that needs no embeddings

MDS, RMDS and kNN reproduce OpenOOD v1.5's published CIFAR-10 standard-protocol
AUROC to within 0.01 points. OpenOOD was cloned separately rather than vendored
here, pinned at commit `8d44375e4c695d03d2b97850b754f24fd4bda447` (tag `v1.5`)
and installed editable into both environments from that one checkout, so the two
cannot drift apart.

| Scorer | near-OOD published | near-OOD reproduced | delta | far-OOD published | far-OOD reproduced | delta |
| ------ | ------------------ | ------------------- | ----- | ----------------- | ------------------ | ----- |
| MDS    | 84.20 (±2.40)      | 84.20               | 0.00  | 89.72 (±1.36)     | 89.72              | 0.00  |
| RMDS   | 89.80 (±0.28)      | 89.80               | 0.00  | 92.20 (±0.21)     | 92.20              | 0.00  |
| kNN    | 90.64 (±0.20)      | 90.65               | 0.01  | 92.96 (±0.14)     | 92.96              | 0.00  |

Published values: OpenOOD v1.5 (Zhang et al., 2024, *Journal of Data-centric
Machine Learning Research*).

### A named thesis number, and exactly what it costs

The **covariance main effect on far-OOD AUROC under the standard protocol**,
reported as **+21.84 points with a 95% interval of [+21.61, +22.07]**. The chain
that produces it:

1. `scripts/extract_embeddings.py` caches CLS and mean-patch embeddings per
   split. It is fully parameterised; the five `scripts/run_*.sh` files are the
   invocation record of the run that produced the cache, not a portable runner.
2. `scripts/build_cifar10c_index_map.py`, then `scripts/build_csid_images.py`,
   build the covariate-shifted tree. Both are fully parameterised, and the
   committed `data_manifests/cifar10c_index_map.csv` lets the first be skipped.
3. `scripts/run_phase2.py` fits every scorer and writes
   `results/runs/<run-id>/scores/per_sample_scores.parquet` beside a
   `run_metadata.json` recording the seed, both commit hashes, the per-scorer
   hyperparameters, the shrinkage diagnostics for both arms and the degeneracy
   report.
4. **A gap, stated rather than hidden.** `xai_ood.bootstrap` implements the whole
   resampling procedure and has **no caller in this repository outside
   `tests/`**. The bootstrap replicate objects the analysis scripts read were
   produced by calling `run_bootstrap` on a cached score dump directly. A reader
   who wants to regenerate them writes that call themselves;
   `tests/test_end_to_end.py` is the closest worked example here.
5. `analysis/derive_contrasts.py` reads those replicate objects, refuses to
   report anything unless it first reproduces the gated marginals table to
   float64 noise, and writes
   `factorial-and-pairwise-contrasts-dataset-averaged.csv`. The number above is
   the row `standard,covariance_main_effect,far_ood`.

So a reader can check the **method** end to end, and can check the **number**
only with a score dump and the replicate objects in hand. Which of those travel
with the thesis is the distribution question left open above.

The other twelve scripts in `analysis/` cover the per-dataset tables, the score
distributions and their figures, method agreement between scorers, the
cost-quality frontier, the inference footprint, and the timing re-measurements.

## Embedding cache layout

```
<data-root>/embeddings/
└── dinov2_vitb14/
    └── res224_bicubic/
        ├── cifar10_train/
        │   ├── cls.npy
        │   ├── patchmean.npy
        │   ├── labels.npy
        │   ├── filelist.txt
        │   └── manifest.json
        ├── cifar10_test/
        └── ...
```

## What is deliberately absent, and why

- **The embeddings.** 656 MB for the nine standard-protocol splits, 1,117 MB
  with the covariate-shifted grid. They are fully derived, and the derivation is
  published in their place: `scripts/extract_embeddings.py` refuses to run unless
  it can hash the backbone checkpoint, and records that hash in the manifest it
  writes; `scripts/cache_verify.py` checks a cache against row counts named in
  advance. The 2026-08-18 extraction predates the checkpoint pin, so its
  manifests record the OpenOOD commit and the imglist SHA-256 instead.
- **The per-sample score dumps.** 19.7 to 88.3 MiB each, roughly 405 MB across
  the seven runs. Excluded by `.gitignore`: a file that size does not belong in a
  git tree, and the largest is close enough to a hosting size limit that one more
  score column would break the push.
- **The one-off audit and verification harnesses.** Each was written to answer
  one question once. They are single-use and unreviewed for publication, and the
  thesis cites their results rather than inviting a rerun.
- **A portable extraction runner.** The five `run_*.sh` files hardcode one
  workstation's paths deliberately, because they are the record of what produced
  the cache. `scripts/cache_verify.py` carries such a path with no such excuse,
  and that is a known deviation rather than a design.
- **`scripts/spot_check.py`** is nine lines of scratch and is not an entry point.

## Provenance, and which commit you are reading

Every run manifest records the commit of the code that produced it. That field
records a hash; it does not promise the hash is reachable from wherever this was
cloned. Two of the seven runs, `2026-09-14-standard-pp-shrunk` and
`2026-09-14-standard-pp-unshrunk`, were produced at a commit that adds the two
normalised Gaussian Full configurations, `marginal_full_pp` and
`class_conditional_full_pp`.

To check whether the copy you are reading can produce them:

```bash
grep -c 'GaussianConfig(' src/xai_ood/methods/gaussian.py
```

**Eight** means those two cells are reachable. **Six** means the copy predates
them and those two cells cannot be reproduced from it.

## Licence

MIT. The full text is in [`LICENSE`](LICENSE).
