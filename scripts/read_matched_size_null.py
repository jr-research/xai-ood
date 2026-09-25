"""Negative controls and the matched-size null for C6, with the reading applied.

Deliberately a **separate script from** ``read_run_diagnostics.py``, and the
separation is the point rather than an accident of how it was written. The
design requires the diagnostics to be read before any
AUROC. Merging the two would let one invocation print a diagnostic and a metric
together, which makes the required ordering a matter of who reads which line
first. Two scripts make the ordering a matter of which one you run.

This one **does** compute AUROCs, and only the ones C6 needs: the twenty
matched-size-null draws and ``pca_residual_all_id``. The other twelve scorers
are not read here, because C6's null has to be read before any headline AUROC
and reading them would defeat that.

Two controls, answering different questions:

* **Shuffled ID and OOD labels.** Confirms the pipeline does not manufacture
  separation. Expected 0.5. This is the integrity check.
* **The matched-size null**, twenty draws of a random subspace at the same
  dimension the variance rule selected, same centre, same fitting split.
  Answers whether the variance-ordered directions were the right ones to
  discard, or whether discarding any ``d`` of 768 leaves an informative
  residual norm. It is read as a **range**, never as a test: no p-value, no
  rank statistic, and no "beats twenty of twenty" phrasing, which would
  function as a test in prose.

The three readings are pre-decided and are applied here rather than chosen
after the numbers appear. **One boundary is an operationalisation and is
flagged as such**: the design distinguishes "clearly above all twenty" from
"overlapping the upper part" from "inside or below", and the middle two
overlap in words. The cut used here is ``> max``, then ``>= median``, then
below. The raw range and every draw are printed either way, so a reader can
apply a different cut to the same numbers.

    python3 read_matched_size_null.py --run-dir results/runs/<run-id>/scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xai_ood.bootstrap.plan import DEFAULT_OOD_GROUPS
from xai_ood.metrics import auroc

#: The configuration the control matches, and whose comparison against the
#: naive marginal-diagonal baseline the control qualifies. C6 names this one.
OBSERVED: str = "pca_residual_all_id"

#: Prefix of the matched-size-null columns.
CONTROL_PREFIX: str = "control_random_subspace_residual"

#: Draws of the shuffled-label control. Five is enough to show 0.5 is not a
#: single lucky permutation; this control has no declared reading to protect.
SHUFFLE_DRAWS: int = 5

READINGS: dict[int, str] = {
    1: (
        "ABOVE all twenty. C6 stands as a claim about WHICH directions were "
        "discarded, which is the claim it was written to make."
    ),
    2: (
        "OVERLAPPING the upper part of the range. The claim is substantially "
        "about dimensionality. Say so, and report the range."
    ),
    3: (
        "INSIDE or BELOW the range. C6 is a claim about dimensionality and not "
        "about subspace structure. A clean negative result, stated rather than "
        "buried."
    ),
}


def reading_for(observed: float, draws: list[float]) -> int:
    """Which of the three pre-decided readings fires.

    The boundary between 2 and 3 is the operationalisation the module docstring
    flags. Nothing here depends on the observed value having been seen first:
    the rule is a function of two numbers and was fixed before either existed.
    """
    ordered = sorted(draws)
    if observed > ordered[-1]:
        return 1
    if observed >= float(np.median(ordered)):
        return 2
    return 3


def group_auroc(
    frame: pd.DataFrame, column: str, datasets: tuple[str, ...], id_mask: np.ndarray
) -> tuple[float, float, dict[str, float]]:
    """Both group estimands for one score column.

    Returns the dataset-averaged figure, which is declared the
    headline because it is what OpenOOD computes and what Phase 1's
    reproduction table already used, and the pooled figure, which is reported
    beside it and never promoted.
    """
    id_scores = frame.loc[id_mask, column].to_numpy(dtype=np.float64)
    per_dataset = {
        dataset: float(
            auroc(
                id_scores,
                frame.loc[frame["split"] == dataset, column].to_numpy(dtype=np.float64),
            )
        )
        for dataset in datasets
    }
    pooled = float(
        auroc(
            id_scores,
            np.concatenate(
                [
                    frame.loc[frame["split"] == d, column].to_numpy(dtype=np.float64)
                    for d in datasets
                ]
            ),
        )
    )
    return float(np.mean(list(per_dataset.values()))), pooled, per_dataset


def shuffled_control(
    frame: pd.DataFrame, datasets: tuple[str, ...], id_mask: np.ndarray, seed: int
) -> list[float]:
    """AUROC with the ID and OOD labels permuted. Expected 0.5.

    The scores are left exactly as computed and only the assignment of rows to
    the two sides is permuted, so this isolates the labelling from the scoring.
    """
    rng = np.random.default_rng(seed)
    id_scores = frame.loc[id_mask, OBSERVED].to_numpy(dtype=np.float64)
    ood_scores = np.concatenate(
        [
            frame.loc[frame["split"] == d, OBSERVED].to_numpy(dtype=np.float64)
            for d in datasets
        ]
    )
    combined = np.concatenate([id_scores, ood_scores])
    n_id = id_scores.size
    out = []
    for _ in range(SHUFFLE_DRAWS):
        shuffled = combined[rng.permutation(combined.size)]
        out.append(float(auroc(shuffled[:n_id], shuffled[n_id:])))
    return out


def report(run_dir: Path, seed: int) -> int:
    manifest: dict[str, Any] = json.loads((run_dir / "run_metadata.json").read_text())
    frame = pd.read_parquet(run_dir / "per_sample_scores.parquet")
    controls = sorted(c for c in frame.columns if c.startswith(CONTROL_PREFIX))
    id_mask = (frame["split"] == "id_test").to_numpy()

    print(f"run      : {manifest['run_id']}")
    print(f"d        : {manifest['control']['d']} of 768")
    print(f"draws    : {manifest['control']['draws']}")
    print(f"read as  : {manifest['control']['read_as']}")

    print(f"\n=== negative control: shuffled ID and OOD labels (expect 0.5) ===")
    for group, datasets in DEFAULT_OOD_GROUPS.items():
        values = shuffled_control(frame, datasets, id_mask, seed)
        print(f"  {group:<9} min {min(values):.4f}  median "
              f"{float(np.median(values)):.4f}  max {max(values):.4f}")

    print(f"\n=== matched-size null, read as a range and not as a test ===")
    for group, datasets in DEFAULT_OOD_GROUPS.items():
        per_control = {
            c: group_auroc(frame, c, datasets, id_mask) for c in controls
        }
        obs_avg, obs_pooled, per_dataset = group_auroc(
            frame, OBSERVED, datasets, id_mask
        )
        for label, index, observed in (
            ("dataset-averaged (HEADLINE, plan 1.7)", 0, obs_avg),
            ("pooled (reported beside, never promoted)", 1, obs_pooled),
        ):
            draws = [per_control[c][index] for c in controls]
            ordered = sorted(draws)
            fired = reading_for(observed, draws)
            print(f"\n  [{group}] {label}")
            print(f"    twenty draws : min {ordered[0]:.4f}  "
                  f"median {float(np.median(ordered)):.4f}  max {ordered[-1]:.4f}"
                  f"  spread {ordered[-1] - ordered[0]:.4f}")
            print(f"    observed {OBSERVED} : {observed:.4f}")
            print(f"    margin over the highest draw : {observed - ordered[-1]:+.4f}")
            print(f"    READING {fired}: {READINGS[fired]}")
        print(f"    per-dataset observed : "
              f"{ {k: round(v, 4) for k, v in per_dataset.items()} }")
        print(f"    every draw (dataset-averaged) : "
              f"{[round(per_control[c][0], 4) for c in controls]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=20260909,
        help="seed for the shuffled-label control only; the twenty draws are "
             "fixed in the dump and nothing here reseeds them",
    )
    options = parser.parse_args(argv)
    return report(options.run_dir, options.shuffle_seed)


if __name__ == "__main__":
    raise SystemExit(main())
