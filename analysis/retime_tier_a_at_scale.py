"""The diagnosis after the pin fired: is the inversion block size, or the ordering?

WHY THIS EXISTS. ``retime_scoring_cost.py`` ran the pinned protocol and R3 of
``retiming-pin-2026-09-15.md`` was violated: three published gaps of 1.5x to
1.8x inverted, all three inside the cheap tier, all three involving a PCA
residual scorer against a marginal Gaussian cell. Per the pin note that stops
the re-timing from being reported as a re-timing. It does not stop a diagnosis,
and there are exactly two candidate explanations to separate.

1. BLOCK SIZE. The published figure is one call over 312,660 rows. The pinned
   protocol used 2,002 rows per call, and the block-size sweep in the same run
   shows per-sample cost is NOT constant in block size and does not even move in
   the same direction for every method: from 2,000 to 8,000 rows the Gaussian
   cells get cheaper per sample and both kNN columns get dearer. If that is the
   cause, the pinned arm measured a different quantity from the published one and
   the inversion says nothing about the published ordering.
2. THE ORDERING. If the cheap tier reorders at a block size close to the
   published call as well, then the published within-tier ordering does not
   reproduce on a second machine, and the Results chapter's cost column is
   ordered by differences it cannot resolve.

THE INTERPRETATION RULE, FIXED BEFORE THIS RAN, so that whichever way it comes
out is a finding and not a choice:

  - The same protocol as the pinned arm, three warm-up rounds discarded and
    fifteen kept, cyclic interleaving, median and order statistics.
  - The seven cheap-tier scorers only. The kNN columns and the four
    class-conditional cells are excluded because R1 and R2 held for them and
    they are not what inverted; excluding them is what buys the 25x larger block
    inside the same time budget.
  - 50,000 rows per call, 25x the pinned arm and the largest block that fits
    beside the fitted scorers on this machine at float64.
  - IF no pinned pair inverts at 50,000 rows, the diagnosis is BLOCK SIZE: the
    published per-sample figure is not comparable to a small-block re-timing, and
    the pinned arm's inversions are an artefact of the block choice.
  - IF the same pairs invert at 50,000 rows, the diagnosis is THE ORDERING: the
    published cheap-tier ordering does not reproduce under repeats on a second
    machine, and no sentence may order those seven cells by cost.
  - IF some invert and some do not, the diagnosis is neither, and the finding is
    that the cheap tier's ordering is block-size dependent, which is itself a
    reason not to order it.

This writes one CSV and prints the verdict. It does not decide anything about
tiers B and C, about the fit timings, or about frontier membership, which is
a separate open item and untouched.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# The repository root, the directory above this one. Overridable with
# XAI_OOD_ROOT so a run tree held outside the checkout can be read in place.
ROOT = Path(os.environ.get("XAI_OOD_ROOT", Path(__file__).resolve().parents[1]))
RUN = ROOT / "results" / "runs" / "2026-09-10-full-spectrum-shrunk"
CACHE = ROOT / "embeddings-local"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TIER_A = (
    "pca_residual_class_mean",
    "pca_residual_class_mean_l2",
    "marginal_diagonal",
    "marginal_full",
    "pca_residual_class_mean_whitened",
    "pca_residual_all_id",
    "pca_residual_all_id_l2",
)
BLOCK_ROWS = 50_000
WARMUP_ROUNDS = 3
KEPT_ROUNDS = 15
PINNED_RATIO = 1.5
SEED = 20260915


def main() -> int:
    from run_phase2 import TRAIN_CACHE_SPLIT, fit_scorers, load_split

    metadata = json.loads((RUN / "scores" / "run_metadata.json").read_text())
    published = {
        name: float(metadata["timing"]["score"][name]["microseconds_per_sample"])
        for name in TIER_A
    }
    order = sorted(TIER_A, key=published.get)

    print(
        f"[machine] {platform.node()}, python {platform.python_version()}, "
        f"numpy {np.__version__}"
    )
    train = load_split(CACHE, TRAIN_CACHE_SPLIT, "cls")
    scorers, _, _ = fit_scorers(train.embeddings, train.labels, shrinkage=True)
    del train

    # The block is drawn from one split rather than stratified across eight,
    # because 50,000 rows is more than six of the eight splits hold. cs-ID is
    # the only split large enough to draw 50,000 rows without replacement, and
    # the data-dependence control in the pinned arm is what licenses that: cost
    # did not depend on which split the rows came from.
    array = np.load(CACHE / "csid" / "cls.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(array.shape[0], size=BLOCK_ROWS, replace=False))
    block = np.asarray(array[pick], dtype=np.float64)
    del array
    print(f"[block] {block.shape[0]} rows x {block.shape[1]} dims from csid, seed {SEED}")

    rows = []
    for round_index in range(WARMUP_ROUNDS + KEPT_ROUNDS):
        rotated = [order[(i + round_index) % len(order)] for i in range(len(order))]
        for position, name in enumerate(rotated):
            start = time.perf_counter()
            values = scorers[name].score(block)
            wall = time.perf_counter() - start
            if values.shape[0] != block.shape[0] or not np.isfinite(values).all():
                sys.exit(f"REFUSING TO REPORT: {name} scored badly on the block")
            rows.append(
                {
                    "round": round_index,
                    "phase": "warmup" if round_index < WARMUP_ROUNDS else "kept",
                    "position": position,
                    "scorer": name,
                    "rows": int(block.shape[0]),
                    "wall_seconds": wall,
                    "wall_microseconds_per_sample": 1e6 * wall / block.shape[0],
                }
            )
        print(
            f"[round] {round_index + 1} of {WARMUP_ROUNDS + KEPT_ROUNDS}, "
            f"{sum(r['wall_seconds'] for r in rows[-len(order):]):.2f} s",
            flush=True,
        )

    kept = [r for r in rows if r["phase"] == "kept"]
    per_method = {}
    for name in order:
        values = sorted(r["wall_microseconds_per_sample"] for r in kept if r["scorer"] == name)
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        per_method[name] = {
            "median": statistics.median(values),
            "lo": values[3],
            "hi": values[11],
            "iqr": quartiles[2] - quartiles[0],
            "min": values[0],
            "max": values[-1],
        }

    median = {name: per_method[name]["median"] for name in order}
    inversions = []
    pinned = 0
    for a, b in combinations(order, 2):
        if published[b] / published[a] < PINNED_RATIO:
            continue
        pinned += 1
        if median[b] / median[a] <= 1.0:
            inversions.append(
                (a, b, published[b] / published[a], median[b] / median[a])
            )

    table = [
        {
            "scorer": name,
            "published_microseconds_per_sample": published[name],
            "published_rank_within_tier": order.index(name) + 1,
            "retimed_median": per_method[name]["median"],
            "retimed_rank_within_tier": sorted(median, key=median.get).index(name) + 1,
            "median_interval_lo": per_method[name]["lo"],
            "median_interval_hi": per_method[name]["hi"],
            "iqr": per_method[name]["iqr"],
            "relative_iqr_percent": 100 * per_method[name]["iqr"] / per_method[name]["median"],
            "min": per_method[name]["min"],
            "max": per_method[name]["max"],
            "rows_per_call": BLOCK_ROWS,
        }
        for name in order
    ]
    path = OUT / "retimed-cheap-tier-at-50k-rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"[write] {path.name}: {len(table)} row(s)")

    print(f"[pinned] {pinned} pair(s) at or above {PINNED_RATIO}x published ratio")
    if not pinned:
        sys.exit("REFUSING TO REPORT: no pinned pair inside the tier, so nothing was tested")
    for a, b, want, got in inversions:
        print(f"[inverted] published {b}/{a} = {want:.3f}, re-timed {got:.3f}")
    if not inversions:
        print("[verdict] BLOCK SIZE: no pinned pair inverts at 50,000 rows per call")
    elif len(inversions) == pinned:
        print("[verdict] THE ORDERING: every pinned pair inverts at 50,000 rows per call")
    else:
        print(
            f"[verdict] NEITHER: {len(inversions)} of {pinned} pinned pairs invert, so the "
            f"cheap-tier ordering is block-size dependent"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
