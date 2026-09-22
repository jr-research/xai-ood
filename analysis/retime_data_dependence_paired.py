"""The data-dependence control, re-run paired, because the first one repeated the defect.

WHAT WENT WRONG THE FIRST TIME. ``retime_scoring_cost.py``'s data-dependence
control scored two rounds of all thirteen scorers on a `cifar10_test` block and
then two rounds on an `svhn` block. **Every one of the thirteen came out slower
on the second block**, by 1.04x to 2.63x. A property of the data would not move
all thirteen the same way; a machine that got busier between the two phases
moves all thirteen the same way. So that control cannot separate what it exists
to separate, and it failed for the exact reason the unreplicated timing does:
**two arms run one after the other attribute drift to whichever arm ran second.**

WHAT THIS DOES INSTEAD. The two blocks are timed **adjacently, inside one
scorer's turn**, and the ratio is formed from that pair before anything is
aggregated. Drift slower than one pair of calls cancels in the ratio. Which of
the two blocks goes first alternates by round, so any residual first-or-second
advantage cancels across rounds too.

THE QUESTION, unchanged: per-sample cost is claimed everywhere in this session
to be a function of row count and dimension and not of the values. These are
dense linear algebra with no data-dependent branching, so the claim should hold
exactly; the mechanism that could break it is subnormal floating point, which is
value-dependent and slow on x86. Reported as a median paired ratio with its
range, per scorer.

Equal row counts, 2,002 each, the same size the pinned arm used.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import sys
import time
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

SPLITS = ("cifar10_test", "svhn")
BLOCK_ROWS = 2_002
WARMUP_ROUNDS = 1
KEPT_ROUNDS = 4
SEED = 20260915


def block_from(split: str) -> np.ndarray:
    array = np.load(CACHE / split / "cls.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(array.shape[0], size=BLOCK_ROWS, replace=False))
    return np.asarray(array[pick], dtype=np.float64)


def main() -> int:
    from run_phase2 import TRAIN_CACHE_SPLIT, fit_scorers, load_split

    metadata = json.loads((RUN / "scores" / "run_metadata.json").read_text())
    published = {
        name: float(entry["microseconds_per_sample"])
        for name, entry in metadata["timing"]["score"].items()
        if not name.startswith("control_")
    }
    order = sorted(published, key=published.get)

    print(f"[machine] {platform.node()}, numpy {np.__version__}")
    train = load_split(CACHE, TRAIN_CACHE_SPLIT, "cls")
    scorers, _, _ = fit_scorers(train.embeddings, train.labels, shrinkage=True)
    del train

    blocks = {split: block_from(split) for split in SPLITS}
    subnormals = {
        split: int((np.abs(b[b != 0]) < np.finfo(np.float64).tiny).sum())
        for split, b in blocks.items()
    }
    print(f"[blocks] {BLOCK_ROWS} rows each, subnormal inputs {subnormals}")

    rows = []
    for round_index in range(WARMUP_ROUNDS + KEPT_ROUNDS):
        first, second = SPLITS if round_index % 2 == 0 else SPLITS[::-1]
        for name in order:
            timings = {}
            for split in (first, second):
                start = time.perf_counter()
                values = scorers[name].score(blocks[split])
                timings[split] = time.perf_counter() - start
                if not np.isfinite(values).all():
                    sys.exit(f"REFUSING TO REPORT: {name} non-finite on {split}")
            rows.append(
                {
                    "round": round_index,
                    "phase": "warmup" if round_index < WARMUP_ROUNDS else "kept",
                    "scorer": name,
                    "first_block": first,
                    "cifar10_test_microseconds_per_sample":
                        1e6 * timings["cifar10_test"] / BLOCK_ROWS,
                    "svhn_microseconds_per_sample":
                        1e6 * timings["svhn"] / BLOCK_ROWS,
                    "paired_ratio_svhn_over_cifar10_test":
                        timings["svhn"] / timings["cifar10_test"],
                }
            )
        print(f"[round] {round_index + 1} of {WARMUP_ROUNDS + KEPT_ROUNDS}", flush=True)

    kept = [r for r in rows if r["phase"] == "kept"]
    table = []
    for name in order:
        ratios = sorted(
            r["paired_ratio_svhn_over_cifar10_test"] for r in kept if r["scorer"] == name
        )
        table.append(
            {
                "scorer": name,
                "published_microseconds_per_sample": published[name],
                "paired_ratio_median": statistics.median(ratios),
                "paired_ratio_min": ratios[0],
                "paired_ratio_max": ratios[-1],
                "pairs": len(ratios),
            }
        )
    path = OUT / "retimed-data-dependence-paired.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"[write] {path.name}: {len(table)} row(s)")

    worst = max(table, key=lambda r: abs(r["paired_ratio_median"] - 1.0))
    print(
        f"[verdict] worst median paired ratio {worst['paired_ratio_median']:.4f} "
        f"on {worst['scorer']}, range "
        f"[{worst['paired_ratio_min']:.4f}, {worst['paired_ratio_max']:.4f}]"
    )
    # The unpaired figures are re-derived from the CSV the first control wrote
    # rather than transcribed, so this comparison cannot go stale against it.
    unpaired_path = OUT / "retimed-scoring-cost-data-dependence.csv"
    if unpaired_path.exists():
        raw = list(csv.DictReader(unpaired_path.open()))
        for row in table:
            name = row["scorer"]
            before = {
                split: statistics.median(
                    float(r["wall_microseconds_per_sample"])
                    for r in raw
                    if r["scorer"] == name and r["split"] == split
                )
                for split in SPLITS
            }
            row["unpaired_ratio"] = before["svhn"] / before["cifar10_test"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        worst_unpaired = max(table, key=lambda r: abs(r["unpaired_ratio"] - 1.0))
        print(
            f"[against the unpaired control] worst unpaired ratio "
            f"{worst_unpaired['unpaired_ratio']:.4f} on {worst_unpaired['scorer']}, "
            f"paired {worst_unpaired['paired_ratio_median']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
