"""Re-time per-method scoring with repeats, warm-up discard and interleaving.

THE DEFECT THIS EXISTS FOR is the unreplicated published timing. The per-method
scoring cost is the ``timing.score`` block of the run manifest: ONE wall-clock
figure per scorer, thirteen scorers run sequentially to completion in one
process, one machine, no repetition, no warm-up discard, no interleaving and
therefore no interval of any kind. Machine drift over that run is attributed to
whichever scorer happened to be running at the time, and the ordering effect is
unmeasured.

WHAT THIS SCRIPT CHANGES, AND ONLY THIS. Four things, each aimed at one clause
of that sentence:

1. REPEATS. Eighteen rounds, the first three discarded, fifteen kept. Fifteen
   because the order statistics x(4) and x(12) of fifteen give a
   distribution-free 96.5 per cent interval for the median whose endpoints are
   not extreme values, so one scheduling stall cannot become an endpoint.
2. WARM-UP DISCARDED EXPLICITLY, and the discarded rounds are written to the
   per-round CSV rather than dropped, so the reader can see what was thrown away
   and how different it was.
3. INTERLEAVING, cyclic. In round r the thirteen scorers are called in the order
   rotated by r, so over any thirteen consecutive rounds each scorer occupies
   each position exactly once. Drift is then spread across all thirteen instead
   of loading onto whichever ran last.
4. MEDIAN AND SPREAD, never a mean. A scheduling stall is a one-sided
   contaminant: it inflates a mean and is indistinguishable from real cost. The
   mean is computed and written anyway, in its own column, because the gap
   between it and the median is the evidence for that sentence.

THE ORDERING IS PINNED, and the pin was written and committed before this ran.
See ``retiming-pin-2026-09-15.md`` beside this file. R1 to R3 there are stop
conditions: if a published gap of 1.5x or more inverts, or a cost tier crosses,
this script exits non-zero, and what is reported is a finding about the harness
rather than a re-timing.

A DIFFERENT MACHINE, WHICH IS A SCOPE LIMIT AND ALSO A SECOND PATH. The
published run was made on the lab workstation at Python 3.10.20. This runs
wherever it is invoked, and records that machine in its own output. Absolute microseconds are therefore NOT
expected to reproduce and nothing here requires them to; what is pinned is the
ordering, which a second machine tests more severely than a repeat on the first.

THREE CONTROLS, because every number above rests on an assumption:

- THE SEQUENTIAL ARM. Three independent passes that imitate the published
  harness exactly: each scorer once, in published order, no interleaving, n = 1.
  This is what says whether a single unreplicated figure lands inside the
  interval the repeats measure, which is the question entry 38 actually asks.
- THE POSITION EFFECT. Cost by position within the round, averaged over kept
  rounds. If being scored late in a round costs more, the published harness was
  measuring order and calling it method.
- DATA DEPENDENCE. Equal-sized blocks drawn from two different splits. Every
  per-sample claim here assumes cost is a function of row count and dimension
  and not of the values. Measured, not asserted.

NO RE-SCORING, NO GPU, NO WRITES OUTSIDE THIS DIRECTORY. The fitted scorers are
rebuilt from the cached ``cls`` embeddings, and the rebuild is checked against
the covariance diagnostics in the published manifest before anything is timed,
so what gets timed is demonstrably the same fitted arm. No score dump is
produced and no published number is recomputed.

Usage:

    python3 retime_scoring_cost.py                 # the full protocol
    python3 retime_scoring_cost.py --calibrate     # one cheap round, sets block
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

# The repository root, the directory above this one. Overridable with
# XAI_OOD_ROOT so a run tree held outside the checkout can be read in place.
ROOT = Path(os.environ.get("XAI_OOD_ROOT", Path(__file__).resolve().parents[1]))
RUN = ROOT / "results" / "runs" / "2026-09-10-full-spectrum-shrunk"
CACHE = ROOT / "embeddings-local"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

#: The eight evaluation splits the published run scored, with the row counts the
#: timing block averaged over. The stratified timing block is drawn in these
#: proportions so it is a subsample of the same population, not a convenience
#: slice of one split.
EVAL_SPLITS = (
    "cifar10_test",
    "csid",
    "cifar100",
    "tin",
    "mnist",
    "svhn",
    "texture",
    "places365",
)

#: Tiers as fixed in the pin note, by published cost. R1 checks these do not cross.
TIERS = {
    "A": (
        "pca_residual_class_mean",
        "pca_residual_class_mean_l2",
        "marginal_diagonal",
        "marginal_full",
        "pca_residual_class_mean_whitened",
        "pca_residual_all_id",
        "pca_residual_all_id_l2",
    ),
    "B": ("class_conditional_diagonal", "class_conditional_full", "rmd", "rmd_pp"),
    "C": ("knn_unnormalized", "knn_normalized"),
}

#: R2's named pair set: the marginal Gaussian cells against the kNN columns, the
#: two orders of magnitude entry 38's argument rests on.
MARGINAL = ("marginal_diagonal", "marginal_full")

#: R3's threshold. A published ratio at or above this is a gap the single
#: measurement resolved and the re-timing must reproduce its sign.
PINNED_RATIO = 1.5

#: R2's threshold.
ORDERS_OF_MAGNITUDE = 50.0

WARMUP_ROUNDS = 3
KEPT_ROUNDS = 15
BLOCK_ROWS = 2_000
SEED = 20260915
SWEEP_BLOCKS = (500, 2_000, 8_000)
SWEEP_ROUNDS = 2
SEQUENTIAL_PASSES = 3


def fail(message: str) -> None:
    sys.exit(f"REFUSING TO REPORT: {message}")


# --------------------------------------------------------------------------- #
# The fitted arm, and the guard that it is the published one
# --------------------------------------------------------------------------- #


def published_timing() -> dict[str, float]:
    """The ``timing.score`` figures for the thirteen scorers, read from the manifest.

    Read rather than transcribed. A copy of these numbers inside this file could
    not learn that the manifest had changed, which is the whole failure mode a
    pinned comparison is supposed to rule out.
    """
    metadata = json.loads((RUN / "scores" / "run_metadata.json").read_text())
    score = metadata["timing"]["score"]
    scorers = sorted(metadata["scorers"])
    out = {name: float(score[name]["microseconds_per_sample"]) for name in scorers}
    if len(out) != 13:
        fail(f"expected thirteen scorers in the manifest, found {len(out)}")
    return out


def published_machine() -> dict[str, Any]:
    metadata = json.loads((RUN / "scores" / "run_metadata.json").read_text())
    return metadata["timing"]["machine"]


def fit_the_published_arm() -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the thirteen shrunk-arm scorers and check them against the manifest.

    The check is the reason this is a function rather than three lines: a timing
    on a differently-fitted object is a timing of something the thesis does not
    report. Every covariance's condition number and selected shrinkage intensity
    must reproduce the manifest's before anything is timed.
    """
    from run_phase2 import TRAIN_CACHE_SPLIT, fit_scorers, load_split  # noqa: PLC0415

    train = load_split(CACHE, TRAIN_CACHE_SPLIT, "cls")
    scorers, _, _ = fit_scorers(train.embeddings, train.labels, shrinkage=True)
    metadata = json.loads((RUN / "scores" / "run_metadata.json").read_text())
    published = metadata["scorers"]

    checked = 0
    for name, entry in published.items():
        if name not in scorers:
            fail(f"{name} is in the manifest and was not rebuilt")
        for fit_name, diagnostics in entry.get("covariances", {}).items():
            scorer = scorers[name]
            fits = {"primary": scorer.primary}
            if getattr(scorer, "background", None) is not None:
                fits["background"] = scorer.background
            if fit_name not in fits:
                fail(f"{name}: manifest names covariance {fit_name}, rebuild has {sorted(fits)}")
            got = fits[fit_name].diagnostics
            if got["name"] != diagnostics["name"]:
                fail(
                    f"{name}: manifest calls this covariance {diagnostics['name']!r} and the "
                    f"rebuild calls it {got['name']!r}, so the slots do not correspond"
                )
            for key in ("condition_number", "shrinkage_intensity", "trace", "n_samples"):
                want = float(diagnostics[key])
                have = float(got[key])
                tolerance = 1e-9 * max(abs(want), 1.0)
                if abs(want - have) > tolerance:
                    fail(
                        f"{name}:{fit_name}: {key} rebuilt as {have!r}, manifest says "
                        f"{want!r}. The arm being timed is not the arm that was published."
                    )
                checked += 1
    if checked == 0:
        fail("the fitted-arm guard compared nothing, so a clean result means nothing")
    print(f"[guard] {checked} covariance diagnostic(s) reproduce the published manifest")
    return scorers, published


# --------------------------------------------------------------------------- #
# The timing block
# --------------------------------------------------------------------------- #


def stratified_block(rows: int, seed: int) -> np.ndarray:
    """``rows`` embedding rows drawn across the eight evaluation splits in proportion.

    Loaded one split at a time and subsampled before the next is opened, because
    the whole evaluation matrix in float64 is 1.9 GB and this machine does not
    have it to spare beside the fitted scorers.
    """
    counts = {
        name: int(json.loads((CACHE / name / "manifest.json").read_text())["row_count"])
        for name in EVAL_SPLITS
    }
    total = sum(counts.values())
    rng = np.random.default_rng(seed)
    blocks = []
    taken = {}
    for name in EVAL_SPLITS:
        want = int(round(rows * counts[name] / total))
        if want == 0:
            continue
        array = np.load(CACHE / name / "cls.npy", mmap_mode="r")
        if array.shape[0] != counts[name]:
            fail(f"{name}: cls.npy has {array.shape[0]} rows, manifest says {counts[name]}")
        pick = np.sort(rng.choice(array.shape[0], size=min(want, array.shape[0]), replace=False))
        blocks.append(np.asarray(array[pick], dtype=np.float64))
        taken[name] = int(len(pick))
        del array
    block = np.concatenate(blocks, axis=0)
    print(
        f"[block] {block.shape[0]} rows x {block.shape[1]} dims, "
        f"stratified over {len(taken)} splits, seed {seed}, "
        f"population {total} rows"
    )
    return block


def single_split_block(name: str, rows: int, seed: int) -> np.ndarray:
    """Equal-sized block from one split, for the data-dependence control."""
    array = np.load(CACHE / name / "cls.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(array.shape[0], size=min(rows, array.shape[0]), replace=False))
    return np.asarray(array[pick], dtype=np.float64)


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #


def time_one_call(scorer: Any, block: np.ndarray) -> tuple[float, float]:
    """Wall clock and process CPU time for one ``score`` call over one block.

    Both, because their difference is the part of a wall-clock figure that was
    not this process working, which is precisely the contamination a single
    unreplicated measurement cannot see.
    """
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    values = scorer.score(block)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    if values.shape[0] != block.shape[0]:
        fail(f"{scorer.name}: scored {values.shape[0]} of {block.shape[0]} rows")
    if not np.isfinite(values).all():
        fail(f"{scorer.name}: non-finite score on the timing block")
    return wall, cpu


def interleaved_rounds(
    scorers: dict[str, Any],
    order: list[str],
    block: np.ndarray,
    *,
    warmup: int,
    kept: int,
    origin: float,
) -> list[dict[str, Any]]:
    """Cyclically interleaved rounds. Returns one row per call, warm-up included."""
    records: list[dict[str, Any]] = []
    n = len(order)
    for round_index in range(warmup + kept):
        rotated = [order[(i + round_index) % n] for i in range(n)]
        for position, name in enumerate(rotated):
            wall, cpu = time_one_call(scorers[name], block)
            records.append(
                {
                    "round": round_index,
                    "phase": "warmup" if round_index < warmup else "kept",
                    "position": position,
                    "scorer": name,
                    "rows": int(block.shape[0]),
                    "wall_seconds": wall,
                    "cpu_seconds": cpu,
                    "wall_microseconds_per_sample": 1e6 * wall / block.shape[0],
                    "elapsed_at_start_seconds": time.perf_counter() - origin - wall,
                }
            )
        done = round_index + 1
        print(
            f"[round] {done:>2} of {warmup + kept} "
            f"({'warm-up' if round_index < warmup else 'kept'}), "
            f"{sum(r['wall_seconds'] for r in records[-n:]):.2f} s for the round",
            flush=True,
        )
    return records


def sequential_passes(
    scorers: dict[str, Any],
    order: list[str],
    block: np.ndarray,
    *,
    passes: int,
    origin: float,
) -> list[dict[str, Any]]:
    """The published harness, imitated: each scorer once, in published order, n = 1."""
    records: list[dict[str, Any]] = []
    for pass_index in range(passes):
        for position, name in enumerate(order):
            wall, cpu = time_one_call(scorers[name], block)
            records.append(
                {
                    "pass": pass_index,
                    "position": position,
                    "scorer": name,
                    "rows": int(block.shape[0]),
                    "wall_seconds": wall,
                    "cpu_seconds": cpu,
                    "wall_microseconds_per_sample": 1e6 * wall / block.shape[0],
                    "elapsed_at_start_seconds": time.perf_counter() - origin - wall,
                }
            )
        print(f"[sequential] pass {pass_index + 1} of {passes} done", flush=True)
    return records


def summarise(values: list[float]) -> dict[str, float]:
    """Median first, and the order-statistic interval the pin note fixed."""
    ordered = sorted(values)
    n = len(ordered)
    if n < 2:
        fail(f"summarise got {n} value(s); a spread needs repeats, which is the point")
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    summary = {
        "n": n,
        "median": statistics.median(ordered),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "iqr": quartiles[2] - quartiles[0],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }
    if n == 15:
        summary["median_lo_x4"] = ordered[3]
        summary["median_hi_x12"] = ordered[11]
    else:
        summary["median_lo_x4"] = ordered[0]
        summary["median_hi_x12"] = ordered[-1]
    summary["relative_iqr_percent"] = 100.0 * summary["iqr"] / summary["median"]
    return summary


# --------------------------------------------------------------------------- #
# The pin
# --------------------------------------------------------------------------- #


def check_the_pin(
    published: dict[str, float], median: dict[str, float]
) -> tuple[list[str], list[dict[str, Any]]]:
    """R1 to R3 from the pin note. Returns violations and the pair table."""
    violations: list[str] = []

    for lower, upper in (("A", "B"), ("B", "C")):
        top = max(median[name] for name in TIERS[lower])
        bottom = min(median[name] for name in TIERS[upper])
        if not top < bottom:
            violations.append(
                f"R1: tier {lower} tops out at {top:.4f} us/sample and tier {upper} "
                f"starts at {bottom:.4f}, so the tiers cross"
            )

    for marginal in MARGINAL:
        for knn in TIERS["C"]:
            ratio = median[knn] / median[marginal]
            if ratio < ORDERS_OF_MAGNITUDE:
                violations.append(
                    f"R2: {knn} is only {ratio:.2f}x {marginal}, under the "
                    f"{ORDERS_OF_MAGNITUDE:.0f}x the two-orders-of-magnitude argument needs"
                )

    pairs: list[dict[str, Any]] = []
    for a, b in combinations(sorted(published, key=published.get), 2):
        published_ratio = published[b] / published[a]
        retimed_ratio = median[b] / median[a]
        pinned = published_ratio >= PINNED_RATIO
        reproduced = retimed_ratio > 1.0
        pairs.append(
            {
                "cheaper_published": a,
                "dearer_published": b,
                "published_ratio": published_ratio,
                "retimed_ratio": retimed_ratio,
                "pinned_by_r3": pinned,
                "order_reproduced": reproduced,
            }
        )
        if pinned and not reproduced:
            violations.append(
                f"R3: published {b} / {a} = {published_ratio:.3f} and the re-timed "
                f"medians give {retimed_ratio:.3f}, so a pinned gap inverted"
            )
    return violations, pairs


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, without scipy, for R5's reported figure."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, index in enumerate(order, start=1):
            out[index] = float(rank)
        return out

    x, y = ranks(a), ranks(b)
    n = len(x)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
    return num / den


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fail(f"{name}: nothing to write")
    path = OUT / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path.name}: {len(rows)} row(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--block", type=int, default=BLOCK_ROWS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ROUNDS)
    parser.add_argument("--kept", type=int, default=KEPT_ROUNDS)
    parser.add_argument("--no-sweep", action="store_true")
    options = parser.parse_args()

    if options.calibrate:
        options.block, options.warmup, options.kept = 1_000, 1, 1
        options.no_sweep = True

    print(
        f"[machine] {platform.node()}, {os.cpu_count()} cpu(s), "
        f"python {platform.python_version()}, numpy {np.__version__}, "
        f"loadavg {os.getloadavg()[0]:.2f}"
    )
    print(f"[machine] published on {published_machine()}")

    published = published_timing()
    order = sorted(published, key=published.get)
    scorers, _ = fit_the_published_arm()
    block = stratified_block(options.block, SEED)

    origin = time.perf_counter()
    calls = interleaved_rounds(
        scorers,
        order,
        block,
        warmup=options.warmup,
        kept=options.kept,
        origin=origin,
    )
    sequential = sequential_passes(
        scorers, order, block, passes=1 if options.calibrate else SEQUENTIAL_PASSES,
        origin=origin,
    )

    kept = [row for row in calls if row["phase"] == "kept"]
    per_method = {
        name: summarise([r["wall_microseconds_per_sample"] for r in kept if r["scorer"] == name])
        for name in order
    }
    warm = {
        name: [r["wall_microseconds_per_sample"] for r in calls
               if r["scorer"] == name and r["phase"] == "warmup"]
        for name in order
    }
    median = {name: per_method[name]["median"] for name in order}

    violations, pairs = check_the_pin(published, median)
    rho = spearman([published[n] for n in order], [median[n] for n in order])
    print(f"[pin] Spearman rank correlation against the published figures: {rho:.6f}")

    table = []
    for name in order:
        summary = per_method[name]
        sequential_here = [r["wall_microseconds_per_sample"] for r in sequential
                           if r["scorer"] == name]
        inside = sum(
            1 for value in sequential_here
            if summary["median_lo_x4"] <= value <= summary["median_hi_x12"]
        )
        table.append(
            {
                "scorer": name,
                "published_microseconds_per_sample": published[name],
                "published_rank": order.index(name) + 1,
                "retimed_median": summary["median"],
                "retimed_rank": sorted(median, key=median.get).index(name) + 1,
                "median_interval_lo": summary["median_lo_x4"],
                "median_interval_hi": summary["median_hi_x12"],
                "iqr": summary["iqr"],
                "relative_iqr_percent": summary["relative_iqr_percent"],
                "min": summary["min"],
                "max": summary["max"],
                "mean": summary["mean"],
                "mean_over_median": summary["mean"] / summary["median"],
                "kept_rounds": summary["n"],
                "warmup_first_round": warm[name][0] if warm[name] else "",
                "warmup_over_median": (warm[name][0] / summary["median"]) if warm[name] else "",
                "sequential_n1_median": statistics.median(sequential_here),
                "sequential_passes_inside_median_interval": inside,
                "sequential_passes": len(sequential_here),
                "retimed_over_published": summary["median"] / published[name],
                "rows_per_call": int(block.shape[0]),
            }
        )

    position = [
        {
            "position": p,
            "calls": len([r for r in kept if r["position"] == p]),
            "median_wall_seconds": statistics.median(
                [r["wall_seconds"] for r in kept if r["position"] == p]
            ),
            "median_relative_to_its_scorer": statistics.median(
                [r["wall_microseconds_per_sample"] / median[r["scorer"]]
                 for r in kept if r["position"] == p]
            ),
        }
        for p in range(len(order))
    ]

    round_totals = [
        {
            "round": r,
            "phase": "warmup" if r < options.warmup else "kept",
            "wall_seconds": sum(row["wall_seconds"] for row in calls if row["round"] == r),
            "cpu_seconds": sum(row["cpu_seconds"] for row in calls if row["round"] == r),
        }
        for r in range(options.warmup + options.kept)
    ]

    write_csv("retimed-scoring-cost-per-method.csv", table)
    write_csv("retimed-scoring-cost-calls.csv", calls)
    write_csv("retimed-scoring-cost-sequential-arm.csv", sequential)
    write_csv("retimed-scoring-cost-pairs.csv", pairs)
    write_csv("retimed-scoring-cost-position-effect.csv", position)
    write_csv("retimed-scoring-cost-round-totals.csv", round_totals)

    if not options.no_sweep:
        sweep = []
        for rows in SWEEP_BLOCKS:
            sweep_block = stratified_block(rows, SEED + rows)
            for round_index in range(SWEEP_ROUNDS):
                rotated = [order[(i + round_index) % len(order)] for i in range(len(order))]
                for name in rotated:
                    wall, cpu = time_one_call(scorers[name], sweep_block)
                    sweep.append(
                        {
                            "block_rows": rows,
                            "round": round_index,
                            "scorer": name,
                            "wall_seconds": wall,
                            "wall_microseconds_per_sample": 1e6 * wall / rows,
                        }
                    )
            del sweep_block
            print(f"[sweep] block {rows} done", flush=True)
        write_csv("retimed-scoring-cost-block-sweep.csv", sweep)

        control = []
        for split in ("cifar10_test", "svhn"):
            control_block = single_split_block(split, BLOCK_ROWS, SEED)
            for round_index in range(SWEEP_ROUNDS):
                for name in order:
                    wall, _ = time_one_call(scorers[name], control_block)
                    control.append(
                        {
                            "split": split,
                            "round": round_index,
                            "scorer": name,
                            "rows": int(control_block.shape[0]),
                            "wall_microseconds_per_sample": 1e6 * wall / control_block.shape[0],
                        }
                    )
            del control_block
            print(f"[control] {split} done", flush=True)
        write_csv("retimed-scoring-cost-data-dependence.csv", control)

    summary_json = {
        "machine": {
            "node": platform.node(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "loadavg_at_end": os.getloadavg(),
        },
        "published_machine": published_machine(),
        "protocol": {
            "warmup_rounds": options.warmup,
            "kept_rounds": options.kept,
            "rows_per_call": int(block.shape[0]),
            "interleaving": "cyclic rotation by round index",
            "seed": SEED,
            "sequential_passes": len({r["pass"] for r in sequential}),
        },
        "spearman_against_published": rho,
        "pin_violations": violations,
        "elapsed_seconds": time.perf_counter() - origin,
    }
    (OUT / "retimed-scoring-cost-summary.json").write_text(
        json.dumps(summary_json, indent=1, sort_keys=True) + "\n"
    )
    print(f"[write] retimed-scoring-cost-summary.json")

    if violations:
        print("\n".join(f"[PIN VIOLATED] {v}" for v in violations))
        print(
            "This is a finding about the harness, not a re-timing. The pin note "
            "says stop, so this exits non-zero and reports nothing as a result."
        )
        return 1
    print("[pin] R1, R2 and R3 all hold: the published ordering reproduces where it was pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
