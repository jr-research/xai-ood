"""Per-dataset marginals for all thirteen scorers, both arms, both protocols, both metrics.

Why this exists. Every table in this directory is dataset-averaged, and a group figure
is not reported without its per-dataset companion, because far-OOD rows are 51 per cent
MNIST and one member can carry a pooled group number on its own. No per-dataset table of record existed anywhere, so the
Results chapter had nothing to put underneath its group numbers. This script is that
table, plus the three derived views the within-group question needs: how far apart the
members of a group are for each method, how far each dataset sits from the rest of its
own group, and whether any dataset is an outlier for every method rather than for some.

It also emits the shrinkage ablation per cell, because the paired construction is the
same one and the two arms are already open.

Three pins, and they are deliberately different paths:

  1. The group averages of the per-dataset points rebuilt here reproduce the gated
     table, the same check derive_contrasts.py makes.
  2. Every per-dataset point in both pickles is recomputed straight off the score
     dumps by an AUROC and an FPR@95 written here from the definitions, not imported
     from the package that produced the pickles. A pickle and the table it fed agree
     by construction; the dump is the only independent source. That transcription
     still shares an ALGORITHM with the package, so pin 2b recomputes the AUROCs a
     second time through the Mann-Whitney rank-sum identity, which does not.
  3. The unshrunk standard-protocol rows are cross-checked against the separate
     2026-09-09-standard dump, which has no bootstrap directory of its own. That is
     what retires the "the unshrunk standard arm needs a fresh bootstrap" problem:
     it does not, the full-spectrum unshrunk run carries the same protocol.

Cross-arm pairing is verified rather than assumed. The two runs share a seed and a
plan, so replicate i is the same resample in both, and the seven scorer columns that
shrinkage does not reach must therefore have element-wise identical replicate arrays.
If they do not, the paired shrinkage differences below are meaningless and the script
exits.

Read-only against the repository's results tree. Writes only into this directory.
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The repository root, the directory above this one. Overridable with
# XAI_OOD_ROOT so a run tree held outside the checkout can be read in place.
ROOT = Path(os.environ.get("XAI_OOD_ROOT", Path(__file__).resolve().parents[1]))
RUNS = ROOT / "results" / "runs"
GATED_STANDARD_TABLE = (
    ROOT / "results" / "tables" / "2026-09-09-standard-shrunk-dataset-averaged.csv"
)
OUT = Path(__file__).resolve().parent

ARM_RUNS = {
    "shrunk": RUNS / "2026-09-10-full-spectrum-shrunk",
    "unshrunk": RUNS / "2026-09-10-full-spectrum-unshrunk",
}
#: The standard-protocol unshrunk dump that has no bootstrap of its own. Used only
#: by pin 3, to show the full-spectrum unshrunk run reproduces it.
STANDARD_UNSHRUNK_DUMP = RUNS / "2026-09-09-standard" / "scores" / "per_sample_scores.parquet"

NEAR = ("cifar100", "tin")
FAR = ("mnist", "svhn", "texture", "places365")
GROUPS = {"near_ood": NEAR, "far_ood": FAR}
GROUP_OF = {d: g for g, ds in GROUPS.items() for d in ds}
DATASETS = NEAR + FAR

#: Protocol definition, copied from xai_ood.bootstrap.plan.PROTOCOL_ID_SPLITS so the
#: pin does not import the package whose output it is checking.
PROTOCOL_ID_SPLITS = {"standard": ("id_test",), "full_spectrum": ("id_test", "csid")}
METRICS = ("auroc", "fpr95")
TPR_TARGET = 0.95
PIN_TOLERANCE = 1e-9
#: Pin 2 and pin 3 compare a rank statistic recomputed from float64 scores against a
#: value computed by a different library build. Scores agree to ~1e-8 absolute, so
#: near-ties can move a single pair out of 9,000 x n_ood; one such flip is below this.
DUMP_PIN_TOLERANCE = 1e-6


class _Stub:
    """Stands in for xai_ood classes, which are not importable outside the repo."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("xai_ood"):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return _Unpickler(handle).load()


# --------------------------------------------------------------------------- #
# The two metrics, written from their definitions rather than imported
# --------------------------------------------------------------------------- #


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """P(score_OOD > score_ID) + 0.5 P(tie), higher = more OOD by package convention."""
    id_sorted = np.sort(np.asarray(id_scores, dtype=np.float64))
    ood = np.asarray(ood_scores, dtype=np.float64)
    below = np.searchsorted(id_sorted, ood, side="left")
    at_or_below = np.searchsorted(id_sorted, ood, side="right")
    wins = below.sum() + 0.5 * (at_or_below - below).sum()
    return float(wins / (id_sorted.size * ood.size))


def fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Fraction of OOD accepted as ID at the ceil(0.95 n_id)-th smallest ID score."""
    id_sorted = np.sort(np.asarray(id_scores, dtype=np.float64))
    rank = min(max(int(np.ceil(TPR_TARGET * id_sorted.size)), 1), id_sorted.size)
    tau = float(id_sorted[rank - 1])
    ood = np.asarray(ood_scores, dtype=np.float64)
    return float(np.count_nonzero(ood <= tau) / ood.size)


METRIC_FN = {"auroc": auroc, "fpr95": fpr95}


def interval(replicates: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(replicates, 2.5)), float(np.percentile(replicates, 97.5))


def point(obj, scorer, protocol, dataset, metric) -> float:
    return float(obj.point[(scorer, protocol, dataset, metric)]) * 100.0


def reps(obj, scorer, protocol, dataset, metric) -> np.ndarray:
    return np.asarray(obj.values[(scorer, protocol, dataset, metric)], dtype=float) * 100.0


def group_mean_reps(obj, scorer, protocol, group, metric) -> np.ndarray:
    """Dataset-averaged: average INSIDE each replicate, the declared estimand."""
    return np.mean([reps(obj, scorer, protocol, d, metric) for d in GROUPS[group]], axis=0)


# --------------------------------------------------------------------------- #
# Pins
# --------------------------------------------------------------------------- #


def deviation(got: float, want: float, where: str) -> float:
    """abs(got - want), refusing NaN rather than absorbing it.

    Every pin below accumulates with max(). `nan > 0.0` is False, so `max(0.0, nan)`
    is 0.0 and a NaN reaches the report as a PERFECT score. A critic pass on
    2026-09-14 planted three NaNs into 312 stored points and every pin printed its
    clean value byte for byte. This is the fix: a non-finite value on either side is
    a failure, not a zero.
    """
    if not (np.isfinite(got) and np.isfinite(want)):
        sys.exit(f"PIN FAILED: non-finite value at {where}: got {got!r}, want {want!r}")
    return abs(got - want)


def pin_group_averages_against_gated_table(obj) -> float:
    gated = {}
    with open(GATED_STANDARD_TABLE) as handle:
        for row in csv.DictReader(handle):
            gated[(row["scorer"], row["group"])] = float(row["auroc"])
    worst = 0.0
    for (scorer, group), want in gated.items():
        got = float(np.mean([point(obj, scorer, "standard", d, "auroc") for d in GROUPS[group]]))
        worst = max(worst, deviation(got, want, f"gated {scorer}/{group}"))
    print(f"PIN 1: {len(gated)} group averages rebuilt from per-dataset points, "
          f"max abs deviation vs the gated table {worst:.3e}")
    if worst > PIN_TOLERANCE:
        sys.exit("PIN 1 FAILED: the per-dataset points do not average to the gated table.")
    return worst


def pin_points_against_dump(obj, arm: str, frame: pd.DataFrame) -> float:
    """Recompute every per-dataset point straight off the score dump."""
    worst, checked = 0.0, 0
    split = frame["split"].to_numpy()
    for protocol, id_splits in PROTOCOL_ID_SPLITS.items():
        id_mask = np.isin(split, id_splits)
        for scorer in obj.scorers:
            column = frame[scorer].to_numpy(dtype=np.float64)
            id_scores = column[id_mask]
            for dataset in DATASETS:
                ood_scores = column[split == dataset]
                if ood_scores.size == 0:
                    sys.exit(f"PIN 2 FAILED: split {dataset!r} has no rows in the dump. "
                             f"An empty universe makes every metric on it vacuous.")
                for metric in METRICS:
                    got = METRIC_FN[metric](id_scores, ood_scores) * 100.0
                    want = point(obj, scorer, protocol, dataset, metric)
                    worst = max(worst, deviation(got, want, f"{scorer}/{protocol}/{dataset}/{metric}"))
                    checked += 1
    print(f"PIN 2 [{arm}]: {checked} per-dataset points recomputed from the score dump, "
          f"max abs deviation {worst:.3e}")
    expected = len(PROTOCOL_ID_SPLITS) * len(obj.scorers) * len(DATASETS) * len(METRICS)
    if checked != expected:
        sys.exit(f"PIN 2 FAILED [{arm}]: checked {checked} cells, expected {expected}. "
                 f"A skipped cell passes silently; a missing one must not.")
    if worst > DUMP_PIN_TOLERANCE:
        sys.exit(f"PIN 2 FAILED [{arm}]: the pickle does not reproduce from the dump.")
    return worst


def pin_auroc_by_a_different_algorithm(obj, arm: str, frame: pd.DataFrame) -> float:
    """Pin 2 shares an algorithm with the package it checks. This one does not.

    pin_points_against_dump transcribes AUROC from the definition, but it reaches it
    the same way xai_ood.metrics does, by searchsorted counts on a sorted ID array.
    Two transcriptions of one algorithm agree for reasons that have nothing to do with
    the numbers being right. The Mann-Whitney identity gets there by a different route:
    rank every score in the pooled sample with mid-ranks for ties, sum the OOD ranks,
    and subtract the rank sum the OOD side would carry if it swept the bottom.
    """
    try:
        from scipy.stats import rankdata
    except ImportError:
        print(f"PIN 2b [{arm}]: SKIPPED, scipy not importable. "
              f"Pin 2 is then the only dump check and it shares its algorithm.")
        return float("nan")
    split = frame["split"].to_numpy()
    id_mask = np.isin(split, PROTOCOL_ID_SPLITS["standard"])
    worst, checked = 0.0, 0
    for scorer in obj.scorers:
        column = frame[scorer].to_numpy(dtype=np.float64)
        id_scores = column[id_mask]
        for dataset in DATASETS:
            ood_scores = column[split == dataset]
            pooled = np.concatenate([id_scores, ood_scores])
            ranks = rankdata(pooled, method="average")[id_scores.size:]
            n_ood = ood_scores.size
            got = float((ranks.sum() - n_ood * (n_ood + 1) / 2.0) / (id_scores.size * n_ood)) * 100.0
            worst = max(worst, deviation(got, point(obj, scorer, "standard", dataset, "auroc"),
                                         f"mannwhitney {scorer}/{dataset}"))
            checked += 1
    print(f"PIN 2b [{arm}]: {checked} standard-protocol AUROCs by the Mann-Whitney "
          f"rank-sum identity, a different algorithm, max abs deviation {worst:.3e}")
    expected = len(obj.scorers) * len(DATASETS)
    if checked != expected:
        sys.exit(f"PIN 2b FAILED [{arm}]: checked {checked} cells, expected {expected}.")
    if worst > DUMP_PIN_TOLERANCE:
        sys.exit(f"PIN 2b FAILED [{arm}]: the two algorithms disagree.")
    return worst


def pin_unshrunk_standard_against_its_own_dump(obj) -> float:
    """The unshrunk standard arm has no bootstrap. Show it needs none."""
    frame = pd.read_parquet(STANDARD_UNSHRUNK_DUMP)
    split = frame["split"].to_numpy()
    id_mask = np.isin(split, PROTOCOL_ID_SPLITS["standard"])
    worst, checked = 0.0, 0
    for scorer in obj.scorers:
        column = frame[scorer].to_numpy(dtype=np.float64)
        id_scores = column[id_mask]
        for dataset in DATASETS:
            ood_scores = column[split == dataset]
            for metric in METRICS:
                got = METRIC_FN[metric](id_scores, ood_scores) * 100.0
                want = point(obj, scorer, "standard", dataset, metric)
                worst = max(worst, deviation(got, want, f"standard-dump {scorer}/{dataset}/{metric}"))
                checked += 1
    print(f"PIN 3: {checked} unshrunk standard-protocol points from the separate "
          f"2026-09-09-standard dump, max abs deviation {worst:.3e}")
    expected = len(obj.scorers) * len(DATASETS) * len(METRICS)
    if checked != expected:
        sys.exit(f"PIN 3 FAILED: checked {checked} cells, expected {expected}.")
    if worst > DUMP_PIN_TOLERANCE:
        sys.exit("PIN 3 FAILED: the full-spectrum unshrunk run is not the standard arm.")
    return worst


def pin_cross_arm_pairing(shrunk, unshrunk) -> list[str]:
    """Paired differences across arms are only defined if the resamples match.

    Shrinkage reaches the six Gaussian columns only. The other seven must have
    element-wise identical replicate arrays if replicate i is the same resample.
    """
    identical, differing = [], []
    for scorer in shrunk.scorers:
        same = all(
            np.array_equal(reps(shrunk, scorer, p, d, m), reps(unshrunk, scorer, p, d, m))
            for p in PROTOCOL_ID_SPLITS for d in DATASETS for m in METRICS
        )
        (identical if same else differing).append(scorer)
    print(f"PIN 4: {len(identical)} of {len(shrunk.scorers)} scorers have element-wise "
          f"identical replicate arrays across arms, so replicate i is the same resample")
    print(f"       shrinkage-invariant : {', '.join(identical)}")
    print(f"       shrinkage-sensitive : {', '.join(differing)}")
    if len(identical) < 5:
        sys.exit("PIN 4 FAILED: too few invariant columns to establish the pairing.")
    return identical


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def write_rows(name, header, rows):
    path = OUT / name
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path.name}  ({len(rows)} rows)")


def per_dataset_rows(objs) -> list[list]:
    rows = []
    for arm, obj in objs.items():
        for protocol in PROTOCOL_ID_SPLITS:
            for scorer in obj.scorers:
                for dataset in DATASETS:
                    for metric in METRICS:
                        low, high = interval(reps(obj, scorer, protocol, dataset, metric))
                        rows.append([
                            arm, protocol, scorer, dataset, GROUP_OF[dataset], metric,
                            f"{point(obj, scorer, protocol, dataset, metric):.4f}",
                            f"{low:.4f}", f"{high:.4f}",
                        ])
    return rows


def within_group_spread_rows(objs) -> list[list]:
    """How far apart the members of a group are, per method. Sorted widest first."""
    rows = []
    for arm, obj in objs.items():
        for protocol in PROTOCOL_ID_SPLITS:
            for metric in METRICS:
                for scorer in obj.scorers:
                    for group, datasets in GROUPS.items():
                        values = {d: point(obj, scorer, protocol, d, metric) for d in datasets}
                        lo_d = min(values, key=values.get)
                        hi_d = max(values, key=values.get)
                        spread = values[hi_d] - values[lo_d]
                        paired = (reps(obj, scorer, protocol, hi_d, metric)
                                  - reps(obj, scorer, protocol, lo_d, metric))
                        low, high = interval(paired)
                        group_value = float(np.mean(list(values.values())))
                        rows.append([
                            arm, protocol, metric, scorer, group,
                            f"{group_value:.4f}",
                            lo_d, f"{values[lo_d]:.4f}", hi_d, f"{values[hi_d]:.4f}",
                            f"{spread:.4f}", f"{low:.4f}", f"{high:.4f}",
                            f"{float(np.std(list(values.values()), ddof=1)):.4f}",
                            "yes" if low <= 0.0 <= high else "no",
                        ])
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[4], -float(r[10])))
    return rows


def dataset_deviation_rows(objs) -> list[list]:
    """Each dataset against the mean of the OTHER members of its own group.

    Leave-one-out rather than deviation from the group mean, because a two-member
    group's deviation from its own mean is the same number twice with opposite signs
    and carries no information the pair gap does not.
    """
    rows = []
    for arm, obj in objs.items():
        for protocol in PROTOCOL_ID_SPLITS:
            for metric in METRICS:
                for scorer in obj.scorers:
                    for group, datasets in GROUPS.items():
                        order = sorted(datasets, key=lambda d: -point(obj, scorer, protocol, d, metric))
                        for dataset in datasets:
                            others = [d for d in datasets if d != dataset]
                            own = reps(obj, scorer, protocol, dataset, metric)
                            rest = np.mean([reps(obj, scorer, protocol, d, metric) for d in others], axis=0)
                            deviation = (point(obj, scorer, protocol, dataset, metric)
                                         - float(np.mean([point(obj, scorer, protocol, d, metric)
                                                          for d in others])))
                            low, high = interval(own - rest)
                            rows.append([
                                arm, protocol, metric, scorer, group, dataset,
                                f"{point(obj, scorer, protocol, dataset, metric):.4f}",
                                f"{deviation:.4f}", f"{low:.4f}", f"{high:.4f}",
                                "yes" if low <= 0.0 <= high else "no",
                                order.index(dataset) + 1, len(datasets),
                            ])
    return rows


def rank_consistency_rows(deviation_rows) -> list[list]:
    """Is a dataset unlike its group for EVERY method, or only for some?"""
    frame = pd.DataFrame(deviation_rows, columns=[
        "arm", "protocol", "metric", "scorer", "group", "dataset",
        "value", "deviation", "ci_low", "ci_high", "spans_zero", "rank", "group_size",
    ])
    frame["deviation"] = frame["deviation"].astype(float)
    rows = []
    for (arm, protocol, metric, dataset), block in frame.groupby(
        ["arm", "protocol", "metric", "dataset"], sort=False
    ):
        n = len(block)
        size = int(block["group_size"].iloc[0])
        rows.append([
            arm, protocol, metric, block["group"].iloc[0], dataset, n,
            int((block["rank"] == 1).sum()), int((block["rank"] == size).sum()),
            int((block["deviation"] > 0).sum()), int((block["deviation"] < 0).sum()),
            int((block["spans_zero"] == "no").sum()),
            f"{block['deviation'].mean():.4f}",
            f"{block['deviation'].min():.4f}", f"{block['deviation'].max():.4f}",
        ])
    return rows


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with a tie correction. Written out: n is 13, so O(n^2) is free."""
    n = a.size
    concordant = discordant = tied_a = tied_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                tied_a += 1
                tied_b += 1
            elif da == 0:
                tied_a += 1
            elif db == 0:
                tied_b += 1
            elif da * db > 0:
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) / 2
    denominator = np.sqrt((pairs - tied_a) * (pairs - tied_b))
    return float((concordant - discordant) / denominator) if denominator else float("nan")


def leaderboard_churn_rows(objs) -> list[list]:
    """Per-dataset method ranking against the group-average ranking.

    The supervisor raised within-group differences as a practical-applicability
    question. This is that question in the form a practitioner asks it: if you pick a
    detector off the group leaderboard, is it the one you would have picked had you
    looked at the single dataset you actually deploy on? The paired difference between
    the group's winner and the dataset's winner carries an interval, because a winner
    that changes by less than the resampling noise is not a different choice.
    """
    rows = []
    for arm, obj in objs.items():
        for protocol in PROTOCOL_ID_SPLITS:
            for metric in METRICS:
                for group, datasets in GROUPS.items():
                    better = (lambda v: -v) if metric == "auroc" else (lambda v: v)
                    group_value = {s: float(np.mean([point(obj, s, protocol, d, metric) for d in datasets]))
                                   for s in obj.scorers}
                    group_order = sorted(group_value, key=lambda s: better(group_value[s]))
                    group_rank = {s: i + 1 for i, s in enumerate(group_order)}
                    group_winner = group_order[0]
                    for dataset in datasets:
                        values = {s: point(obj, s, protocol, dataset, metric) for s in obj.scorers}
                        order = sorted(values, key=lambda s: better(values[s]))
                        rank = {s: i + 1 for i, s in enumerate(order)}
                        winner = order[0]
                        paired = (reps(obj, winner, protocol, dataset, metric)
                                  - reps(obj, group_winner, protocol, dataset, metric))
                        low, high = interval(paired)
                        tau = kendall_tau_b(
                            np.array([group_rank[s] for s in obj.scorers], dtype=float),
                            np.array([rank[s] for s in obj.scorers], dtype=float),
                        )
                        rows.append([
                            arm, protocol, metric, group, dataset,
                            group_winner, f"{group_value[group_winner]:.4f}",
                            winner, f"{values[winner]:.4f}",
                            f"{values[winner] - values[group_winner]:.4f}",
                            f"{low:.4f}", f"{high:.4f}",
                            "yes" if low <= 0.0 <= high else "no",
                            f"{tau:.4f}",
                            sum(1 for s in obj.scorers if rank[s] != group_rank[s]),
                            max(abs(rank[s] - group_rank[s]) for s in obj.scorers),
                        ])
    return rows


def leave_one_dataset_out_rows(objs) -> list[list]:
    """Recompute the group leaderboard with each member dataset dropped in turn.

    The question this answers is not whether a group average is a good summary, which
    is row 12's, but how much of the ORDER at the top of the leaderboard belongs to
    the methods and how much to which datasets the suite happens to contain. Dehghani
    et al.'s benchmark-lottery result is the general form of it; this is the local
    measurement. Only the far-OOD group is informative here, since dropping one of a
    two-member near-OOD group leaves a single dataset rather than an average.
    """
    rows = []
    for arm, obj in objs.items():
        for protocol in PROTOCOL_ID_SPLITS:
            for metric in METRICS:
                for group, datasets in GROUPS.items():
                    if len(datasets) < 3:
                        continue
                    full = {s: float(np.mean([point(obj, s, protocol, d, metric) for d in datasets]))
                            for s in obj.scorers}
                    better = (lambda v: -v) if metric == "auroc" else (lambda v: v)
                    order = sorted(full, key=lambda s: better(full[s]))
                    for held_out in ("none",) + datasets:
                        members = [d for d in datasets if d != held_out]
                        values = {s: float(np.mean([point(obj, s, protocol, d, metric) for d in members]))
                                  for s in obj.scorers}
                        ranking = sorted(values, key=lambda s: better(values[s]))
                        for position, scorer in enumerate(ranking, start=1):
                            rows.append([
                                arm, protocol, metric, group, held_out, len(members), scorer,
                                f"{values[scorer]:.4f}", position,
                                order.index(scorer) + 1,
                                position - (order.index(scorer) + 1),
                            ])
    return rows


def shrinkage_rows(shrunk, unshrunk) -> list[list]:
    """Shrunk minus unshrunk, paired within replicate, per cell and per dataset."""
    rows = []
    for protocol in PROTOCOL_ID_SPLITS:
        for metric in METRICS:
            for scorer in shrunk.scorers:
                for group in GROUPS:
                    delta = (float(np.mean([point(shrunk, scorer, protocol, d, metric) for d in GROUPS[group]]))
                             - float(np.mean([point(unshrunk, scorer, protocol, d, metric) for d in GROUPS[group]])))
                    low, high = interval(group_mean_reps(shrunk, scorer, protocol, group, metric)
                                         - group_mean_reps(unshrunk, scorer, protocol, group, metric))
                    rows.append([protocol, metric, scorer, group, "group_average",
                                 f"{delta:.4f}", f"{low:.4f}", f"{high:.4f}",
                                 "yes" if low <= 0.0 <= high else "no"])
                for dataset in DATASETS:
                    delta = (point(shrunk, scorer, protocol, dataset, metric)
                             - point(unshrunk, scorer, protocol, dataset, metric))
                    low, high = interval(reps(shrunk, scorer, protocol, dataset, metric)
                                         - reps(unshrunk, scorer, protocol, dataset, metric))
                    rows.append([protocol, metric, scorer, GROUP_OF[dataset], dataset,
                                 f"{delta:.4f}", f"{low:.4f}", f"{high:.4f}",
                                 "yes" if low <= 0.0 <= high else "no"])
    return rows


def main() -> int:
    objs = {arm: load_pickle(run / "bootstrap" / "bootstrap_singleton.pkl")
            for arm, run in ARM_RUNS.items()}
    for arm, obj in objs.items():
        print(f"{arm:9s}: {len(obj.scorers)} scorers, groups {obj.ood_groups}, "
              f"protocols {obj.protocols}, B={obj.n_replicates}, seed={obj.seed}")
    print()

    pin_group_averages_against_gated_table(objs["shrunk"])
    for arm, run in ARM_RUNS.items():
        frame = pd.read_parquet(run / "scores" / "per_sample_scores.parquet")
        pin_points_against_dump(objs[arm], arm, frame)
        pin_auroc_by_a_different_algorithm(objs[arm], arm, frame)
    pin_unshrunk_standard_against_its_own_dump(objs["unshrunk"])
    pin_cross_arm_pairing(objs["shrunk"], objs["unshrunk"])
    print()

    write_rows(
        "per-dataset-marginals-both-arms-both-protocols.csv",
        ["arm", "protocol", "scorer", "dataset", "group", "metric", "value", "ci_low", "ci_high"],
        per_dataset_rows(objs),
    )
    write_rows(
        "within-group-spread-by-method.csv",
        ["arm", "protocol", "metric", "scorer", "group", "group_average",
         "min_dataset", "min_value", "max_dataset", "max_value",
         "spread", "spread_ci_low", "spread_ci_high", "sd_across_datasets",
         "spread_interval_spans_zero"],
        within_group_spread_rows(objs),
    )
    deviations = dataset_deviation_rows(objs)
    write_rows(
        "dataset-against-rest-of-its-group.csv",
        ["arm", "protocol", "metric", "scorer", "group", "dataset", "value",
         "deviation_from_rest", "ci_low", "ci_high", "interval_spans_zero",
         "rank_in_group", "group_size"],
        deviations,
    )
    write_rows(
        "dataset-outlier-consistency-across-methods.csv",
        ["arm", "protocol", "metric", "group", "dataset", "n_methods",
         "n_rank_first", "n_rank_last", "n_deviation_positive", "n_deviation_negative",
         "n_intervals_excluding_zero", "mean_deviation", "min_deviation", "max_deviation"],
        rank_consistency_rows(deviations),
    )
    write_rows(
        "per-dataset-leaderboard-churn.csv",
        ["arm", "protocol", "metric", "group", "dataset", "group_winner",
         "group_winner_group_value", "dataset_winner", "dataset_winner_value",
         "winner_advantage_on_this_dataset", "ci_low", "ci_high",
         "interval_spans_zero", "kendall_tau_b_vs_group_ranking",
         "n_methods_changing_rank", "largest_rank_move"],
        leaderboard_churn_rows(objs),
    )
    write_rows(
        "leave-one-dataset-out-leaderboard.csv",
        ["arm", "protocol", "metric", "group", "held_out", "n_datasets", "scorer",
         "group_value", "rank", "rank_with_all_datasets", "rank_change"],
        leave_one_dataset_out_rows(objs),
    )
    write_rows(
        "shrinkage-ablation-per-cell.csv",
        ["protocol", "metric", "scorer", "group", "level", "shrunk_minus_unshrunk",
         "ci_low", "ci_high", "interval_spans_zero"],
        shrinkage_rows(objs["shrunk"], objs["unshrunk"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
