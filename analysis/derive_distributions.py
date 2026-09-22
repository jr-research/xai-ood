"""The per-sample score distributions, which every table in this directory reduces away.

Three questions, all answered off the score dumps and nothing else.

  1. The shape of the ID and OOD score distributions, per method and per dataset, in a
     form a figure can draw. Emitted as quantile grids rather than raw scores, because a
     312,660-row dump is not a figure input and a histogram of it is not reproducible
     from a CSV.
  2. The right tail in the two full-covariance cells, reproduced here by code written
     from the definitions rather than transcribed from the earlier report of it, then
     extended in the two directions that report left open: whether the tail is the SAME
     IMAGES across the two full-covariance cells, and whether it exists on the OOD and
     corrupted rows or only on the clean in-distribution ones.
  3. How much of the between-method variance in a group's AUROC each member dataset
     carries. An exact additive decomposition, not a ranking.

Which methods are really the same method is settled elsewhere and not re-answered here.
A separate redundancy audit counts eleven distinct detectors, with twelve as the floor,
and `method-agreement-*.csv` in this directory reaches the same two redundancies. What
this script adds is the raw-score Spearman matrix as a third path to those two, stated
as such in its own output rather than emitted as a rival count. An effective rank is
NOT a second count of detectors and is not read as one anywhere below.

Shared or disjoint failures are likewise settled, by `worst-wrong-overlap.csv` and
`worst-wrong-membership-profile.csv`. What is added here instead is the one part of that
question which touches question 2: whether the clean-ID images in the full-covariance
right tail are the same images each method ranks most OOD-like.

FOUR PINS, and each asserts the number of cells it checked, because a skipped cell
passes silently:

  A. Every per-dataset AUROC drawn or quoted here is recomputed from the dump and
     compared against the per-dataset table of record. 156 cells, standard protocol,
     both arms.
  B. `rmd` is bit-exactly `class_conditional_full - marginal_full` in both arms. That
     relation is structural: if it fails, the dump is not the dump.
  C. Row counts per split, against the published split sizes.
  D. Non-finite values exit rather than accumulate. A NaN compared with `>` is silently
     False and an earlier version of the pins in this directory passed a planted NaN as
     a perfect score.

Read-only against the repository's results tree. Writes only into this directory.

Needs pyarrow, which the `xai-ood` environment carries:

    python derive_distributions.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# The repository root, the directory above this one. Overridable with
# XAI_OOD_ROOT so a run tree held outside the checkout can be read in place.
ROOT = Path(os.environ.get("XAI_OOD_ROOT", Path(__file__).resolve().parents[1]))
RUNS = ROOT / "results" / "runs"
OUT = Path(__file__).resolve().parent

ARM_DUMP = {
    "shrunk": RUNS / "2026-09-10-full-spectrum-shrunk" / "scores" / "per_sample_scores.parquet",
    "unshrunk": RUNS / "2026-09-10-full-spectrum-unshrunk" / "scores" / "per_sample_scores.parquet",
}
TABLE_OF_RECORD = OUT / "per-dataset-marginals-both-arms-both-protocols.csv"

SCORERS = (
    "marginal_diagonal", "marginal_full", "class_conditional_diagonal",
    "class_conditional_full", "rmd", "rmd_pp", "knn_normalized", "knn_unnormalized",
    "pca_residual_all_id", "pca_residual_class_mean", "pca_residual_all_id_l2",
    "pca_residual_class_mean_l2", "pca_residual_class_mean_whitened",
)
FULL_CELLS = ("marginal_full", "class_conditional_full")
DIAGONAL_CELLS = ("marginal_diagonal", "class_conditional_diagonal")

NEAR = ("cifar100", "tin")
FAR = ("mnist", "svhn", "texture", "places365")
GROUPS = {"near_ood": NEAR, "far_ood": FAR}
GROUP_OF = {d: g for g, ds in GROUPS.items() for d in ds}
DATASETS = NEAR + FAR

#: Copied from xai_ood.bootstrap.plan.PROTOCOL_ID_SPLITS rather than imported, so a pin
#: does not depend on the package whose output it checks. Same convention as
#: derive_per_dataset.py.
PROTOCOL_ID_SPLITS = {"standard": ("id_test",), "full_spectrum": ("id_test", "csid")}

#: Published split sizes, for pin C.
EXPECTED_ROWS = {
    "id_test": 9000, "cifar100": 9000, "tin": 7793, "mnist": 70000,
    "svhn": 26032, "texture": 5640, "places365": 35195, "csid": 150000,
}

#: The table of record is written to four decimals, so a recomputation off the dump
#: cannot agree more closely than its rounding. An independent pin against the same
#: table measured 5.000e-05 for the same reason.
TABLE_TOLERANCE = 1e-4

#: Quantile grid for the figure inputs. Dense in the tails, because the tail is the
#: finding and a 1%-spaced grid cannot draw a p99.9.
QUANTILES = np.concatenate([
    np.array([0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075]),
    np.arange(0.01, 0.99 + 1e-9, 0.01),
    np.array([0.9925, 0.995, 0.9975, 0.999, 0.9995, 0.9999, 1.0]),
])

#: Tail sizes for the same-images question. A single value cannot carry the claim; the
#: sweep is in the emitted CSV and the record quotes the range.
TAIL_FRACTIONS = (0.001, 0.005, 0.01, 0.05, 0.10)
HEADLINE_TAIL = 0.05

#: Bin count for the emitted histogram grid. 300 resolves the tail without drawing noise
#: at 5,640 rows, which is the smallest universe here.
BINS = 300


class Pins:
    """Every pin reports how many cells it checked. A pin that checked none is a fail."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, int, float, float, bool]] = []

    def record(self, name: str, examined: int, worst: float, tolerance: float) -> None:
        ok = examined > 0 and np.isfinite(worst) and worst <= tolerance
        self.rows.append((name, examined, worst, tolerance, ok))

    def report(self) -> bool:
        print("\nPINS")
        every_ok = True
        for name, examined, worst, tol, ok in self.rows:
            every_ok &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: examined {examined} cells, "
                  f"worst {worst:.3e}, tolerance {tol:.0e}")
        if not every_ok:
            print("\nREFUSED: a pin failed. Nothing below it is reportable.")
        return every_ok


def require_finite(name: str, values: np.ndarray) -> np.ndarray:
    """Pin D. A non-finite value exits rather than propagating into a max()."""
    if not np.isfinite(values).all():
        sys.stderr.write(
            f"REFUSED: {name} contains {int((~np.isfinite(values)).sum())} non-finite "
            f"values of {values.size}. A NaN compared with '>' is False and would have "
            f"arrived as a perfect score.\n")
        raise SystemExit(2)
    return values


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load(arm: str) -> pd.DataFrame:
    columns = ["split", "image_id", "corruption", "severity", *SCORERS]
    frame = pq.read_table(ARM_DUMP[arm], columns=columns).to_pandas()
    for scorer in SCORERS:
        require_finite(f"{arm}/{scorer}", frame[scorer].to_numpy(float))
    return frame


def id_pool(frame: pd.DataFrame, protocol: str) -> np.ndarray:
    return frame["split"].isin(PROTOCOL_ID_SPLITS[protocol]).to_numpy()


# --------------------------------------------------------------------------- #
# Metrics, written from the definitions
# --------------------------------------------------------------------------- #

def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Mann-Whitney form, on the 0-100 scale, higher score meaning more out-of-distribution.

    Ties get half credit, which is what the rank-sum identity gives and what every AUROC
    in this project uses.
    """
    if id_scores.size == 0 or ood_scores.size == 0:
        sys.stderr.write("REFUSED: AUROC over an empty universe.\n")
        raise SystemExit(2)
    pooled = np.concatenate([id_scores, ood_scores])
    ranks = pd.Series(pooled).rank(method="average").to_numpy()
    n_id, n_ood = id_scores.size, ood_scores.size
    rank_sum_ood = ranks[n_id:].sum()
    return 100.0 * (rank_sum_ood - n_ood * (n_ood + 1) / 2.0) / (n_id * n_ood)


def skewness(values: np.ndarray) -> float:
    """Third standardised moment, population form, written out rather than imported.

    This is deliberately the same estimator the note being checked used, because the
    question is whether its number reproduces, not whether a different estimator of a
    different quantity lands somewhere else.
    """
    centred = values - values.mean()
    return float((centred ** 3).mean() / values.std() ** 3)


def shape(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)),
        "skewness": skewness(values),
        "p01": float(np.quantile(values, 0.01)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "frac_above_1p5_median": float((values > 1.5 * np.median(values)).mean()),
    }


# --------------------------------------------------------------------------- #
# Pins
# --------------------------------------------------------------------------- #

def table_of_record() -> dict:
    wanted = {}
    with TABLE_OF_RECORD.open() as fh:
        for row in csv.DictReader(fh):
            if row["metric"] != "auroc":
                continue
            wanted[(row["arm"], row["protocol"], row["scorer"], row["dataset"])] = float(row["value"])
    return wanted


def pin_a(frames: dict, pins: Pins) -> None:
    """Every per-dataset AUROC drawn or quoted below, against the table of record."""
    wanted = table_of_record()
    worst, examined = 0.0, 0
    for arm, frame in frames.items():
        for protocol in PROTOCOL_ID_SPLITS:
            pool = id_pool(frame, protocol)
            for scorer in SCORERS:
                column = frame[scorer].to_numpy(float)
                id_scores = column[pool]
                for dataset in DATASETS:
                    key = (arm, protocol, scorer, dataset)
                    if key not in wanted:
                        sys.stderr.write(f"REFUSED: {key} absent from the table of record.\n")
                        raise SystemExit(2)
                    got = auroc(id_scores, column[(frame["split"] == dataset).to_numpy()])
                    worst = max(worst, abs(got - wanted[key]))
                    examined += 1
    pins.record("A, per-dataset AUROC against the table of record", examined, worst, TABLE_TOLERANCE)


def pin_b(frames: dict, pins: Pins) -> None:
    """The exact structural relation between the three cells. If it fails the dump is not the dump."""
    worst, examined = 0.0, 0
    for arm, frame in frames.items():
        difference = (frame["class_conditional_full"].to_numpy(float)
                      - frame["marginal_full"].to_numpy(float))
        worst = max(worst, float(np.abs(frame["rmd"].to_numpy(float) - difference).max()))
        examined += 1
    pins.record("B, rmd == class_conditional_full - marginal_full", examined, worst, 0.0)


def pin_c(frames: dict, pins: Pins) -> None:
    worst, examined = 0.0, 0
    for arm, frame in frames.items():
        counts = frame.groupby("split").size().to_dict()
        for split, expected in EXPECTED_ROWS.items():
            if split not in counts:
                sys.stderr.write(f"REFUSED: split {split} missing from the {arm} dump.\n")
                raise SystemExit(2)
            worst = max(worst, abs(counts[split] - expected))
            examined += 1
    pins.record("C, row counts per split", examined, float(worst), 0.0)


# --------------------------------------------------------------------------- #
# Question 1. Distribution shape, per method and per dataset
# --------------------------------------------------------------------------- #

def emit_quantiles(frames: dict) -> None:
    """Quantile grids, one row per arm x protocol x scorer x split. The figure input.

    Raw scores are not emitted: 312,660 rows x 13 columns is a dump, not a figure input,
    and it already exists upstream. A quantile grid redraws every histogram
    below to the eye and is 2 MB rather than 60.
    """
    path = OUT / "score-quantiles-by-method-and-split.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "protocol", "scorer", "split", "role", "group", "n",
                         "quantile", "value"])
        for arm, frame in frames.items():
            for protocol in PROTOCOL_ID_SPLITS:
                pool = id_pool(frame, protocol)
                for scorer in SCORERS:
                    column = frame[scorer].to_numpy(float)
                    universe = [("id", "ID pool", column[pool])]
                    universe += [("ood", d, column[(frame["split"] == d).to_numpy()])
                                 for d in DATASETS]
                    for role, split, values in universe:
                        levels = np.quantile(values, QUANTILES)
                        group = GROUP_OF.get(split, "")
                        for q, v in zip(QUANTILES, levels):
                            writer.writerow([arm, protocol, scorer, split, role, group,
                                             values.size, f"{q:.6f}", f"{v:.6f}"])
    print(f"  wrote {path.name}")


def emit_histograms(frames: dict) -> None:
    """Binned counts, the figure input proper. The drawing script computes nothing.

    One shared bin grid per scorer, spanning the 0.05th to 99.95th percentile of every
    row that scorer produced, so panels in a row of the figure are on one axis and a
    reader can compare them. Counts are emitted per split plus for each protocol's ID
    pool, because summing two splits to make a pool is a computation and belongs here.

    Shrunk arm only: it is the reporting arm and the unshrunk one is checked in the
    tail tables above, where the comparison is the point.
    """
    arm = "shrunk"
    frame = frames[arm]
    payload = {
        "_source": str(ARM_DUMP[arm]),
        "_arm": arm,
        "_bins": BINS,
        "_note": "Generated by derive_distributions.py. The figure script draws these and computes nothing.",
        "scorers": {},
    }
    for scorer in SCORERS:
        column = frame[scorer].to_numpy(float)
        lo, hi = np.quantile(column, [0.0005, 0.9995])
        edges = np.linspace(float(lo), float(hi), BINS + 1)
        entry = {"edges": [round(float(e), 6) for e in edges], "counts": {}, "auroc": {}}
        universes = {s: (frame["split"] == s).to_numpy() for s in ("id_test", "csid", *DATASETS)}
        universes["id_pool_standard"] = id_pool(frame, "standard")
        universes["id_pool_full_spectrum"] = id_pool(frame, "full_spectrum")
        for name, mask in universes.items():
            counts, _ = np.histogram(column[mask], bins=edges)
            entry["counts"][name] = [int(c) for c in counts]
            entry.setdefault("n", {})[name] = int(mask.sum())
            entry.setdefault("clipped", {})[name] = int(
                ((column[mask] < edges[0]) | (column[mask] > edges[-1])).sum())
        for protocol in PROTOCOL_ID_SPLITS:
            id_scores = column[id_pool(frame, protocol)]
            for dataset in DATASETS:
                entry["auroc"][f"{protocol}/{dataset}"] = round(
                    auroc(id_scores, column[(frame["split"] == dataset).to_numpy()]), 4)
        entry["median"] = {
            name: round(float(np.median(column[mask])), 6) for name, mask in universes.items()
        }
        payload["scorers"][scorer] = entry
    path = OUT / "distribution-histograms.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {path.name}")


def emit_overlap_summary(frames: dict) -> None:
    """One row per drawn panel: the two distributions' shape, the AUROC, and the overlap.

    The overlap coefficient is the area shared by the two densities, estimated on a
    common 512-bin grid. It is the number a histogram shows and an AUROC does not: two
    cells can separate equally well and share very different amounts of mass.
    """
    path = OUT / "id-against-ood-overlap.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "protocol", "scorer", "dataset", "group", "auroc",
                         "overlap_area", "id_median", "ood_median", "median_shift_in_id_sd",
                         "id_skewness", "ood_skewness", "ood_below_id_median_frac"])
        for arm, frame in frames.items():
            for protocol in PROTOCOL_ID_SPLITS:
                pool = id_pool(frame, protocol)
                for scorer in SCORERS:
                    column = frame[scorer].to_numpy(float)
                    id_scores = column[pool]
                    id_sd = id_scores.std(ddof=1)
                    for dataset in DATASETS:
                        ood = column[(frame["split"] == dataset).to_numpy()]
                        lo = min(np.quantile(id_scores, 0.0005), np.quantile(ood, 0.0005))
                        hi = max(np.quantile(id_scores, 0.9995), np.quantile(ood, 0.9995))
                        edges = np.linspace(lo, hi, 513)
                        p, _ = np.histogram(id_scores, bins=edges, density=False)
                        q, _ = np.histogram(ood, bins=edges, density=False)
                        p = p / max(p.sum(), 1)
                        q = q / max(q.sum(), 1)
                        writer.writerow([
                            arm, protocol, scorer, dataset, GROUP_OF[dataset],
                            f"{auroc(id_scores, ood):.4f}",
                            f"{float(np.minimum(p, q).sum()):.4f}",
                            f"{float(np.median(id_scores)):.6f}",
                            f"{float(np.median(ood)):.6f}",
                            f"{float((np.median(ood) - np.median(id_scores)) / id_sd):.4f}",
                            f"{skewness(id_scores):.4f}", f"{skewness(ood):.4f}",
                            f"{float((ood < np.median(id_scores)).mean()):.4f}",
                        ])
    print(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Question 2. The right tail
# --------------------------------------------------------------------------- #

def emit_tail_shape(frames: dict) -> None:
    """The shape statistics, per arm, per cell, per split. Clean ID, every OOD set, cs-ID.

    This is the independent reproduction. The note that first reported the tail measured
    five cells on one split of one arm; this measures thirteen scorers on eight splits of
    two arms, with the estimator written out above rather than imported.
    """
    path = OUT / "tail-shape-by-cell-and-split.csv"
    rows = []
    for arm, frame in frames.items():
        for scorer in SCORERS:
            column = frame[scorer].to_numpy(float)
            for split in ("id_test", *DATASETS, "csid"):
                values = column[(frame["split"] == split).to_numpy()]
                row = {"arm": arm, "scorer": scorer, "split": split}
                row.update(shape(values))
                rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
    print(f"  wrote {path.name}")
    return pd.DataFrame(rows)


def emit_tail_membership(frames: dict) -> None:
    """Is it the SAME IMAGES in the tail across the two full-covariance cells?

    The question the note left open, and the one that decides whether the tail is a
    property of the images or of the scoring rule. Measured on clean ID rows, where the
    tail was found, by the overlap of the top-fraction of each cell over a sweep of tail
    sizes. Chance is reported beside every observed value, because a Jaccard null is
    p^2/(2p - p^2) and not p, and the two were conflated once in this directory already.

    The diagonal pair is the control: if the full pair overlaps no more than the diagonal
    pair does, the answer is the scoring rule and the tail is incidental.
    """
    pairs = [
        ("marginal_full", "class_conditional_full", "the two full-covariance cells"),
        ("marginal_diagonal", "class_conditional_diagonal", "control, the two diagonal cells"),
        ("marginal_full", "marginal_diagonal", "control, across the covariance shape"),
        ("class_conditional_full", "class_conditional_diagonal", "control, across the covariance shape"),
    ]
    path = OUT / "tail-membership-across-cells.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "split", "cell_a", "cell_b", "pair_role", "tail_fraction",
                         "tail_size", "shared", "jaccard", "jaccard_chance",
                         "overlap_coefficient", "overlap_chance"])
        for arm, frame in frames.items():
            for split in ("id_test", "svhn", "csid"):
                mask = (frame["split"] == split).to_numpy()
                for a, b, role in pairs:
                    va = frame[a].to_numpy(float)[mask]
                    vb = frame[b].to_numpy(float)[mask]
                    n = va.size
                    order_a, order_b = np.argsort(-va), np.argsort(-vb)
                    for fraction in TAIL_FRACTIONS:
                        k = max(1, int(round(fraction * n)))
                        set_a, set_b = set(order_a[:k].tolist()), set(order_b[:k].tolist())
                        shared = len(set_a & set_b)
                        jaccard = shared / len(set_a | set_b)
                        p = k / n
                        writer.writerow([
                            arm, split, a, b, role, f"{fraction:.4f}", k, shared,
                            f"{jaccard:.4f}", f"{p * p / (2 * p - p * p):.6f}",
                            f"{2 * shared / (len(set_a) + len(set_b)):.4f}", f"{p:.6f}",
                        ])
    print(f"  wrote {path.name}")


def emit_tail_against_worst_wrong(frames: dict) -> None:
    """Are the clean-ID tail images the ones each method ranks most OOD-like?

    The link between question 2 and the per-dataset failure analysis, which owns the
    general version of this. An ID image with a very high score is exactly a false positive, so
    the full-covariance right tail is a candidate explanation for that method's worst-wrong
    ID set. Measured as the share of `marginal_full`'s clean-ID tail that each of the
    other twelve also places in its own equally sized tail.
    """
    path = OUT / "tail-against-worst-wrong-id.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "reference_cell", "other_scorer", "tail_fraction",
                         "tail_size", "shared", "share_of_reference_tail", "chance"])
        for arm, frame in frames.items():
            mask = (frame["split"] == "id_test").to_numpy()
            for reference in FULL_CELLS:
                ref = frame[reference].to_numpy(float)[mask]
                n = ref.size
                order_ref = np.argsort(-ref)
                for scorer in SCORERS:
                    if scorer == reference:
                        continue
                    other = np.argsort(-frame[scorer].to_numpy(float)[mask])
                    for fraction in TAIL_FRACTIONS:
                        k = max(1, int(round(fraction * n)))
                        shared = len(set(order_ref[:k].tolist()) & set(other[:k].tolist()))
                        writer.writerow([arm, reference, scorer, f"{fraction:.4f}", k,
                                         shared, f"{shared / k:.4f}", f"{k / n:.6f}"])
    print(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Question 3. Which dataset carries the between-method variance
# --------------------------------------------------------------------------- #

def emit_variance_decomposition(frames: dict) -> None:
    """An exact additive split of a group's between-method AUROC variance by dataset.

    The group AUROC of method m is the unweighted mean of its member datasets' AUROCs,
    A_m = sum_d w_d A_md with w_d = 1/|G|. Variance over the thirteen methods is then

        Var_m(A) = sum_d w_d Cov_m(A_.d, A_.)

    exactly, with no residual, because covariance is bilinear. Each dataset's share is
    that term over the total. Shares sum to 1 by construction and may be NEGATIVE, which
    is the interesting case: a dataset whose method ordering runs against the group's
    removes between-method variance rather than adding it.

    Also emitted: the between-method variance each dataset carries ALONE, which is a
    different question and answers "how much do methods disagree here", and the group
    variance recomputed with each dataset dropped.

    The unit is squared AUROC points on the 0-100 scale. Nothing here is a significance
    verdict: these are thirteen methods, not a sample from a population of methods.
    """
    wanted = table_of_record()
    path = OUT / "between-method-variance-by-dataset.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "protocol", "group", "dataset", "group_variance",
                         "dataset_variance_alone", "covariance_with_group",
                         "contribution", "share_of_group_variance",
                         "group_variance_without_this_dataset", "variance_ratio_without"])
        for arm in frames:
            for protocol in PROTOCOL_ID_SPLITS:
                for group, members in GROUPS.items():
                    per_dataset = np.array([
                        [wanted[(arm, protocol, s, d)] for d in members] for s in SCORERS
                    ])  # 13 methods x |G| datasets
                    weight = 1.0 / len(members)
                    group_scores = per_dataset.mean(axis=1)
                    group_var = float(group_scores.var(ddof=1))
                    for index, dataset in enumerate(members):
                        column = per_dataset[:, index]
                        covariance = float(np.cov(column, group_scores, ddof=1)[0, 1])
                        contribution = weight * covariance
                        if len(members) > 2:
                            kept = [j for j in range(len(members)) if j != index]
                            without = float(per_dataset[:, kept].mean(axis=1).var(ddof=1))
                            ratio = f"{without / group_var:.4f}"
                            without_str = f"{without:.4f}"
                        else:
                            without_str, ratio = "", ""
                        writer.writerow([
                            arm, protocol, group, dataset, f"{group_var:.4f}",
                            f"{float(column.var(ddof=1)):.4f}", f"{covariance:.4f}",
                            f"{contribution:.4f}", f"{contribution / group_var:.4f}",
                            without_str, ratio,
                        ])
    print(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Item 3 of the prompt: the correlation matrix, as a citation rather than a rival count
# --------------------------------------------------------------------------- #

def emit_spearman(frames: dict) -> None:
    """Raw-score Spearman across the thirteen, on identical rows, both protocols.

    A third path to the two redundancies the detector count rests on. It is NOT a
    second count of detectors, and an effective rank must not be read as one: the count
    is eleven, from the separate redundancy audit, with twelve as the floor.
    """
    path = OUT / "raw-score-spearman-by-protocol.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "protocol", "row_subset", "n_rows", "scorer_a", "scorer_b", "spearman"])
        for arm, frame in frames.items():
            subsets = {
                "clean_id": (frame["split"] == "id_test").to_numpy(),
                "ood_rows": frame["split"].isin(DATASETS).to_numpy(),
                "all_rows": np.ones(len(frame), dtype=bool),
            }
            for subset, mask in subsets.items():
                ranks = np.column_stack([
                    pd.Series(frame[s].to_numpy(float)[mask]).rank(method="average").to_numpy()
                    for s in SCORERS
                ])
                matrix = np.corrcoef(ranks, rowvar=False)
                for i, a in enumerate(SCORERS):
                    for j, b in enumerate(SCORERS):
                        if j <= i:
                            continue
                        writer.writerow([arm, "both", subset, int(mask.sum()), a, b,
                                         f"{matrix[i, j]:.6f}"])
    print(f"  wrote {path.name}")


def main() -> int:
    print("Loading the two full-spectrum dumps.")
    frames = {arm: load(arm) for arm in ARM_DUMP}
    for arm, frame in frames.items():
        print(f"  {arm}: {len(frame)} rows, {len(SCORERS)} scorer columns")

    pins = Pins()
    pin_b(frames, pins)
    pin_c(frames, pins)
    print("\nPin A recomputes 156 per-dataset AUROCs per arm from the dump. This is the slow one.")
    pin_a(frames, pins)
    if not pins.report():
        return 2

    print("\nEMITTING")
    emit_quantiles(frames)
    emit_histograms(frames)
    emit_overlap_summary(frames)
    emit_tail_shape(frames)
    emit_tail_membership(frames)
    emit_tail_against_worst_wrong(frames)
    emit_variance_decomposition(frames)
    emit_spearman(frames)

    print("\nWHAT THIS RUN DID NOT CHECK, printed every time:")
    for blind in (
        "the scores themselves. It takes the dumps as given and does not re-derive them",
        "the full-spectrum protocol's cs-ID rows as a per-dataset universe, only as ID pool",
        "any interval. Nothing here is resampled and nothing here gets a significance verdict",
        "detector redundancy and shared failures, both settled elsewhere and not repeated here",
    ):
        print(f"  - {blind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
