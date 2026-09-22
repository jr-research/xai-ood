"""Which methods agree with each other, and where they fail, on the same rows.

Two questions, answered off one shared quantity.

The quantity. An earlier pass measured Spearman between raw score columns over all
78 pairs and recorded the objection to its own instrument, which is the reason this
script exists rather than repeating it: rank correlation over all rows at once is
not the comparison AUROC makes. AUROC depends only on how in-distribution rows are
ranked against out-of-distribution ones. So the per-sample quantity used here is the
ID-percentile

    u(x) = P(score_ID < score_x) + 0.5 P(score_ID = score_x)

computed against that protocol's ID pool. Its mean over an OOD dataset IS that
dataset's AUROC, exactly, which is what pins this script: the mean of u has to
reproduce the per-dataset AUROC table to float64 noise or nothing below it is
measuring what it claims. u is also the natural failure coordinate, because an OOD
image with u near 0 is one the method places deep inside the ID distribution.

Question 2, agreement. The correlation of u across methods, over the OOD rows that
set the number. Reported as a matrix per group and protocol, plus an eigenvalue
spectrum. It does NOT emit a count of distinct detectors. That count is
eleven, settled by the separate redundancy audit, and an effective rank must not be
read as a second one.

Question 3, failures. The worst-wrong rows per method: OOD images with the highest u,
which the method ranks most ID-like, and ID images with the highest raw score, which
it ranks most OOD-like. Shared across methods means the representation put those
images in the wrong place and no scoring rule recovered them; disjoint means the
scoring rule chose them.

Read-only against the repository's results tree. Writes only into this directory.
"""

from __future__ import annotations

import csv
import itertools
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
OUT = Path(__file__).resolve().parent

#: The shrunk arm is the reporting arm. Both protocols live in this one dump.
DUMP = RUNS / "2026-09-10-full-spectrum-shrunk" / "scores" / "per_sample_scores.parquet"
PICKLE = RUNS / "2026-09-10-full-spectrum-shrunk" / "bootstrap" / "bootstrap_singleton.pkl"

NEAR = ("cifar100", "tin")
FAR = ("mnist", "svhn", "texture", "places365")
GROUPS = {"near_ood": NEAR, "far_ood": FAR}
DATASETS = NEAR + FAR
PROTOCOL_ID_SPLITS = {"standard": ("id_test",), "full_spectrum": ("id_test", "csid")}
PIN_TOLERANCE = 1e-9

#: The worst-wrong set size, as a fraction of the rows it is drawn from. Stated as a
#: parameter because every overlap number below scales with it; the sweep at the end
#: of the run output is there so no single value has to be argued for.
TAIL_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
HEADLINE_FRACTION = 0.01

#: Cuts for the agreement clustering. A range rather than a value, on purpose: a split
#: that survives all of them is not a choice of threshold.
CLUSTER_CUTS = (0.05, 0.1, 0.2, 0.5, 1.0, 1.2)

#: The two known redundancies, named here so the overlap statistics can be
#: reported with and without them. rmd is bit-exactly class_conditional_full minus
#: marginal_full; pca_residual_class_mean_whitened ranks near-identically to
#: marginal_full. Dropping them is not a claim that they are uninteresting, it is a
#: control on "methods agree" being an artefact of counting one method twice.
REDUNDANT = ("rmd", "pca_residual_class_mean_whitened")


class _Stub:
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


def write_rows(name, header, rows):
    path = OUT / name
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path.name}  ({len(rows)} rows)")


def id_percentile(id_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """P(ID < x) + 0.5 P(ID = x), per row. Its mean over an OOD set is that AUROC."""
    id_sorted = np.sort(id_scores)
    below = np.searchsorted(id_sorted, scores, side="left")
    at_or_below = np.searchsorted(id_sorted, scores, side="right")
    return (below + 0.5 * (at_or_below - below)) / id_sorted.size


def spearman_matrix(columns: np.ndarray) -> np.ndarray:
    """Correlation of the column ranks. Columns are variables, rows observations."""
    ranked = np.apply_along_axis(
        lambda c: pd.Series(c).rank(method="average").to_numpy(), 0, columns
    )
    return np.corrcoef(ranked, rowvar=False)


def effective_rank(matrix: np.ndarray) -> float:
    """exp of the Shannon entropy of the eigenvalue spectrum, normalised to trace.

    Read as a spread statistic on the correlation structure and nothing else. It is
    not a count of detectors, and must not be read as one.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    weights = eigenvalues / eigenvalues.sum()
    return float(np.exp(-(weights * np.log(weights)).sum()))


def simulated_null_effective_rank(k: int, n: int, draws: int = 200,
                                  seed: int = 20260914) -> tuple[float, float]:
    """Effective rank of k INDEPENDENT columns at this matrix's own row count.

    The ceiling is k and the null is not the middle of the range: independent columns
    land just under the ceiling, and how far under depends on n. A null quoted at the
    wrong n is a caption error rather than a retraction, but it is still wrong, and a
    critic pass on 2026-09-14 caught exactly that here. So it is computed per matrix.
    """
    generator = np.random.default_rng([seed, k, n])
    values = [effective_rank(np.corrcoef(generator.standard_normal((n, k)), rowvar=False))
              for _ in range(draws)]
    return float(np.mean(values)), float(np.std(values))


def cluster_rows(scorers, correlation_rows) -> list[list]:
    """Average-linkage on 1 - rho, reported at several cuts rather than at one.

    A correlation threshold cannot by itself decide whether two scorers are the same
    detector. The answer to that is not to pick a
    better threshold, it is to report the partition at cuts spanning an order of
    magnitude and say which split survives all of them.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    index = {s: i for i, s in enumerate(scorers)}
    rows = []
    for protocol in PROTOCOL_ID_SPLITS:
        for group in GROUPS:
            matrix = np.eye(len(scorers))
            for record in correlation_rows:
                if record[0] != protocol or record[1] != group or record[2] != "ood_rows":
                    continue
                i, j = index[record[3]], index[record[4]]
                matrix[i, j] = matrix[j, i] = float(record[5])
            distance = 1.0 - matrix
            np.fill_diagonal(distance, 0.0)
            distance = (distance + distance.T) / 2.0
            tree = linkage(squareform(distance, checks=False), method="average")
            for cut in CLUSTER_CUTS:
                labels = fcluster(tree, cut, criterion="distance")
                groups = {}
                for name, label in zip(scorers, labels):
                    groups.setdefault(int(label), []).append(name)
                for label, members in sorted(groups.items()):
                    rows.append([protocol, group, f"{cut:g}", len(groups), label,
                                 ";".join(sorted(members))])
    return rows


def main() -> int:
    frame = pd.read_parquet(DUMP)
    reference = load_pickle(PICKLE)
    scorers = list(reference.scorers)
    split = frame["split"].to_numpy()
    print(f"dump {DUMP.parent.parent.name}: {len(frame)} rows, {len(scorers)} scorers")

    # ------------------------------------------------------------------ #
    # The shared quantity, and the pin on it
    # ------------------------------------------------------------------ #
    u = {}
    worst = 0.0
    checked = 0
    for protocol, id_splits in PROTOCOL_ID_SPLITS.items():
        id_mask = np.isin(split, id_splits)
        for scorer in scorers:
            column = frame[scorer].to_numpy(dtype=np.float64)
            values = id_percentile(column[id_mask], column)
            u[(protocol, scorer)] = values
            for dataset in DATASETS:
                rows = values[split == dataset]
                if rows.size == 0:
                    sys.exit(f"PIN FAILED: split {dataset!r} has no rows; the mean is vacuous.")
                got = float(rows.mean()) * 100.0
                want = float(reference.point[(scorer, protocol, dataset, "auroc")]) * 100.0
                if not (np.isfinite(got) and np.isfinite(want)):
                    # max() absorbs NaN and reports it as a perfect score. Refuse it.
                    sys.exit(f"PIN FAILED: non-finite at {scorer}/{protocol}/{dataset}")
                worst = max(worst, abs(got - want))
                checked += 1
    expected = len(DATASETS) * len(scorers) * len(PROTOCOL_ID_SPLITS)
    print(f"PIN: the mean of u over each of {checked} of an expected {expected} "
          f"(scorer, protocol, dataset) cells reproduces the per-dataset AUROC, "
          f"max abs deviation {worst:.3e}")
    if checked != expected:
        sys.exit(f"PIN FAILED: checked {checked} cells, expected {expected}.")
    if worst > PIN_TOLERANCE:
        sys.exit("PIN FAILED: u is not the quantity AUROC averages. Stopping.")
    print()

    # ------------------------------------------------------------------ #
    # Question 2: agreement on the comparison that sets the number
    # ------------------------------------------------------------------ #
    correlation_rows, spectrum_rows = [], []
    for protocol in PROTOCOL_ID_SPLITS:
        for group, datasets in GROUPS.items():
            mask = np.isin(split, datasets)
            columns = np.column_stack([u[(protocol, s)][mask] for s in scorers])
            matrix = spearman_matrix(columns)
            for i, j in itertools.combinations(range(len(scorers)), 2):
                correlation_rows.append([
                    protocol, group, "ood_rows", scorers[i], scorers[j],
                    f"{matrix[i, j]:.6f}", int(mask.sum()),
                ])
            # A group-level correlation is itself a group average, and the rule that
            # applies to a group AUROC applies to this instrument too: far-OOD rows are
            # 51% MNIST, so one dataset can carry a pooled rho. The per-dataset rows
            # below are the companion that rule requires.
            for dataset in datasets:
                inner = np.isin(split, [dataset])
                per_dataset = spearman_matrix(
                    np.column_stack([u[(protocol, s)][inner] for s in scorers]))
                for i, j in itertools.combinations(range(len(scorers)), 2):
                    correlation_rows.append([
                        protocol, group, dataset, scorers[i], scorers[j],
                        f"{per_dataset[i, j]:.6f}", int(inner.sum()),
                    ])
            keep = [k for k, s in enumerate(scorers) if s not in REDUNDANT]
            null_mean, null_sd = simulated_null_effective_rank(len(scorers), int(mask.sum()))
            spectrum_rows.append([
                protocol, group, "all_13", f"{effective_rank(matrix):.4f}",
                int(mask.sum()), f"{null_mean:.4f}", f"{null_sd:.4f}",
                f"{matrix[np.triu_indices(len(scorers), 1)].mean():.6f}",
                f"{matrix[np.triu_indices(len(scorers), 1)].min():.6f}",
                f"{matrix[np.triu_indices(len(scorers), 1)].max():.6f}",
            ])
            reduced = matrix[np.ix_(keep, keep)]
            null_mean, null_sd = simulated_null_effective_rank(len(keep), int(mask.sum()))
            spectrum_rows.append([
                protocol, group, "without_redundant_11", f"{effective_rank(reduced):.4f}",
                int(mask.sum()), f"{null_mean:.4f}", f"{null_sd:.4f}",
                f"{reduced[np.triu_indices(len(keep), 1)].mean():.6f}",
                f"{reduced[np.triu_indices(len(keep), 1)].min():.6f}",
                f"{reduced[np.triu_indices(len(keep), 1)].max():.6f}",
            ])
    write_rows(
        "method-agreement-on-id-percentile.csv",
        ["protocol", "group", "rows", "scorer_a", "scorer_b", "spearman", "n_rows"],
        correlation_rows,
    )
    write_rows(
        "method-agreement-clusters.csv",
        ["protocol", "group", "cut", "n_clusters", "cluster", "members"],
        cluster_rows(scorers, correlation_rows),
    )
    write_rows(
        "method-agreement-spectrum.csv",
        ["protocol", "group", "set", "effective_rank", "n_rows",
         "null_effective_rank_mean", "null_effective_rank_sd", "mean_offdiagonal",
         "min_offdiagonal", "max_offdiagonal"],
        spectrum_rows,
    )

    # ------------------------------------------------------------------ #
    # Question 3: where the methods fail
    # ------------------------------------------------------------------ #
    overlap_rows, membership_rows = [], []
    for protocol in PROTOCOL_ID_SPLITS:
        for side, label in (("ood", "ranked most ID-like"), ("id", "ranked most OOD-like")):
            universes = DATASETS if side == "ood" else ("id_test",)
            for universe in universes:
                mask = split == universe
                n = int(mask.sum())
                for fraction in TAIL_FRACTIONS:
                    k = max(int(round(fraction * n)), 1)
                    tails = {}
                    for scorer in scorers:
                        if side == "ood":
                            # lowest u: placed deepest inside the ID distribution
                            order = np.argsort(u[(protocol, scorer)][mask], kind="stable")
                        else:
                            # highest u: an ID image scored more OOD than the OOD rows
                            order = np.argsort(-u[(protocol, scorer)][mask], kind="stable")
                        tails[scorer] = set(order[:k].tolist())
                    counts = np.zeros(n, dtype=np.int32)
                    for members in tails.values():
                        counts[list(members)] += 1
                    union = int((counts > 0).sum())
                    kept = [s for s in scorers if s not in REDUNDANT]
                    counts_kept = np.zeros(n, dtype=np.int32)
                    for scorer in kept:
                        counts_kept[list(tails[scorer])] += 1
                    pairs = [
                        len(tails[a] & tails[b]) / len(tails[a] | tails[b])
                        for a, b in itertools.combinations(scorers, 2)
                    ]
                    overlap_rows.append([
                        protocol, side, universe, n, f"{fraction:g}", k,
                        f"{float(np.mean(pairs)):.4f}", f"{float(np.min(pairs)):.4f}",
                        f"{float(np.max(pairs)):.4f}",
                        union, int((counts == len(scorers)).sum()),
                        int((counts == 1).sum()),
                        int((counts_kept == len(kept)).sum()),
                        f"{k / n:.6f}", f"{(k / n) ** 2 / (2 * k / n - (k / n) ** 2):.6f}",
                    ])
                    if abs(fraction - HEADLINE_FRACTION) < 1e-12:
                        for threshold in range(1, len(scorers) + 1):
                            membership_rows.append([
                                protocol, side, universe, k, threshold,
                                int((counts >= threshold).sum()),
                            ])
    write_rows(
        "worst-wrong-overlap.csv",
        ["protocol", "side", "universe", "n_rows", "tail_fraction", "tail_size",
         "mean_pairwise_jaccard", "min_pairwise_jaccard", "max_pairwise_jaccard",
         "union_size", "n_in_all_13", "n_in_exactly_1", "n_in_all_11_non_redundant",
         "chance_tail_rate", "chance_pairwise_jaccard"],
        overlap_rows,
    )
    write_rows(
        "worst-wrong-membership-profile.csv",
        ["protocol", "side", "universe", "tail_size", "at_least_n_methods", "n_images"],
        membership_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
