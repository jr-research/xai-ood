"""Derive every contrast the thesis reports, from the persisted bootstrap replicate arrays.

No table in either run carries a contrast interval, so every
difference the write-up quotes has to be rebuilt from the replicates. This script is that
rebuild, written down once instead of retyped, so the numbers live in a file rather than in
a terminal.

Two things it does that a hand derivation kept getting wrong:

  1. It reads bootstrap_singleton.pkl, whose groups are the six datasets, and averages the
     per-dataset arrays INSIDE each replicate. That is the declared dataset-averaged
     estimand. bootstrap_grouped.pkl holds the POOLED replicates and is a different
     quantity; reading it by mistake is what produced the pooled degradation figures.

  2. It pins itself before it reports anything, by rebuilding all 26 dataset-averaged
     marginals and comparing them to the gated CSV. If that check does not print a
     deviation at float64 noise, nothing below it can be trusted.

Read-only. Writes only into this directory.
"""
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# The repository root, the directory above this one. Overridable with
# XAI_OOD_ROOT so a run tree held outside the checkout can be read in place.
ROOT = Path(os.environ.get("XAI_OOD_ROOT", Path(__file__).resolve().parents[1]))
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"
OUT = Path(__file__).resolve().parent

STANDARD_RUN = RUNS / "2026-09-09-standard-shrunk"
FULL_SPECTRUM_RUN = RUNS / "2026-09-10-full-spectrum-shrunk"
GATED_STANDARD_TABLE = TABLES / "2026-09-09-standard-shrunk-dataset-averaged.csv"

NEAR = ["cifar100", "tin"]
FAR = ["mnist", "svhn", "texture", "places365"]
GROUPS = {"near_ood": NEAR, "far_ood": FAR}

FACTORIAL = {
    "marginal_diagonal": ("marginal", "diagonal"),
    "marginal_full": ("marginal", "full"),
    "class_conditional_diagonal": ("class_conditional", "diagonal"),
    "class_conditional_full": ("class_conditional", "full"),
}


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


def load(path):
    with open(path, "rb") as handle:
        return _Unpickler(handle).load()


def group_point(obj, scorer, protocol, group):
    return float(np.mean([obj.point[(scorer, protocol, d, "auroc")] for d in GROUPS[group]])) * 100.0


def group_replicates(obj, scorer, protocol, group):
    arrays = [np.asarray(obj.values[(scorer, protocol, d, "auroc")], dtype=float) for d in GROUPS[group]]
    return np.mean(arrays, axis=0) * 100.0


def interval(replicates):
    return float(np.percentile(replicates, 2.5)), float(np.percentile(replicates, 97.5))


def pin_against_gated_table(obj):
    """Refuse to report anything unless the construction reproduces the gated table."""
    gated = {}
    with open(GATED_STANDARD_TABLE) as handle:
        for row in csv.DictReader(handle):
            gated[(row["scorer"], row["group"])] = (
                float(row["auroc"]), float(row["ci_low"]), float(row["ci_high"])
            )
    worst = 0.0
    for (scorer, group), (auroc, low, high) in gated.items():
        point = group_point(obj, scorer, "standard", group)
        rebuilt_low, rebuilt_high = interval(group_replicates(obj, scorer, "standard", group))
        worst = max(worst, abs(point - auroc), abs(rebuilt_low - low), abs(rebuilt_high - high))
    print(f"PIN: {len(gated)} dataset-averaged rows rebuilt from replicates, "
          f"max abs deviation vs the gated table {worst:.3e}")
    if worst > 1e-9:
        sys.exit("PIN FAILED: the construction does not reproduce the gated table. Stopping.")
    return worst


def write_rows(name, header, rows):
    path = OUT / name
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path.name}  ({len(rows)} rows)")


def degradation_rows(obj):
    """Standard minus full-spectrum, per scorer and group. Positive means AUROC lost."""
    rows = []
    for scorer in obj.scorers:
        if scorer.startswith("control_"):
            continue
        for group in GROUPS:
            point = group_point(obj, scorer, "standard", group) - group_point(obj, scorer, "full_spectrum", group)
            replicates = (group_replicates(obj, scorer, "standard", group)
                          - group_replicates(obj, scorer, "full_spectrum", group))
            low, high = interval(replicates)
            rows.append([scorer, group, f"{point:.4f}", f"{low:.4f}", f"{high:.4f}"])
    rows.sort(key=lambda r: -float(r[2]))
    return rows


def factorial_rows(obj, protocol):
    """The two main effects, the interaction, and the four simple effects."""
    def cell(name, group, which):
        return (group_point(obj, name, protocol, group) if which == "point"
                else group_replicates(obj, name, protocol, group))

    definitions = {
        "covariance_main_effect": lambda g, w: (
            (cell("marginal_full", g, w) + cell("class_conditional_full", g, w)) / 2
            - (cell("marginal_diagonal", g, w) + cell("class_conditional_diagonal", g, w)) / 2),
        "label_main_effect": lambda g, w: (
            (cell("class_conditional_diagonal", g, w) + cell("class_conditional_full", g, w)) / 2
            - (cell("marginal_diagonal", g, w) + cell("marginal_full", g, w)) / 2),
        "interaction": lambda g, w: (
            (cell("class_conditional_full", g, w) - cell("class_conditional_diagonal", g, w))
            - (cell("marginal_full", g, w) - cell("marginal_diagonal", g, w))),
        "simple_full_minus_diagonal_within_marginal": lambda g, w: (
            cell("marginal_full", g, w) - cell("marginal_diagonal", g, w)),
        "simple_full_minus_diagonal_within_class_conditional": lambda g, w: (
            cell("class_conditional_full", g, w) - cell("class_conditional_diagonal", g, w)),
        "simple_class_conditional_minus_marginal_within_diagonal": lambda g, w: (
            cell("class_conditional_diagonal", g, w) - cell("marginal_diagonal", g, w)),
        "simple_class_conditional_minus_marginal_within_full": lambda g, w: (
            cell("class_conditional_full", g, w) - cell("marginal_full", g, w)),
    }
    rows = []
    for label, fn in definitions.items():
        for group in GROUPS:
            point = fn(group, "point")
            low, high = interval(fn(group, "replicates"))
            spans_zero = "yes" if low <= 0.0 <= high else "no"
            rows.append([protocol, label, group, f"{point:.4f}", f"{low:.4f}", f"{high:.4f}", spans_zero])
    return rows


def other_contrast_rows(obj, protocol):
    pairs = [
        ("rmd_pp_minus_rmd", "rmd_pp", "rmd"),
        ("knn_normalized_minus_unnormalized", "knn_normalized", "knn_unnormalized"),
        ("pca_residual_all_id_minus_marginal_diagonal", "pca_residual_all_id", "marginal_diagonal"),
    ]
    rows = []
    for label, left, right in pairs:
        for group in GROUPS:
            point = group_point(obj, left, protocol, group) - group_point(obj, right, protocol, group)
            low, high = interval(group_replicates(obj, left, protocol, group)
                                 - group_replicates(obj, right, protocol, group))
            spans_zero = "yes" if low <= 0.0 <= high else "no"
            rows.append([protocol, label, group, f"{point:.4f}", f"{low:.4f}", f"{high:.4f}", spans_zero])
    return rows


def main():
    standard = load(STANDARD_RUN / "bootstrap" / "bootstrap_singleton.pkl")
    full_spectrum = load(FULL_SPECTRUM_RUN / "bootstrap" / "bootstrap_singleton.pkl")

    print(f"standard run   : groups {standard.ood_groups}, B={standard.n_replicates}, seed={standard.seed}")
    print(f"full-spectrum  : groups {full_spectrum.ood_groups}, B={full_spectrum.n_replicates}, "
          f"seed={full_spectrum.seed}, protocols {full_spectrum.protocols}")
    pin_against_gated_table(standard)
    pin_against_gated_table(full_spectrum)

    write_rows(
        "degradation-dataset-averaged.csv",
        ["scorer", "group", "auroc_lost", "ci_low", "ci_high"],
        degradation_rows(full_spectrum),
    )

    contrasts = factorial_rows(standard, "standard") + other_contrast_rows(standard, "standard")
    contrasts += factorial_rows(full_spectrum, "full_spectrum") + other_contrast_rows(full_spectrum, "full_spectrum")
    write_rows(
        "factorial-and-pairwise-contrasts-dataset-averaged.csv",
        ["protocol", "contrast", "group", "point", "ci_low", "ci_high", "interval_spans_zero"],
        contrasts,
    )

    marginals = []
    for scorer in full_spectrum.scorers:
        if scorer.startswith("control_"):
            continue
        for protocol in full_spectrum.protocols:
            for group in GROUPS:
                for metric in ("auroc", "fpr95"):
                    arrays = [np.asarray(full_spectrum.values[(scorer, protocol, d, metric)], dtype=float)
                              for d in GROUPS[group]]
                    replicates = np.mean(arrays, axis=0) * 100.0
                    point = float(np.mean([full_spectrum.point[(scorer, protocol, d, metric)]
                                           for d in GROUPS[group]])) * 100.0
                    low, high = interval(replicates)
                    marginals.append([protocol, scorer, group, metric,
                                      f"{point:.4f}", f"{low:.4f}", f"{high:.4f}"])
    write_rows(
        "marginals-both-protocols-auroc-and-fpr95-dataset-averaged.csv",
        ["protocol", "scorer", "group", "metric", "value", "ci_low", "ci_high"],
        marginals,
    )


if __name__ == "__main__":
    main()
