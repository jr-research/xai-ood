"""AUROC and FPR@95TPR, defined exactly once for the whole project.

The bootstrap needs both *inside* every replicate, and the run needs the same
AUROC to cross-check against OpenOOD's own on an identical score array. Two
implementations of one metric is the failure this
package avoids everywhere else -- one ``l2_normalize``, one ``score_frame``, one
rank correlation -- so there is one here too.

**Scale.** Everything in this module returns a raw fraction in ``[0, 1]``.
The 0-100 display scale is applied at exactly one point, where a metric enters
the results table, by ``xai_ood.visualization.style.to_display_scale``.
``xai_ood.bootstrap`` is where that point is; nothing here scales anything.

**Sign convention.** Every scorer in ``xai_ood.methods`` returns
higher = more OOD, so ``auroc(id_scores, ood_scores)`` is
``P(score_OOD > score_ID) + 0.5 P(tie)`` and higher is better detection. There is
no ``higher_is_ood`` flag, for the same reason ``xai_ood.methods.dump.Scorer``
has none: the rule is a package invariant, not a per-call setting.

One of the two rules below is the standard definition of its statistic and the
other is a real choice this package makes. Both are stated explicitly, because
two defensible implementations can disagree in the third decimal and the
difference would look like a real result.

* **Ties get half credit** in AUROC. This is the standard convention, not a
  local one: it is what ``sklearn.metrics.roc_auc_score`` computes, it is the
  Mann-Whitney U statistic with mid-ranks, and it is what the trapezoidal ROC
  integral gives. ``tests/test_metrics.py`` pins the equality against
  scikit-learn at ``rel=1e-12``, including a tie-heavy case. With continuous
  embedding distances exact ties are rare, but a diagonal-covariance score on
  quantised inputs can produce them, and "rare" is not "never".
* **The 95%-TPR threshold is an observed ID score**, specifically the
  ``ceil(0.95 * n_id)``-th smallest, which is the *smallest* observed score at
  which at least 95% of ID samples are still accepted as ID. This one is a
  choice. The alternative is to interpolate between the two order statistics
  bracketing the target, which hits the target TPR exactly at the cost of a
  threshold no sample achieved. The declared procedure requires *that* the
  threshold be recomputed inside each replicate but says nothing about *how* it
  is read off a finite sample, so this rule is not derived from it. Whether
  OpenOOD reads the same threshold from an identical score array is what the
  OpenOOD cross-check measures; that number lives in the cross-check record,
  and this module does not carry a copy of it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "TPR_TARGET",
    "AUROC_TIE_CONVENTION",
    "THRESHOLD_CONVENTION",
    "as_scores",
    "auroc",
    "auroc_against_sorted_id",
    "threshold_at_tpr",
    "fpr_at_tpr",
    "fpr_against_sorted_id",
    "evaluate",
    "metric_definitions",
    "METRICS",
]

#: The operating point: the target fraction of ID samples still accepted as ID.
#: 0.95 is the point the OOD detection literature reports FPR at, which is why
#: the metric is named FPR@95TPR. It is a target, not an achieved value; see
#: ``evaluate``, which reports the TPR actually reached alongside the FPR.
TPR_TARGET: float = 0.95

#: Recorded in the manifest so a reader knows which of the two defensible tie
#: conventions produced the numbers.
AUROC_TIE_CONVENTION: str = "half_credit_for_ties (Mann-Whitney U with mid-ranks)"

#: Likewise for the threshold.
THRESHOLD_CONVENTION: str = (
    "smallest observed ID score with empirical TPR >= target; "
    "order statistic ceil(target * n_id); no interpolation"
)

#: The two metrics reported per configuration, and the set ``bootstrap.run``
#: validates a caller's ``metrics=`` argument against. It does not bind the
#: results table: ``bootstrap.tables`` and ``schema.RESULTS_COLUMNS`` both spell
#: the two names out as literals, so this constant is a guard on the run rather
#: than a single source the table is derived from.
METRICS: tuple[str, ...] = ("auroc", "fpr95")


def as_scores(values: Any, *, name: str) -> np.ndarray:
    """Coerce to a finite 1-D float64 score array, or raise.

    NaN is rejected rather than propagated. ``np.sort`` puts NaN last and
    ``np.searchsorted`` against a NaN-terminated array returns silently wrong
    counts, so a single NaN score would move an AUROC without raising anywhere.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(
            f"{name}: empty score array. AUROC and FPR@95 are undefined with no "
            f"samples on one side."
        )
    if not np.isfinite(array).all():
        n_bad = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"{name}: {n_bad} non-finite score(s). NaN sorts last and would make "
            f"searchsorted return counts that are wrong without raising."
        )
    return array


# --------------------------------------------------------------------------- #
# AUROC
# --------------------------------------------------------------------------- #


def auroc_against_sorted_id(id_sorted: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC given an **already sorted** ID score array. Returns a fraction.

    Split out from ``auroc`` because the bootstrap calls it with one sorted
    ID array against several OOD groups: one index set per replicate is applied
    to every scorer, so the ID resample is shared across groups within a
    replicate, and sorting it once per (replicate, protocol, scorer) rather than
    once per group is the difference between a comfortable run and an
    uncomfortable one, and more so the larger the replicate count.

    ``searchsorted`` with ``side="left"`` counts ID scores strictly below an OOD
    score; with ``side="right"`` it counts those at or below. The difference is
    the tie count, which gets half credit.
    """
    n_id = int(id_sorted.size)
    n_ood = int(ood_scores.size)
    if n_id == 0 or n_ood == 0:
        raise ValueError("AUROC needs at least one sample on each side")
    below = np.searchsorted(id_sorted, ood_scores, side="left")
    at_or_below = np.searchsorted(id_sorted, ood_scores, side="right")
    wins = below.sum() + 0.5 * (at_or_below - below).sum()
    return float(wins / (n_id * n_ood))


def auroc(id_scores: Any, ood_scores: Any) -> float:
    """``P(score_OOD > score_ID) + 0.5 P(tie)``, a fraction in ``[0, 1]``.

    Higher = better OOD detection, given the package's higher = more OOD scoring
    convention. Equivalent to ``sklearn.metrics.roc_auc_score`` with OOD as the
    positive class; ``tests/test_metrics.py`` pins that equivalence behind
    ``pytest.importorskip`` so it un-skips wherever scikit-learn is installed.
    """
    id_sorted = np.sort(as_scores(id_scores, name="id_scores"))
    return auroc_against_sorted_id(id_sorted, as_scores(ood_scores, name="ood_scores"))


# --------------------------------------------------------------------------- #
# FPR@95TPR
# --------------------------------------------------------------------------- #


def threshold_at_tpr(id_sorted: np.ndarray, tpr: float = TPR_TARGET) -> float:
    """The score threshold accepting at least ``tpr`` of a **sorted** ID array.

    Returns an observed ID score, never an interpolated one: the
    ``ceil(tpr * n_id)``-th smallest. Everything at or below it is classified
    ID, so the achieved TPR is at least ``ceil(tpr * n_id) / n_id``, which is
    itself at least ``tpr``. With distinct ID scores that lower bound is the
    achieved value, and is the smallest attainable one that clears the target.
    With a tied block spanning the threshold the whole block is accepted and the
    achieved TPR is higher, which is why ``evaluate`` counts it from the data
    rather than computing it from the rank.

    The declared procedure requires this to be **recomputed inside each
    replicate** from that replicate's ID scores rather than fixed from the full
    sample, "otherwise the interval conditions on a quantity that is itself
    estimated". That is a property of the caller, not of this function, and
    ``xai_ood.bootstrap`` is where it is enforced -- but it is the reason
    this takes an ID array rather than a precomputed threshold.
    """
    if not 0.0 < tpr <= 1.0:
        raise ValueError(f"tpr must be in (0, 1]; got {tpr}")
    n_id = int(id_sorted.size)
    if n_id == 0:
        raise ValueError("cannot read a threshold off an empty ID array")
    rank = int(np.ceil(tpr * n_id))
    rank = min(max(rank, 1), n_id)
    return float(id_sorted[rank - 1])


def fpr_against_sorted_id(
    id_sorted: np.ndarray, ood_scores: np.ndarray, tpr: float = TPR_TARGET
) -> float:
    """FPR at the ``tpr`` operating point, given an already sorted ID array."""
    tau = threshold_at_tpr(id_sorted, tpr)
    n_ood = int(ood_scores.size)
    if n_ood == 0:
        raise ValueError("FPR needs at least one OOD sample")
    return float(np.count_nonzero(ood_scores <= tau) / n_ood)


def fpr_at_tpr(id_scores: Any, ood_scores: Any, tpr: float = TPR_TARGET) -> float:
    """Fraction of OOD samples accepted as ID at the ``tpr`` operating point.

    Lower is better, unlike every other number in this module. The results table
    keeps them in separate columns for exactly that reason.
    """
    id_sorted = np.sort(as_scores(id_scores, name="id_scores"))
    return fpr_against_sorted_id(
        id_sorted, as_scores(ood_scores, name="ood_scores"), tpr
    )


# --------------------------------------------------------------------------- #
# Both metrics at once, plus the operating point actually reached
# --------------------------------------------------------------------------- #


def evaluate(
    id_scores: Any, ood_scores: Any, *, tpr: float = TPR_TARGET
) -> dict[str, float]:
    """Both metrics plus the diagnostics that say what the operating point was.

    ``achieved_tpr`` is not decoration. At small *n* the 95% target is not
    exactly attainable -- with 37 ID samples the threshold accepts 36 of them,
    a TPR of 0.973 -- and an FPR quoted against an unstated 0.973 operating
    point is a different number from one quoted against 0.95. Inside a bootstrap
    replicate the effective *n* varies too, since a cluster resample does not
    draw a fixed row count.
    """
    id_sorted = np.sort(as_scores(id_scores, name="id_scores"))
    ood = as_scores(ood_scores, name="ood_scores")
    tau = threshold_at_tpr(id_sorted, tpr)
    return {
        "auroc": auroc_against_sorted_id(id_sorted, ood),
        "fpr95": fpr_against_sorted_id(id_sorted, ood, tpr),
        "threshold": tau,
        "achieved_tpr": float(np.count_nonzero(id_sorted <= tau) / id_sorted.size),
        "n_id": int(id_sorted.size),
        "n_ood": int(ood.size),
    }


def metric_definitions(tpr: float = TPR_TARGET) -> dict[str, Any]:
    """Manifest fragment: what these numbers mean, in the artefact itself.

    A results table whose AUROC convention lives only in a docstring is a table
    that cannot be defended later.
    """
    return {
        "scale": "fraction_0_1 (converted to 0-100 once, at the results table)",
        "sign_convention": "higher score = more OOD",
        "auroc": "P(score_OOD > score_ID) + 0.5 P(tie)",
        "auroc_ties": AUROC_TIE_CONVENTION,
        "fpr_at_tpr_target": float(tpr),
        "fpr_threshold_rule": THRESHOLD_CONVENTION,
        "fpr_threshold_recomputed_per_replicate": True,
    }
