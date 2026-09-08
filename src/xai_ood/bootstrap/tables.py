"""Every table and the manifest fragment: what leaves this package.

The top layer of the package's three-layer stack::

    plan.py   <-  run.py   <-  tables.py

**The 0-1 to 0-100 conversion happens here and only here**, in
``_display``, whose only callers are the three table emitters below. That
placement is the display-scale rule, and keeping it in the top layer is the
reason the split is drawn where it is: nothing under this module has any
business knowing about a display scale.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..metrics import metric_definitions
from ..schema import (
    RESULTS_COLUMNS,
    RESULTS_DTYPES,
    SPLIT_ROLES,
    encode_hyperparams,
    validate_results_frame,
)
from ..visualization.style import to_display_scale
from .plan import PROTOCOL_ID_SPLITS
from .run import QUANTILE_METHOD, BootstrapResult

__all__ = [
    "REQUIRED_PAIRWISE_CLAIMS",
    "FPR95_FULL_SPECTRUM_CAVEAT",
    "DIFFERENCE_COLUMNS",
    "DEGRADATION_COLUMNS",
    "results_table",
    "differences_table",
    "degradation_table",
    "bootstrap_manifest",
]


# --------------------------------------------------------------------------- #
# Fixed parameters. None of these is a knob.
# --------------------------------------------------------------------------- #

#: Every pairwise claim the thesis makes, as pairs of configuration names. The
#: reported quantity for each is the percentile interval of the
#: within-replicate difference. Enumerated here so
#: ``differences_table`` can be asked for "the required set" and raise on a
#: missing scorer, rather than a claim resting on nothing but a pair of
#: marginal CIs.
#:
#: The four 2x2 cells give six pairs (all of them, since the factorial's point
#: is which factor moved the number), then the two normalization ablations, then
#: the PCA-residual components against the naive marginal-diagonal cell -- the
#: comparative baseline declared for it. Cross-scorer
#: *degradation* comparisons are not here: they are differences of differences
#: and come from ``BootstrapResult.degradation_difference``.
REQUIRED_PAIRWISE_CLAIMS: tuple[tuple[str, str], ...] = (
    # the 2x2 factorial, all six pairs
    ("marginal_full", "marginal_diagonal"),
    ("class_conditional_diagonal", "marginal_diagonal"),
    ("class_conditional_full", "marginal_diagonal"),
    ("class_conditional_diagonal", "marginal_full"),
    ("class_conditional_full", "marginal_full"),
    ("class_conditional_full", "class_conditional_diagonal"),
    # normalization ablations
    ("rmd_pp", "rmd"),
    ("knn_normalized", "knn_unnormalized"),
    # PCA-residual against its stated comparative baseline
    ("pca_residual_all_id", "marginal_diagonal"),
    ("pca_residual_class_mean", "marginal_diagonal"),
)

#: The FPR@95 caveat under the full-spectrum protocol. Carried in the manifest
#: and in the table's ``caveat`` column, because the number is the thing that
#: travels and it will be quoted without its surrounding prose.
FPR95_FULL_SPECTRUM_CAVEAT: str = (
    "Under the full-spectrum protocol the 95%-TPR operating point is set almost "
    "entirely by cs-ID mass (roughly 94% of the ID side), so FPR@95 degradation "
    "is not decomposable the way AUROC degradation is and is not presented as a "
    "per-scorer robustness measure on its own."
)

#: Columns of ``differences_table``. Deliberately *not* the frozen results
#: schema: that schema has one row per (configuration x protocol x OOD group)
#: evaluation and no room for a pair. The difference intervals are a different
#: object and get a different table rather than a widened one.
DIFFERENCE_COLUMNS: tuple[str, ...] = (
    "comparison",
    "scorer_a",
    "scorer_b",
    "protocol",
    "ood_group",
    "metric",
    "delta",
    "delta_ci_low",
    "delta_ci_high",
    "excludes_zero",
    "level",
    "n_replicates",
    "seed",
)

#: Columns of ``degradation_table``.
DEGRADATION_COLUMNS: tuple[str, ...] = (
    "scorer",
    "ood_group",
    "metric",
    "standard",
    "full_spectrum",
    "degradation",
    "degradation_ci_low",
    "degradation_ci_high",
    "excludes_zero",
    "level",
    "n_replicates",
    "seed",
    "caveat",
)


# --------------------------------------------------------------------------- #
# 6. Tables
# --------------------------------------------------------------------------- #


def _display(value: float) -> float:
    """**The one 0-1 to 0-100 conversion point in the bootstrap module.**

    Stated once in ``visualization/style.py``: AUROC and FPR@95 are on
    the 0-100 scale project-wide, and ``to_display_scale`` converts at exactly
    one point -- where a metric enters the results table -- and never again
    downstream. This is that point. ``xai_ood.metrics`` returns fractions,
    ``Interval`` and ``BootstrapResult`` hold fractions, and the three
    table emitters below are the only callers of this function.

    A second conversion elsewhere would produce an AUROC of 9,846, which the
    schema validator catches, or a silently doubled FPR, which it does not.
    """
    return float(to_display_scale(value))


def results_table(
    result: BootstrapResult,
    *,
    seed: int,
    repo_commit: str,
    openood_commit: str,
    timestamp: str,
    variants: Mapping[str, str] | None = None,
    hyperparams: Mapping[str, Mapping[str, Any]] | None = None,
    datasets: Mapping[str, str] | None = None,
    scorers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """The headline table, under the frozen schema, with marginal CIs attached.

    One row per (scorer x protocol x OOD group). ``auroc_ci_low`` and
    ``auroc_ci_high`` are the **marginal** percentile interval, present for
    readability only. Overlap between two of them is **not** the test; the
    paired differences that are live in ``differences_table``, and
    ``Interval.excludes_zero`` raises rather than answering for a marginal
    interval.

    ``seed``, ``repo_commit``, ``openood_commit`` and ``timestamp`` are required
    keyword arguments, following ``methods/manifest.py``: a table that looks
    complete and silently carries no provenance is worse than one that refuses
    to be built. ``timestamp`` is passed in rather than generated so this module
    needs no ``datetime`` import -- ``tests/test_import_hygiene.py``'s allowed
    set is a deliberate list and widening it for a convenience is not a trade
    worth making.

    Note the frozen schema has CI columns for AUROC only. FPR@95's interval is
    not dropped, it simply does not belong in a 14-column table that was frozen
    before it existed; it is available from ``BootstrapResult.marginal`` and
    appears in the degradation table.
    """
    variants = dict(variants or {})
    hyperparams = dict(hyperparams or {})
    datasets = dict(datasets or {})
    names = tuple(scorers) if scorers is not None else result.scorers

    rows: list[dict[str, Any]] = []
    for scorer in names:
        for protocol in result.protocols:
            for group in result.ood_groups:
                if group not in SPLIT_ROLES:
                    raise ValueError(
                        f"OOD group {group!r} is not a split_role in the frozen "
                        f"schema {list(SPLIT_ROLES)}; the results table's "
                        f"split_role column is enumerated."
                    )
                auroc_interval = result.marginal(scorer, protocol, group, "auroc")
                fpr = result.point[(scorer, protocol, group, "fpr95")]
                rows.append(
                    {
                        "dataset": datasets.get(
                            group, "+".join(result.plan_diagnostics["ood_groups"][group])
                        ),
                        "split_role": group,
                        "scorer": scorer,
                        "variant": variants.get(scorer, pd.NA),
                        "hyperparams": encode_hyperparams(hyperparams.get(scorer)),
                        "protocol": protocol,
                        "auroc": _display(auroc_interval.point),
                        "auroc_ci_low": _display(auroc_interval.low),
                        "auroc_ci_high": _display(auroc_interval.high),
                        "fpr95": _display(fpr),
                        "seed": seed,
                        "repo_commit": repo_commit,
                        "openood_commit": openood_commit,
                        "timestamp": timestamp,
                    }
                )

    frame = pd.DataFrame(rows, columns=list(RESULTS_COLUMNS))
    frame = frame.astype({c: RESULTS_DTYPES[c] for c in RESULTS_COLUMNS})
    validate_results_frame(frame)
    return frame


def differences_table(
    result: BootstrapResult,
    *,
    pairs: Iterable[tuple[str, str]] | None = None,
    protocols: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    require_all: bool = True,
) -> pd.DataFrame:
    """The paired differences, one row per claim.

    ``pairs`` defaults to ``REQUIRED_PAIRWISE_CLAIMS``. With
    ``require_all=True`` (the default) a pair naming a scorer the run does not
    have raises, rather than being skipped: the failure mode this prevents is a
    claim resting on two marginal CIs because its difference row was quietly
    absent from the table.
    """
    wanted = list(pairs) if pairs is not None else list(REQUIRED_PAIRWISE_CLAIMS)
    protocols = tuple(protocols) if protocols is not None else result.protocols
    groups = tuple(groups) if groups is not None else result.ood_groups
    metrics = tuple(metrics) if metrics is not None else result.metrics

    known = set(result.scorers)
    missing = sorted(
        {s for pair in wanted for s in pair if s not in known}
    )
    if missing and require_all:
        raise ValueError(
            f"paired differences were requested for scorer(s) {missing}, which "
            f"this run does not have. Every pairwise claim the thesis makes "
            f"needs a difference interval; dropping the row silently would "
            f"leave the claim resting on marginal CIs. Pass "
            f"require_all=False if this run deliberately covers a subset."
        )
    wanted = [p for p in wanted if p[0] in known and p[1] in known]

    rows: list[dict[str, Any]] = []
    for scorer_a, scorer_b in wanted:
        for protocol in protocols:
            for group in groups:
                for metric in metrics:
                    interval = result.difference(
                        scorer_a, scorer_b, protocol, group, metric
                    )
                    rows.append(
                        {
                            "comparison": f"{scorer_a} - {scorer_b}",
                            "scorer_a": scorer_a,
                            "scorer_b": scorer_b,
                            "protocol": protocol,
                            "ood_group": group,
                            "metric": metric,
                            "delta": _display(interval.point),
                            "delta_ci_low": _display(interval.low),
                            "delta_ci_high": _display(interval.high),
                            "excludes_zero": interval.excludes_zero,
                            "level": result.level,
                            "n_replicates": result.n_replicates,
                            "seed": result.seed,
                        }
                    )
    return pd.DataFrame(rows, columns=list(DIFFERENCE_COLUMNS))


def degradation_table(
    result: BootstrapResult,
    *,
    groups: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    scorers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-scorer standard-to-full-spectrum drop, paired by photograph.

    This is the covariate-shift result. ``degradation = standard -
    full_spectrum`` for every metric, with the sign reading given in
    ``BootstrapResult.degradation_replicates``, and the FPR@95 rows carry
    ``FPR95_FULL_SPECTRUM_CAVEAT`` in the ``caveat`` column so the number
    does not travel without it.
    """
    if not {"standard", "full_spectrum"} <= set(result.protocols):
        raise ValueError(
            f"degradation needs both protocols; this run has "
            f"{list(result.protocols)}"
        )
    names = tuple(scorers) if scorers is not None else result.scorers
    groups = tuple(groups) if groups is not None else result.ood_groups
    metrics = tuple(metrics) if metrics is not None else result.metrics

    rows: list[dict[str, Any]] = []
    for scorer in names:
        for group in groups:
            for metric in metrics:
                interval = result.degradation(scorer, group, metric)
                rows.append(
                    {
                        "scorer": scorer,
                        "ood_group": group,
                        "metric": metric,
                        "standard": _display(
                            result.point[(scorer, "standard", group, metric)]
                        ),
                        "full_spectrum": _display(
                            result.point[(scorer, "full_spectrum", group, metric)]
                        ),
                        "degradation": _display(interval.point),
                        "degradation_ci_low": _display(interval.low),
                        "degradation_ci_high": _display(interval.high),
                        "excludes_zero": interval.excludes_zero,
                        "level": result.level,
                        "n_replicates": result.n_replicates,
                        "seed": result.seed,
                        "caveat": (
                            FPR95_FULL_SPECTRUM_CAVEAT if metric == "fpr95" else ""
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=list(DEGRADATION_COLUMNS))

# --------------------------------------------------------------------------- #
# 7. Manifest
# --------------------------------------------------------------------------- #


def bootstrap_manifest(result: BootstrapResult) -> dict[str, Any]:
    """The manifest fragment for a bootstrap run.

    The seed and the index sets are recorded so reruns reproduce the same
    pairing. The **seed plus the procedure** is what
    reproduces the index sets; the sets themselves are one array of ~300,000
    int64 per replicate and are not written. What is recorded is everything
    needed to rebuild them: the seed, the replicate count, the cluster
    universe sizes, which
    splits each protocol treats as ID, and which datasets each OOD group is
    composed of. ``tests/test_bootstrap.py`` pins that a rerun from the recorded
    seed reproduces the replicate arrays bit for bit, which is the property this
    fragment is claiming.
    """
    return {
        "procedure": "cluster_bootstrap_over_source_photographs",
        "seed": result.seed,
        "n_replicates": result.n_replicates,
        "interval": "percentile",
        "bca": False,
        "level": result.level,
        "quantile_method": QUANTILE_METHOD,
        "resamples_shared_across_scorers": True,
        "comparisons": "within-replicate paired differences; CI overlap is not the test",
        "degradation_pairing": "same drawn ids on both protocol legs",
        "protocol_id_splits": {k: list(v) for k, v in PROTOCOL_ID_SPLITS.items()},
        "scorers": list(result.scorers),
        "protocols": list(result.protocols),
        "ood_groups": {
            g: list(result.plan_diagnostics["ood_groups"][g])
            for g in result.ood_groups
        },
        "metrics": list(result.metrics),
        "metric_definitions": metric_definitions(result.tpr_target),
        "fpr95_full_spectrum_caveat": FPR95_FULL_SPECTRUM_CAVEAT,
        "plan": dict(result.plan_diagnostics),
    }
