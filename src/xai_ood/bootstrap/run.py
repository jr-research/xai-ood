"""The intervals, the result, and the loop.

The middle layer of the package's three-layer stack::

    plan.py   <-  run.py   <-  tables.py

Imports from ``xai_ood.bootstrap.plan``, and is imported by
``xai_ood.bootstrap.tables``.

Everything here is on the raw 0-1 scale, because that is what
``xai_ood.metrics`` returns. The 0-100 conversion lives in ``tables.py`` and
happens at exactly one point there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..metrics import (
    METRICS,
    TPR_TARGET,
    auroc_against_sorted_id,
    fpr_against_sorted_id,
)
from ..schema import CANONICAL_SORT_KEY, assert_aligned
from .plan import BootstrapPlan, _key_digest

__all__ = [
    "B_DEFAULT",
    "CI_LEVEL",
    "QUANTILE_METHOD",
    "Interval",
    "BootstrapResult",
    "percentile_interval",
    "replicate_indices",
    "run_bootstrap",
]


# --------------------------------------------------------------------------- #
# Fixed parameters. None of these is a knob.
# --------------------------------------------------------------------------- #


#: The default replicate count. Not "roughly 1,000"; 1,000.
#:
#: The count an analysis actually uses is a ``run_bootstrap`` argument and is
#: recorded in the manifest, so a quoted interval always states the count
#: behind it rather than inheriting this one silently. Raising it is cheap and
#: never invalidates a scoring run: the bootstrap consumes cached per-sample
#: scores, so a larger count costs CPU and recomputes no score.
B_DEFAULT: int = 1_000

#: 95% percentile intervals. BCa is explicitly not used -- percentile intervals
#: are adequate here and simpler to state, which is the whole reason the
#: procedure was pinned before any interval was computed.
CI_LEVEL: float = 0.95

#: numpy's default linear interpolation between order statistics. Named rather
#: than left implicit because the 2.5% point rarely lands on an order statistic:
#: at B = 1,000 it falls between the 25th and 26th, at B = 5,000 between the
#: 125th and 126th, and "percentile interval" does not by itself say which of
#: the nine conventional definitions is meant.
QUANTILE_METHOD: str = "linear"


# --------------------------------------------------------------------------- #
# 3. Intervals
# --------------------------------------------------------------------------- #


def percentile_interval(
    values: np.ndarray, level: float = CI_LEVEL
) -> tuple[float, float]:
    """The two-sided ``level`` percentile interval of a replicate array.

    Percentile, not BCa. The tails are ``(1 - level) / 2`` each.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot take a percentile interval of no replicates")
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(array, [alpha, 1.0 - alpha], method=QUANTILE_METHOD)
    return float(low), float(high)


@dataclass(frozen=True)
class Interval:
    """A point estimate and its percentile interval, on the raw 0-1 scale.

    ``kind`` is what stops the no-CI-overlap rule from being advice. A
    ``"marginal"`` interval
    is the per-scorer CI that appears in the results table for readability; a
    ``"difference"`` interval is the within-replicate paired difference that is
    the actual test. Asking a marginal interval whether it excludes zero raises,
    because the only reason to ask is to compare two of them by overlap.
    """

    label: str
    kind: str
    point: float
    low: float
    high: float
    level: float
    n_replicates: int

    def __post_init__(self) -> None:
        if self.kind not in ("marginal", "difference"):
            raise ValueError(f"unknown interval kind {self.kind!r}")

    @property
    def width(self) -> float:
        return float(self.high - self.low)

    @property
    def excludes_zero(self) -> bool:
        """Whether the difference interval excludes zero, which is significance.

        Raises on a marginal interval. This is deliberate and it is the whole
        point of the ``kind`` field: because the cells share test images and are
        positively correlated, ``Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)`` is
        strictly smaller than the overlap heuristic assumes, so reading two
        marginal CIs for overlap understates real differences in exactly the
        comparison the factorial exists to make.
        """
        if self.kind != "difference":
            raise ValueError(
                f"{self.label}: this is a marginal interval, and "
                f"marginal-CI overlap is explicitly NOT the test. "
                f"Significance is 'the *difference* interval "
                f"excludes zero'. Use BootstrapResult.difference(), "
                f".degradation() or .degradation_difference() instead."
            )
        return not (self.low <= 0.0 <= self.high)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "label": self.label,
            "kind": self.kind,
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "n_replicates": self.n_replicates,
        }
        if self.kind == "difference":
            out["excludes_zero"] = self.excludes_zero
        return out

# --------------------------------------------------------------------------- #
# 4. The result
# --------------------------------------------------------------------------- #

Key = tuple[str, str, str, str]  # (scorer, protocol, ood_group, metric)


@dataclass(frozen=True)
class BootstrapResult:
    """Per-replicate metric values, plus the full-sample point estimates.

    ``values[(scorer, protocol, group, metric)]`` is a ``(B,)`` array;
    ``point[...]`` is the same metric on the unresampled data. Point estimates
    come from the data, never from the mean of the replicates -- the percentile
    interval is placed around the estimate, not around the bootstrap mean.

    Everything is on the raw 0-1 scale. Only the table emitters convert.
    """

    values: Mapping[Key, np.ndarray]
    point: Mapping[Key, float]
    scorers: tuple[str, ...]
    protocols: tuple[str, ...]
    ood_groups: tuple[str, ...]
    metrics: tuple[str, ...]
    seed: int
    n_replicates: int
    level: float
    tpr_target: float
    plan_diagnostics: Mapping[str, Any]

    # -- lookup ------------------------------------------------------------- #

    def _key(self, scorer: str, protocol: str, group: str, metric: str) -> Key:
        key = (scorer, protocol, group, metric)
        if key not in self.values:
            raise KeyError(
                f"no replicates for {key}. Known scorers {list(self.scorers)}, "
                f"protocols {list(self.protocols)}, groups {list(self.ood_groups)}, "
                f"metrics {list(self.metrics)}."
            )
        return key

    def replicates(
        self, scorer: str, protocol: str, group: str, metric: str = "auroc"
    ) -> np.ndarray:
        """The ``(B,)`` array of replicate values. Raw 0-1 scale."""
        return self.values[self._key(scorer, protocol, group, metric)]

    # -- marginal ----------------------------------------------------------- #

    def marginal(
        self, scorer: str, protocol: str, group: str, metric: str = "auroc"
    ) -> Interval:
        """Per-scorer CI, for readability in the results table.

        **Not** the test for any pairwise claim; see ``difference`` and the
        note on ``Interval.excludes_zero``.
        """
        key = self._key(scorer, protocol, group, metric)
        low, high = percentile_interval(self.values[key], self.level)
        return Interval(
            label=f"{scorer} [{protocol}/{group}] {metric}",
            kind="marginal",
            point=float(self.point[key]),
            low=low,
            high=high,
            level=self.level,
            n_replicates=self.n_replicates,
        )

    # -- paired differences -------------------------------------------------- #

    def difference(
        self,
        scorer_a: str,
        scorer_b: str,
        protocol: str,
        group: str,
        metric: str = "auroc",
    ) -> Interval:
        """The within-replicate difference ``A - B``, and its interval.

        ``delta_b = M_b(A) - M_b(B)`` computed inside each replicate *b* from
        the *same* resample, then the percentile interval of ``{delta_b}``.
        Significance is ``excludes_zero``.
        """
        key_a = self._key(scorer_a, protocol, group, metric)
        key_b = self._key(scorer_b, protocol, group, metric)
        deltas = self.values[key_a] - self.values[key_b]
        low, high = percentile_interval(deltas, self.level)
        return Interval(
            label=f"{scorer_a} - {scorer_b} [{protocol}/{group}] {metric}",
            kind="difference",
            point=float(self.point[key_a] - self.point[key_b]),
            low=low,
            high=high,
            level=self.level,
            n_replicates=self.n_replicates,
        )

    # -- degradation --------------------------------------------------------- #

    def degradation_replicates(
        self, scorer: str, group: str, metric: str = "auroc"
    ) -> np.ndarray:
        """``standard - full_spectrum``, per replicate, from the same drawn ids.

        One sign rule for every metric rather than a per-metric one, because a
        metric-dependent sign convention is its own bug source. For AUROC a
        positive value is the degradation the covariate-shift result is about.
        For FPR@95, where lower is better, a positive value means the operating
        point got *better* under full-spectrum -- and see
        ``FPR95_FULL_SPECTRUM_CAVEAT`` before reading anything into it.
        """
        standard = self.values[self._key(scorer, "standard", group, metric)]
        full = self.values[self._key(scorer, "full_spectrum", group, metric)]
        return standard - full

    def degradation(
        self, scorer: str, group: str, metric: str = "auroc"
    ) -> Interval:
        """Interval of the paired standard-to-full-spectrum drop."""
        deltas = self.degradation_replicates(scorer, group, metric)
        low, high = percentile_interval(deltas, self.level)
        point = float(
            self.point[self._key(scorer, "standard", group, metric)]
            - self.point[self._key(scorer, "full_spectrum", group, metric)]
        )
        return Interval(
            label=f"degradation {scorer} [{group}] {metric}",
            kind="difference",
            point=point,
            low=low,
            high=high,
            level=self.level,
            n_replicates=self.n_replicates,
        )

    def degradation_difference(
        self, scorer_a: str, scorer_b: str, group: str, metric: str = "auroc"
    ) -> Interval:
        """Cross-scorer degradation: ``delta(A) - delta(B)``, within replicate.

        Cross-scorer comparisons of the degradation are themselves paired
        differences, so this is a difference of differences and all four AUROCs
        come from one resample.
        """
        deltas = self.degradation_replicates(
            scorer_a, group, metric
        ) - self.degradation_replicates(scorer_b, group, metric)
        low, high = percentile_interval(deltas, self.level)
        point = self.degradation(scorer_a, group, metric).point - self.degradation(
            scorer_b, group, metric
        ).point
        return Interval(
            label=f"degradation {scorer_a} - {scorer_b} [{group}] {metric}",
            kind="difference",
            point=float(point),
            low=low,
            high=high,
            level=self.level,
            n_replicates=self.n_replicates,
        )

# --------------------------------------------------------------------------- #
# 5. The run
# --------------------------------------------------------------------------- #


def _normalise_frames(
    frames: pd.DataFrame | Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    name: str,
) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(frames, pd.DataFrame):
        return [(name, frames)]
    if isinstance(frames, Mapping):
        return [(str(k), v) for k, v in frames.items()]
    return [(f"{name}[{i}]", f) for i, f in enumerate(frames)]


def _score_columns(
    items: Sequence[tuple[str, pd.DataFrame]],
    scorers: Sequence[str] | None,
    name: str,
) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for frame_name, frame in items:
        for column in frame.columns:
            if column in CANONICAL_SORT_KEY:
                continue
            if scorers is not None and column not in scorers:
                continue
            if column in columns:
                raise ValueError(
                    f"{name}: score column {column!r} appears in more than one "
                    f"frame ({frame_name} among them). One scorer, one column: "
                    f"two columns of the same name would silently make the "
                    f"paired difference of a scorer with itself."
                )
            values = np.asarray(frame[column], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{name}: score column {column!r} in {frame_name} has "
                    f"{int((~np.isfinite(values)).sum())} non-finite value(s)."
                )
            columns[column] = values
    if not columns:
        raise ValueError(f"{name}: no score columns found in the input frame(s)")
    if scorers is not None:
        absent = [s for s in scorers if s not in columns]
        if absent:
            raise ValueError(f"{name}: requested scorer(s) {absent} are not present")
    return columns


def replicate_indices(
    plan: BootstrapPlan, rng: np.random.Generator, n_replicates: int
):
    """Yield ``(id_rows, ood_rows)`` row-position maps, one pair per replicate.

    The whole sampling procedure is the loop below, which is why it is a
    public generator rather than a loop body: they can be read on their own, and
    ``tests/test_bootstrap.py`` drives them directly to recompute every metric
    independently of ``run_bootstrap``.

    Per replicate:

    * one draw of ``|I|`` photograph indices with replacement, applied to
      **both** protocol indexes, which is what makes the degradation paired
      across protocols rather than two independent estimates;
    * one draw per OOD dataset, sized to that dataset, so groups keep their
      composition.

    Draw order is fixed -- ID first, then datasets in plan order -- because it
    is what the recorded seed reproduces.
    """
    n_id_clusters = plan.n_id_clusters
    protocols = plan.protocols
    ood_names = tuple(plan.ood_by_dataset)
    for _ in range(n_replicates):
        drawn_id = rng.integers(0, n_id_clusters, size=n_id_clusters)
        id_rows = {p: plan.id_by_protocol[p].gather(drawn_id) for p in protocols}
        drawn_ood = {
            d: rng.integers(
                0, plan.ood_by_dataset[d].n_clusters,
                size=plan.ood_by_dataset[d].n_clusters,
            )
            for d in ood_names
        }
        ood_rows = {
            group: np.concatenate(
                [plan.ood_by_dataset[d].gather(drawn_ood[d]) for d in datasets]
            )
            for group, datasets in plan.ood_groups.items()
        }
        yield id_rows, ood_rows


def run_bootstrap(
    frames: pd.DataFrame | Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    plan: BootstrapPlan,
    *,
    seed: int,
    n_replicates: int = B_DEFAULT,
    level: float = CI_LEVEL,
    scorers: Sequence[str] | None = None,
    metrics: Sequence[str] = METRICS,
    tpr: float = TPR_TARGET,
    progress: Any = None,
    name: str = "bootstrap",
) -> BootstrapResult:
    """Run the cluster bootstrap over every scorer, protocol and OOD group.

    ``frames`` is one per-sample score frame from
    ``xai_ood.methods.dump.score_frame``, or several -- a mapping of name to
    frame, or a sequence. Several frames go through
    ``xai_ood.schema.assert_aligned``, which is the check that matters here:
    per-frame ordering is necessary and not sufficient, since two frames can
    each be perfectly sorted and still cover different row sets, and one shared
    resample index would then misalign them without either per-frame check
    firing.

    ``seed`` is required and is recorded in the manifest. Every random draw in
    the procedure comes from ``numpy.random.default_rng(seed)`` in this
    function, in a fixed order, so the run reproduces exactly.

    The loop is replicates on the outside and scorers on the inside, which is
    one index set per replicate applied to every scorer's array, expressed as
    control flow rather than as a comment. Reversing the loops would make
    it possible for two scorers to see different resamples, and every paired
    difference in the thesis depends on their not doing so.
    """
    items = _normalise_frames(frames, name)
    assert_aligned([f for _, f in items], names=[n for n, _ in items])
    index = items[0][1]

    if len(index) != plan.n_rows:
        raise ValueError(
            f"{name}: the frames have {len(index)} rows but the plan was built "
            f"for {plan.n_rows}. Build the plan from the same index."
        )
    if _key_digest(index) != plan.key_digest:
        raise ValueError(
            f"{name}: the frames' canonical keys do not match the plan's. Same "
            f"row count, different rows -- a shared resample index would point "
            f"at the wrong photographs."
        )

    if n_replicates < 1:
        raise ValueError(f"{name}: n_replicates must be >= 1; got {n_replicates}")
    metrics = tuple(metrics)
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"{name}: unknown metric(s) {unknown}; expected {list(METRICS)}")

    columns = _score_columns(items, scorers, name)
    scorer_names = tuple(scorers) if scorers is not None else tuple(columns)
    protocols = plan.protocols
    groups = plan.groups

    keys: list[Key] = [
        (s, p, g, m)
        for s in scorer_names
        for p in protocols
        for g in groups
        for m in metrics
    ]
    values: dict[Key, np.ndarray] = {
        k: np.empty(n_replicates, dtype=np.float64) for k in keys
    }

    # Point estimates: the same metrics on the unresampled data. `positions`
    # covers exactly the rows of that protocol / group, so this is one code path
    # with the replicate loop rather than a second one that could disagree.
    point: dict[Key, float] = {}
    group_rows_full = plan.group_positions()
    for scorer in scorer_names:
        column = columns[scorer]
        for protocol in protocols:
            id_sorted = np.sort(column[plan.id_by_protocol[protocol].positions])
            for group in groups:
                ood = column[group_rows_full[group]]
                for metric in metrics:
                    point[(scorer, protocol, group, metric)] = _metric(
                        metric, id_sorted, ood, tpr
                    )

    rng = np.random.default_rng(seed)

    for b, (id_rows, ood_rows) in enumerate(
        replicate_indices(plan, rng, n_replicates)
    ):
        # One index set, every scorer.
        for scorer in scorer_names:
            column = columns[scorer]
            # The OOD side is sorted once per (replicate, scorer, group) and
            # reused across protocols. Sorting it is not required by the metric
            # and cannot change the metric's value -- AUROC accumulates integer
            # counts, so the sum is exact whatever order they arrive in -- but a
            # binary search whose needles arrive in order walks the haystack
            # sequentially instead of jumping around 1.2 MB of it. Measured
            # 2.6x on the far-OOD group, which is a third of the whole run.
            ood_sorted = {g: np.sort(column[ood_rows[g]]) for g in groups}
            for protocol in protocols:
                # Sorted once per (replicate, protocol, scorer) and reused
                # across OOD groups. The per-replicate FPR threshold is read
                # off *this* array, never off the full sample.
                id_sorted = np.sort(column[id_rows[protocol]])
                for group in groups:
                    for metric in metrics:
                        values[(scorer, protocol, group, metric)][b] = _metric(
                            metric, id_sorted, ood_sorted[group], tpr
                        )
        if progress is not None:
            progress(b + 1)

    return BootstrapResult(
        values=values,
        point=point,
        scorers=scorer_names,
        protocols=protocols,
        ood_groups=groups,
        metrics=metrics,
        seed=int(seed),
        n_replicates=int(n_replicates),
        level=float(level),
        tpr_target=float(tpr),
        plan_diagnostics=dict(plan.diagnostics),
    )


def _metric(metric: str, id_sorted: np.ndarray, ood: np.ndarray, tpr: float) -> float:
    if metric == "auroc":
        return auroc_against_sorted_id(id_sorted, ood)
    if metric == "fpr95":
        return fpr_against_sorted_id(id_sorted, ood, tpr)
    raise ValueError(f"unknown metric {metric!r}")
