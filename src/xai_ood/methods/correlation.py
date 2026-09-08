"""Rank correlation in numpy, because this package may not import scipy.

The obvious call for the PCA-residual degeneracy control is
``scipy.stats.spearmanr``. ``tests/test_import_hygiene.py`` forbids it in
``src/``, and that rule is load-bearing rather than decorative: Ledoit-Wolf is
reimplemented here and kNN uses numpy exact search instead of
faiss, both so the whole suite stays developable and testable with only numpy
and pandas installed. Spearman is a rank transform
followed by Pearson, which is a few lines, so the same trade applies here at a
much smaller cost.

The algorithm is unchanged from scipy's: **average ranks for ties**, then the
Pearson correlation of the two rank vectors.
``tests/test_spearman.py`` checks it twice -- against hand-computed examples that
need no optional dependency, and against ``scipy.stats.spearmanr`` behind
``pytest.importorskip``, which un-skips wherever scipy is installed.

What needs this
---------------
The PCA-residual family's **degeneracy control**: is the class-mean residual
anything more than a rescaled embedding norm? See
``xai_ood.methods.pca_residual``.

Deliberately *not* returning a p-value
--------------------------------------
``scipy.stats.spearmanr`` returns ``(rho, p)``. This returns only ``rho``. The
p-value there tests rho != 0, which at n = 9,000 is significant for any rho
worth reporting and answers a question nobody asked:
the pre-declared decision rule is a **threshold on the effect size**
(rho >= 0.95), not a significance test. Reporting a p-value alongside it invites
exactly the confusion the pre-declaration exists to prevent. Recomputing it here
would also mean reimplementing a t-distribution survival function in numpy, for
a number that should not be reported.
"""

from __future__ import annotations

import numpy as np

__all__ = ["average_ranks", "pearson_r", "spearman_rho"]


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of ``x``, 1-based, **ties sharing their average rank**.

    ``[10, 20, 20, 40]`` ranks as ``[1, 2.5, 2.5, 4]``. This is scipy's
    ``rankdata(method="average")`` and is what makes the result Spearman's rho
    rather than an arbitrary tie-broken approximation of it.

    Ties are not a hypothetical here. The class-mean residual and the embedding
    norm are float64 and will rarely tie exactly, but a *sweep* over a coarse
    grid, an integer severity, or a score array that saturates can all produce
    them, and silently breaking ties by input order would make the correlation
    depend on the row order of the dump.
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("average_ranks: empty input has no ranks")
    if not np.isfinite(values).all():
        raise ValueError(
            "average_ranks: input contains NaN or infinity. Ranking these is "
            "undefined and numpy would sort them to the end rather than raise, "
            "which would produce a plausible-looking correlation from bad input."
        )

    order = np.argsort(values, kind="stable")
    ordered = values[order]

    # Group boundaries: a new group starts wherever the sorted value changes.
    starts_group = np.empty(ordered.size, dtype=bool)
    starts_group[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=starts_group[1:])
    group = np.cumsum(starts_group) - 1

    positions = np.arange(1, ordered.size + 1, dtype=np.float64)
    group_totals = np.bincount(group, weights=positions)
    group_sizes = np.bincount(group)
    ranks_in_sorted_order = (group_totals / group_sizes)[group]

    ranks = np.empty(ordered.size, dtype=np.float64)
    ranks[order] = ranks_in_sorted_order
    return ranks


def pearson_r(a: np.ndarray, b: np.ndarray, *, name: str = "pearson_r") -> float:
    """Pearson product-moment correlation of two 1-D arrays.

    Raises on a constant input rather than returning NaN. scipy returns NaN with
    a warning; here a constant array is not a numerical edge case to absorb but
    a statement that the thing being correlated carries no information at all,
    which for a score array means the scorer is degenerate. Input that is already
    wrong raises at the point it is detected instead of producing a plausible
    number downstream.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"{name}: length mismatch, {x.size} against {y.size}")
    if x.size < 2:
        raise ValueError(f"{name}: need at least 2 observations, got {x.size}")

    dx = x - x.mean()
    dy = y - y.mean()
    nx = float(np.sqrt(np.dot(dx, dx)))
    ny = float(np.sqrt(np.dot(dy, dy)))
    if nx == 0.0 or ny == 0.0:
        constant = "first" if nx == 0.0 else "second"
        raise ValueError(
            f"{name}: the {constant} input is constant, so the correlation is "
            f"undefined (zero variance). A constant score array means the "
            f"scorer separates nothing; that is the finding, not a number to "
            f"paper over with NaN."
        )
    return float(np.dot(dx, dy) / (nx * ny))


def spearman_rho(a: np.ndarray, b: np.ndarray, *, name: str = "spearman_rho") -> float:
    """Spearman rank correlation: Pearson on average ranks. Returns rho only.

    Identical to ``scipy.stats.spearmanr(a, b).statistic`` for any input either
    accepts, which ``tests/test_spearman.py`` asserts where scipy is installed.

    Being rank-based, this is invariant under any strictly monotone transform of
    either argument -- which is the entire reason the degeneracy control uses it.
    "The class-mean residual is a rescaled embedding norm" is a claim about
    *order*, and a Pearson correlation could be dragged below 1 by a nonlinear
    but perfectly order-preserving relationship between the two, understating
    exactly the degeneracy the control exists to detect.
    """
    return pearson_r(average_ranks(a), average_ranks(b), name=name)
