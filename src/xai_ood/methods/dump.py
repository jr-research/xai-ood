"""The one per-sample score-dump function, shared by every scorer family.

``assert_canonical_order`` makes every score dump end with a check that row *i*
means the same image in every scorer's array. That only works if there is one
dump function rather than one per family, so this lives here rather than inside
``gaussian.py``, ``knn.py`` or the PCA-residual module.

Any object with a ``name`` and a ``score(x) -> (n,) array`` satisfies
``Scorer``; nothing here knows or cares which family it came from.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from ..schema import CANONICAL_SORT_KEY, assert_canonical_order, canonical_sort

__all__ = ["Scorer", "score_frame"]


@runtime_checkable
class Scorer(Protocol):
    """Structural type every scorer in this package satisfies.

    ``score`` returns **higher = more OOD**, without exception. See the package
    docstring; that rule is the whole reason this Protocol has no ``higher_is_ood``
    flag for a scorer to get wrong.
    """

    name: str

    def score(self, x: np.ndarray) -> np.ndarray: ...


def score_frame(
    index: pd.DataFrame,
    embeddings: np.ndarray,
    scorers: Mapping[str, Scorer] | Sequence[Scorer],
    *,
    name: str = "scores",
) -> pd.DataFrame:
    """Score every row and return a canonically-ordered per-sample dump.

    ``index`` carries the four canonical key columns
    ``(split, image_id, corruption, severity)`` and has exactly as many rows as
    ``embeddings``, in the same order. ``corruption`` and ``severity`` are
    null-valued (not absent) for non-cs-ID rows.

    The returned frame is sorted into canonical order **and the embeddings are
    permuted with it**, so a score never parts company with its row. The last
    thing this function does is call
    ``xai_ood.schema.assert_canonical_order``, which is what that helper was
    built for: the bootstrap applies one index set to every scorer's
    array, and a misalignment there produces wrong paired differences without
    ever raising.

    Scorers from different families can be mixed in one call, which is the
    point: a single dump covering the Gaussian cells, kNN and PCA-residual is
    trivially aligned, whereas three separate dumps would need
    ``xai_ood.schema.assert_aligned`` to prove the same thing afterwards.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if len(index) != x.shape[0]:
        raise ValueError(
            f"{name}: index has {len(index)} rows but embeddings has {x.shape[0]}. "
            f"These must correspond row for row before sorting."
        )

    scorer_list = list(scorers.values()) if isinstance(scorers, Mapping) else list(scorers)
    if not scorer_list:
        raise ValueError(f"{name}: no scorers passed")

    duplicates = {s.name for s in scorer_list if [t.name for t in scorer_list].count(s.name) > 1}
    if duplicates:
        raise ValueError(
            f"{name}: duplicate scorer name(s) {sorted(duplicates)}. Column names "
            f"must be unique or one scorer's scores silently overwrite another's."
        )

    # The same failure one step over: a scorer named after a key column would
    # overwrite the column that defines row identity. On a cs-ID frame, scores
    # landing inside the valid severity range corrupt the dump without raising.
    reserved = {s.name for s in scorer_list} & set(CANONICAL_SORT_KEY)
    if reserved:
        raise ValueError(
            f"{name}: scorer name(s) {sorted(reserved)} collide with the canonical "
            f"key columns {list(CANONICAL_SORT_KEY)} and would overwrite them."
        )

    keys = index.loc[:, list(CANONICAL_SORT_KEY)].copy()
    keys["_row"] = np.arange(len(keys))
    ordered = canonical_sort(keys, name=name)
    permutation = ordered["_row"].to_numpy()

    out = ordered.drop(columns="_row")
    scored = x[permutation]
    for scorer in scorer_list:
        out[scorer.name] = scorer.score(scored)

    return assert_canonical_order(out, name=name)
