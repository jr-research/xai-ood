"""The kNN family: k-th nearest-neighbour distance (Sun et al., 2022).

The non-parametric counterpoint to the Gaussian family. Sun et al. verify that
learned embeddings fail a multivariate-normality test, which undercuts the
Gaussian-mixture assumption the Mahalanobis family rests on, and drop the
parametric assumption entirely::

    r_k(z) = || z - z_(k) ||_2

the Euclidean distance from a test embedding to the **k-th** nearest embedding
in the reference set. Higher = more OOD, with no sign flip anywhere.

Design decisions
----------------
* **k = 50, fixed a priori**, identical for both variants. Not convention: Sun
  et al. select k by validation over {1, 10, 20, 50, 100, 200, 500, 1000, 3000,
  5000} and report k = 50 for CIFAR-10 with a 50,000-image reference set, and
  this thesis's reference set is *that same split at that same size*. Only the
  backbone differs, so the paper's own "optimal k scales with reference-set
  size" caveat does not bite. It is a **default argument**, never a literal at a
  call site, so the k sweep reuses this function rather than a copy.
* **k-th nearest neighbour, not k-averaged.** Sun et al. report the two as
  empirically similar but theoretically motivate only the k-th (a Bayes-optimal
  argument under a Huber contamination model), so the k-th is what ships.
  ``KnnScorer.score_k_averaged`` exists **only** so the difference can be
  measured on request; it is not one of the reported variants.
* **Both normalized and unnormalized**, as a flag, so normalization is a factor
  applied consistently across scorer families rather than baked into one scorer
  and ablated on another. Sun et al. normalize; their ablation puts 61 FPR95
  points on it. Whether that transfers to frozen DINOv2 CLS embeddings is an
  empirical question this thesis answers, not an assumption.
* **Reference set is the CIFAR-10 train split, 50,000 images**, label-free.
  Fit on train, evaluate on test, no exceptions.

The one OpenOOD quirk not to copy
---------------------------------
Their normalizer is ``x / norm(x) + 1e-10``, which adds a constant to every
component *after* dividing rather than guarding the denominator. Harmless at
these magnitudes but wrong, and it is the documented explanation for any
third-decimal disagreement with their published numbers. This module does not
reimplement normalization at all: it imports
``xai_ood.methods.covariance.l2_normalize``, which guards the denominator,
so there is exactly one L2 normalization in the codebase and RMD++ and kNN
cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..reporting import validate_reporting_tier
from ..schema import encode_hyperparams
from .covariance import l2_normalize

__all__ = [
    "DEFAULT_K",
    "SWEEP_K",
    "KnnConfig",
    "KnnScorer",
    "CONFIGURATIONS",
    "config_by_name",
    "fit_knn",
    "fit_all_knn",
    "REFERENCE_SPLIT",
]

#: The headline value. A default argument, never a literal at a call site.
DEFAULT_K: int = 50

#: The reference set, named once. This is a fixed decision, not a runtime
#: quantity: Sun et al. (2022) use the same 50,000-image CIFAR-10 train split,
#: which is the whole justification for k = 50 transferring. A literal is
#: therefore the right shape -- unlike rho, the condition numbers and d, which
#: are measured at runtime and must never appear as constants. It lives here
#: rather than inline in ``KnnConfig.hyperparams()``, so the value reaching the
#: manifest is the same one the scorer used.
REFERENCE_SPLIT: str = "cifar10_train"

#: The appendix sensitivity grid. Reported, **never used to reselect**:
#: sweeping against the test AUROC being reported would advantage kNN over the
#: three families with no comparable knob. Pre-declared failure rule: a ranking
#: flip inside this grid is written up as a k-sensitivity limitation, not fixed
#: by choosing a better k.
SWEEP_K: tuple[int, ...] = (1, 10, 20, 50, 100, 200, 500, 1000)

#: Query rows per distance block. 50,000 references at 768 dimensions means a
#: full 9,000 x 50,000 float64 distance matrix would be 3.6 GB; blocking keeps
#: the peak at roughly ``_QUERY_BLOCK * n_reference * 8`` bytes, about 200 MB at
#: the default. Purely a memory knob: results are identical for any block size,
#: which ``test_block_size_does_not_change_the_scores`` asserts.
#:
#: That figure accounts for the distance block only. **On top of it, each
#: variant holds its own float64 copy of the reference set** -- 0.29 GiB at
#: 50,000 x 768 -- because ``KnnScorer.__init__`` does its own
#: ``np.asarray(reference, dtype=np.float64)`` and the cache is float32. With
#: both variants resident that is 0.57 GiB of reference alone, before the
#: Gaussian fits. Deliberate: the unnormalized variant could share a view, but
#: an independent copy is cheap insurance against one variant mutating what the
#: other reads. Worth knowing when every scorer is resident at once.
_QUERY_BLOCK: int = 512


@dataclass(frozen=True)
class KnnConfig:
    """One kNN variant. Normalization is the only factor; k is shared.

    ``reporting_tier`` is keyword-only with no default, so ``fit_knn``'s
    k-override path has to carry the tier forward explicitly rather than
    defaulting it. See ``xai_ood.reporting``.
    """

    name: str
    normalize: bool
    k: int = DEFAULT_K
    reporting_tier: str = field(kw_only=True)

    def __post_init__(self) -> None:
        validate_reporting_tier(self.reporting_tier, name=self.name)

    def hyperparams(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "aggregation": "kth",  # not "k_averaged"; see the module docstring
            "normalize": self.normalize,
            "metric": "euclidean",
            "reference_split": REFERENCE_SPLIT,
        }

    def encoded_hyperparams(self) -> str:
        return encode_hyperparams(self.hyperparams())


#: The two reported variants. Names match ``visualization.style._SERIES_COLOR``.
#:
#: Tiers declared 2026-08-31. ``knn_unnormalized`` is
#: **primary**: it is the non-parametric counterpoint the Gaussian family is
#: measured against, and unnormalized is the primary variant for the same reason
#: it is in the PCA-residual family, so that normalization stays an ablation
#: rather than being baked into the headline scorer. ``knn_normalized`` is
#: **appendix**, as the other half of that ablation. Both run.
CONFIGURATIONS: tuple[KnnConfig, ...] = (
    KnnConfig("knn_normalized", True, reporting_tier="appendix"),
    KnnConfig("knn_unnormalized", False, reporting_tier="primary"),
)

_BY_NAME: dict[str, KnnConfig] = {c.name: c for c in CONFIGURATIONS}


def config_by_name(name: str) -> KnnConfig:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown kNN configuration {name!r}; the two reported ones are {sorted(_BY_NAME)}"
        ) from None


class KnnScorer:
    """A kNN scorer holding its reference set. ``score(x)`` is higher = more OOD.

    **The reference set must not contain the query.** Scoring the reference set
    against itself gives ``r_1 = 0`` for every row, because each point is its own
    nearest neighbour. That is not a bug and is not guarded against, because the
    protocol never does it -- the reference set is CIFAR-10 *train* and every
    evaluated split is disjoint from it. It is documented because a
    sanity check that scores train against train would otherwise look broken.
    """

    def __init__(self, config: KnnConfig, reference: np.ndarray) -> None:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim != 2:
            raise ValueError(f"{config.name}: expected a 2-D (n, p) reference set, got {ref.shape}")
        if ref.shape[0] < config.k:
            raise ValueError(
                f"{config.name}: k = {config.k} exceeds the reference set size "
                f"{ref.shape[0]}. There is no k-th nearest neighbour."
            )
        self.config = config
        # One L2 normalization in the codebase, imported not reimplemented.
        self.reference = l2_normalize(ref) if config.normalize else ref
        self._reference_sq = np.einsum("ij,ij->i", self.reference, self.reference)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def n_reference(self) -> int:
        return int(self.reference.shape[0])

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        return l2_normalize(x) if self.config.normalize else x

    def _sorted_prefix(self, x: np.ndarray, k_max: int, block: int) -> np.ndarray:
        """The ``k_max`` smallest squared distances per query, ascending.

        Distances come from the Gram identity
        ``d^2 = ||q||^2 + ||r||^2 - 2 q.r`` rather than explicit differences,
        because an explicit ``(block, n_ref, p)`` array is not affordable at
        9,000 x 50,000. This is also what faiss's ``IndexFlatL2`` does, so it
        matches Sun et al.'s own computation. The identity loses relative
        precision for near-coincident points, where the subtraction cancels; the
        residual error is far below the spacing that decides a k-th-neighbour
        ranking, and ``test_gram_distances_match_explicit_pairwise_norms``
        bounds it. Negative values from that cancellation are clipped to zero
        before the square root.
        """
        q = self._prepare(x)
        if q.shape[1] != self.reference.shape[1]:
            raise ValueError(
                f"{self.name}: query has {q.shape[1]} dimensions, reference has "
                f"{self.reference.shape[1]}"
            )

        out = np.empty((q.shape[0], k_max), dtype=np.float64)
        for start in range(0, q.shape[0], block):
            stop = min(start + block, q.shape[0])
            chunk = q[start:stop]
            d2 = (
                np.einsum("ij,ij->i", chunk, chunk)[:, None]
                + self._reference_sq[None, :]
                - 2.0 * (chunk @ self.reference.T)
            )
            np.maximum(d2, 0.0, out=d2)
            # Partition once at k_max, then sort only that prefix: O(n_ref) plus
            # O(k_max log k_max) instead of a full O(n_ref log n_ref) sort.
            partitioned = np.partition(d2, k_max - 1, axis=1)[:, :k_max]
            out[start:stop] = np.sort(partitioned, axis=1)
        return out

    def score(
        self, x: np.ndarray, *, k: int | None = None, block: int = _QUERY_BLOCK
    ) -> np.ndarray:
        """Distance to the k-th nearest reference embedding. Higher = more OOD.

        ``k`` defaults to the configuration's value (50). It is exposed so the k
        sweep calls **this** function rather than a second implementation; the
        headline table uses the default regardless of what the sweep shows.
        """
        k = self.config.k if k is None else int(k)
        self._check_k(k)
        return np.sqrt(self._sorted_prefix(x, k, block)[:, k - 1])

    def score_sweep(
        self,
        x: np.ndarray,
        ks: Sequence[int] = SWEEP_K,
        *,
        block: int = _QUERY_BLOCK,
    ) -> dict[int, np.ndarray]:
        """Scores for several k from **one** distance pass.

        The appendix figure needs eight values of k. Calling
        ``score`` eight times would recompute the same 9,000 x 50,000
        distance matrix eight times; this partitions once at ``max(ks)``.
        Values are identical to the per-k calls, which
        ``test_sweep_agrees_with_individual_scores`` asserts.
        """
        ks = sorted({int(k) for k in ks})
        for k in ks:
            self._check_k(k)
        prefix = self._sorted_prefix(x, max(ks), block)
        return {k: np.sqrt(prefix[:, k - 1]) for k in ks}

    def score_k_averaged(
        self, x: np.ndarray, *, k: int | None = None, block: int = _QUERY_BLOCK
    ) -> np.ndarray:
        """Mean distance over the k nearest neighbours. **Not a reported variant.**

        Present only so the k-th-versus-k-averaged difference can be measured on
        request. Sun et al. report the two as empirically similar and motivate
        only the k-th theoretically, and this thesis reports the k-th. Do
        not put this in the results table.
        """
        k = self.config.k if k is None else int(k)
        self._check_k(k)
        return np.sqrt(self._sorted_prefix(x, k, block)).mean(axis=1)

    def _check_k(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"{self.name}: k must be at least 1, got {k}")
        if k > self.n_reference:
            raise ValueError(
                f"{self.name}: k = {k} exceeds the reference set size {self.n_reference}"
            )

    def diagnostics(self) -> dict[str, Any]:
        """Manifest fragment. Everything here is read off the fitted scorer."""
        return {
            "scorer": self.name,
            "hyperparams": self.config.encoded_hyperparams(),
            "reporting_tier": self.config.reporting_tier,
            "n_reference": self.n_reference,
            "n_features": int(self.reference.shape[1]),
            "normalize": bool(self.config.normalize),
            "k": int(self.config.k),
            "aggregation": "kth",
        }


def fit_knn(config: KnnConfig | str, reference: np.ndarray, *, k: int | None = None) -> KnnScorer:
    """Build a kNN scorer over ``reference`` (the CIFAR-10 train embeddings).

    "Fitting" is only normalization and caching the squared norms; kNN has no
    parameters to estimate, which is the point of the family. It is also
    **label-free**, so it sits on the same side of the marginal /
    class-conditional factor as the marginal Gaussian cells.
    """
    if isinstance(config, str):
        config = config_by_name(config)
    if k is not None:
        # The tier is carried forward rather than defaulted: a k-override is a
        # sweep point of the same reported variant, not a new one.
        config = KnnConfig(
            config.name,
            config.normalize,
            int(k),
            reporting_tier=config.reporting_tier,
        )
    return KnnScorer(config, reference)


def fit_all_knn(
    reference: np.ndarray,
    *,
    k: int | None = None,
    configs: Iterable[KnnConfig] | None = None,
) -> dict[str, KnnScorer]:
    """Both reported variants over one reference set."""
    chosen = tuple(configs) if configs is not None else CONFIGURATIONS
    return {c.name: fit_knn(c, reference, k=k) for c in chosen}
