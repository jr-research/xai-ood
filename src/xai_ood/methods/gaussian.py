"""The eight Gaussian-family configurations, built on one covariance code path.

Four cells of the 2x2 factorial, plus RMD and RMD++, plus the two normalised
Full cells added on 2026-09-13 (see "Added after the results existed" below)::

                    |  Diagonal                   |  Full (pooled)
    ----------------+-----------------------------+---------------------------
    Marginal        |  marginal_diagonal          |  marginal_full
                    |  (the naive per-dimension   |  (Mahalanobis to the
                    |   baseline)                 |   global mean)
    Class-cond.     |  class_conditional_diagonal |  class_conditional_full
                    |                             |  (MDS, Lee et al. 2018)

    rmd     = class_conditional_full - marginal_full   (Ren et al. 2021)
    rmd_pp  = the same, on L2-normalised embeddings    (Mueller & Hein 2025)

    marginal_full_pp          = marginal_full          on L2-normalised features
    class_conditional_full_pp = class_conditional_full on L2-normalised features
                                (this one is Mahalanobis++ itself)

The two cells of the **Full** column are literally RMD's two terms, so reporting
them separately is a decomposition of a scorer already in the design rather
than an added experiment, and it is the only way to say *why* RMD helps if it does.

Sign convention, no exceptions: **higher = more OOD.** Every score here is a
squared Mahalanobis distance or a difference of two, so higher is already more
OOD and nothing is negated. See the package docstring.

Squared, not square-rooted
--------------------------
All distances are **squared** Mahalanobis. For the four non-relative cells this
is free: AUROC is invariant under the monotone square root. For RMD it is not a
free choice, because a difference of squares is not a monotone function of a
difference of roots, and Ren et al. (2021) define both of RMD's terms as squared
Mahalanobis distances. Squared throughout, so the four cells stay literally the
components of RMD rather than merely its cousins.

RMD is a score, not a distance, and is routinely negative for ID samples. That
is expected: it is the *excess* of class-specific deviation over generic
unusualness. Do not clip it.

Added after the results existed
-------------------------------
``marginal_full_pp`` and ``class_conditional_full_pp`` were added on
**2026-09-13, after the first results existed**, prompted by reading Mueller and
Hein 2025 in full. **They are not part of the pre-specified 2x2 factorial and
must never be printed inside it.** The factorial is the four cells above and
nothing else, and its two main effects and its interaction are computed from
those four alone.

They were added because the design already shipped the *derived* score on the
normalised arm without either score it is derived from: ``rmd_pp`` is by
construction ``class_conditional_full_pp - marginal_full_pp``, exactly as
``rmd`` is the difference of the two unnormalised Full cells. The unnormalised
arm carried that decomposition and the normalised arm did not. Completing it is
the reason; where the completed cells land on a leaderboard is not.

Both carry tier ``appendix``, the lowest tier that still guarantees a cell
appears in the results tables, and the tier this project's declared vocabulary
assigns to normalisation ablations that are not themselves a primary claim. A
cell added after the results cannot be allowed to earn more narrative weight
than one fixed before them. The tier assignment itself was fixed on 2026-08-31;
these two entries are a dated amendment to it rather than part of it, and
``tests/test_reporting_tiers.py`` marks them as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..reporting import validate_reporting_tier
from ..schema import encode_hyperparams
from .covariance import CovarianceFit, fit_covariance, l2_normalize
from .dump import score_frame

__all__ = [
    "GaussianConfig",
    "GaussianScorer",
    "CONFIGURATIONS",
    "config_by_name",
    "fit_scorer",
    "fit_all_scorers",
    "score_frame",  # re-exported from .dump; it is not Gaussian-specific
]


@dataclass(frozen=True)
class GaussianConfig:
    """One point in the factorial. Four booleans, a name and a tier; no logic.

    ``reporting_tier`` is keyword-only and has **no default**, so a configuration
    added later cannot be constructed without one. See ``xai_ood.reporting``:
    the tier governs how much narrative a configuration earns, never whether it
    runs.
    """

    name: str
    class_conditional: bool
    diagonal: bool
    relative: bool  # subtract the marginal-full background term (RMD)
    l2_normalize: bool
    reporting_tier: str = field(kw_only=True)

    def __post_init__(self) -> None:
        validate_reporting_tier(self.reporting_tier, name=self.name)

    def hyperparams(self, *, shrinkage: bool) -> dict[str, Any]:
        """The dict that goes into the results table's ``hyperparams`` column."""
        return {
            "class_conditional": self.class_conditional,
            "covariance": "diagonal" if self.diagonal else "full",
            "relative": self.relative,
            "l2_normalize": self.l2_normalize,
            "shrinkage": shrinkage,
            "shrinkage_target": "ledoit_wolf_scaled_identity" if shrinkage else None,
            "pooling": "pooled_within_class" if self.class_conditional else "marginal",
        }

    def encoded_hyperparams(self, *, shrinkage: bool) -> str:
        return encode_hyperparams(self.hyperparams(shrinkage=shrinkage))


#: The eight configurations, in reporting order. Names match the keys in
#: ``xai_ood.visualization.style._SERIES_COLOR`` so a scorer keeps one colour
#: across every figure in the thesis.
#:
#: Tiers declared 2026-08-31. The four 2x2 cells are
#: **primary** because the factorial's two main effects and its interaction are
#: computed from exactly those four cells and nothing else. RMD and RMD++ are
#: **secondary**: RMD is the difference of the two cells in the Full column, so
#: the Full column already carries its decomposition, and the RMD/RMD++ contrast
#: is a normalization ablation rather than a factor of the design. All eight run.
#:
#: The last two were added 2026-09-13, after results existed, at tier
#: ``appendix``. See "Added after the results existed" in the module docstring:
#: their date, their reason and the constraint on how they may be reported are
#: recorded there rather than here, because that is the text a reader hits first.
CONFIGURATIONS: tuple[GaussianConfig, ...] = (
    GaussianConfig("marginal_diagonal", False, True, False, False, reporting_tier="primary"),
    GaussianConfig("marginal_full", False, False, False, False, reporting_tier="primary"),
    GaussianConfig("class_conditional_diagonal", True, True, False, False, reporting_tier="primary"),
    GaussianConfig("class_conditional_full", True, False, False, False, reporting_tier="primary"),
    GaussianConfig("rmd", True, False, True, False, reporting_tier="secondary"),
    GaussianConfig("rmd_pp", True, False, True, True, reporting_tier="secondary"),
    # Post-hoc, 2026-09-13. Not cells of the factorial.
    GaussianConfig("marginal_full_pp", False, False, False, True, reporting_tier="appendix"),
    GaussianConfig("class_conditional_full_pp", True, False, False, True, reporting_tier="appendix"),
)

_BY_NAME: dict[str, GaussianConfig] = {c.name: c for c in CONFIGURATIONS}


def config_by_name(name: str) -> GaussianConfig:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown configuration {name!r}; the eight are "
            f"{sorted(_BY_NAME)}"
        ) from None


class GaussianScorer:
    """A fitted configuration. ``score(x)`` returns higher = more OOD.

    Holds one fit for the four plain cells, and two for the relative ones (the
    class-conditional fit and the marginal-full background fit). The background
    fit is always **full**, never diagonal: RMD is defined as the difference of
    the two cells in the Full column.
    """

    def __init__(
        self,
        config: GaussianConfig,
        primary: CovarianceFit,
        background: CovarianceFit | None = None,
        *,
        shrinkage: bool = False,
    ) -> None:
        self.config = config
        self.primary = primary
        self.background = background
        self.shrinkage = bool(shrinkage)

    @property
    def name(self) -> str:
        return self.config.name

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        return l2_normalize(x) if self.config.l2_normalize else x

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score embeddings. Higher = more OOD. Returns a float64 ``(n,)`` array."""
        z = self._prepare(x)
        if self.config.class_conditional:
            primary_term = self.primary.class_conditional_mahalanobis_sq(z)
        else:
            primary_term = self.primary.mahalanobis_sq(z)

        if not self.config.relative:
            return primary_term

        assert self.background is not None  # constructed together, see fit_scorer
        return primary_term - self.background.mahalanobis_sq(z)

    def diagnostics(self) -> dict[str, Any]:
        """Manifest fragment for this scorer: one entry per fitted covariance.

        Every value here is read off the fit at runtime. The condition numbers
        in particular are the measurement that settles the shrinkage-conditioning
        claim, and they are unavailable retroactively without refitting.
        """
        out: dict[str, Any] = {
            "scorer": self.config.name,
            "hyperparams": self.config.encoded_hyperparams(shrinkage=self.shrinkage),
            # Not in hyperparams(): a tier is a reporting decision, not a
            # parameter of the estimator, and the results
            # table's hyperparams column is what distinguishes two runs of the
            # same scorer. It belongs in the manifest, which is where the
            # declarations are checkable against what ran.
            "reporting_tier": self.config.reporting_tier,
            "covariances": {"primary": dict(self.primary.diagnostics)},
        }
        if self.background is not None:
            out["covariances"]["background"] = dict(self.background.diagnostics)
        return out


def fit_scorer(
    config: GaussianConfig | str,
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    shrinkage: bool = False,
) -> GaussianScorer:
    """Fit one configuration on ID **train** embeddings.

    ``labels`` is required for the class-conditional configurations and ignored
    for the marginal ones. Fit on train, evaluate on test, no exceptions.
    """
    if isinstance(config, str):
        config = config_by_name(config)
    if config.class_conditional and labels is None:
        raise ValueError(
            f"{config.name}: labels are required for a class-conditional "
            f"configuration. Passing None fits the marginal model instead, "
            f"which would silently collapse two cells of the factorial into one."
        )

    primary = fit_covariance(
        embeddings,
        labels if config.class_conditional else None,
        diagonal=config.diagonal,
        shrinkage=shrinkage,
        l2_normalize_first=config.l2_normalize,
        name=f"{config.name}:primary",
    )

    background = None
    if config.relative:
        # RMD's background term: marginal, full, same normalisation, same
        # shrinkage flag. Never diagonal -- RMD is the difference of the two
        # cells in the Full column.
        background = fit_covariance(
            embeddings,
            None,
            diagonal=False,
            shrinkage=shrinkage,
            l2_normalize_first=config.l2_normalize,
            name=f"{config.name}:background",
        )

    return GaussianScorer(config, primary, background, shrinkage=shrinkage)


def fit_all_scorers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    shrinkage: bool = False,
    configs: Iterable[GaussianConfig] | None = None,
) -> dict[str, GaussianScorer]:
    """Fit all six configurations from one pass over the ID train embeddings."""
    chosen = tuple(configs) if configs is not None else CONFIGURATIONS
    return {
        c.name: fit_scorer(c, embeddings, labels, shrinkage=shrinkage) for c in chosen
    }


# ``score_frame`` lives in ``xai_ood.methods.dump``, because it is not
# Gaussian-specific and kNN and the PCA-residual scorer need the same one. It is
# re-exported above so ``xai_ood.methods.gaussian.score_frame`` also resolves.
