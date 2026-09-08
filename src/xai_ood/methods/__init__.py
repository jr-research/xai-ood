"""OOD scorers.

**One rule, one place: every function in this package returns "higher = more OOD."**
No exceptions, no per-scorer discretion. OpenOOD's postprocessors return a
*confidence* (higher = more in-distribution), so exactly one adapter flips the
sign at the OpenOOD seam:

    def ood_score_from_openood_confidence(conf):
        return -conf

Never negate an OpenOOD confidence anywhere else. If a second negation appears,
that is the bug already happening: mixing a distance-as-score with a
confidence-as-score yields an AUROC of roughly one minus its true value, which
reads as a plausible 0.13 rather than as a crash.

``tests/test_sign_convention_and_naive_baseline.py`` turns this rule into code.

Score dumps go out in the canonical row order defined in ``xai_ood.schema``;
call ``assert_canonical_order`` at the end of every dump function.

Contents: the 2x2 Gaussian factorial with RMD and RMD++, both kNN variants, and
the PCA-residual family with the numpy rank correlation it needs. All four
families have landed.

Every family shares two things and must keep sharing them: one L2 normalization
(``xai_ood.methods.covariance.l2_normalize``, which guards the denominator)
and one score-dump function (``xai_ood.methods.dump.score_frame``, which
ends in ``assert_canonical_order``). A second copy of either is a bug in
waiting, not a convenience.
"""

from __future__ import annotations

from typing import Any

from ..reporting import (
    REPORTING_TIERS,
    SHRINKAGE_ABLATION_TIER,
    validate_reporting_tier,
)
from .covariance import (
    NORM_EPS,
    CovarianceFit,
    fit_covariance,
    l2_normalize,
    ledoit_wolf_shrinkage,
    shrink_covariance,
)
from .dump import Scorer, score_frame

# Family-scoped names are prefixed at package level even though they are
# unprefixed inside their own modules. ``CONFIGURATIONS`` is unambiguous in
# gaussian.py and in knn.py; imported side by side into one namespace it is not,
# and the PCA-residual family adds a third. Import the prefixed names from here, or the
# bare names from the module -- but never let ``CONFIGURATIONS`` mean whichever
# family happened to be imported last.
from .gaussian import CONFIGURATIONS as GAUSSIAN_CONFIGURATIONS
from .gaussian import (
    GaussianConfig,
    GaussianScorer,
    fit_all_scorers,
    fit_scorer,
)
from .knn import CONFIGURATIONS as KNN_CONFIGURATIONS
from .knn import (
    DEFAULT_K,
    SWEEP_K,
    KnnConfig,
    KnnScorer,
    fit_all_knn,
    fit_knn,
)
from .correlation import average_ranks, pearson_r, spearman_rho
from .pca_residual import CONFIGURATIONS as PCA_RESIDUAL_CONFIGURATIONS
from .pca_residual import (
    DEGENERACY_RHO_THRESHOLD,
    SUBSPACES,
    SWEEP_D,
    VARIANCE_THRESHOLD,
    PcaResidualConfig,
    PcaResidualScorer,
    SubspaceFit,
    degeneracy_report,
    explained_variance_ratio,
    fit_all_pca_residual,
    fit_pca_residual,
    select_dimension_by_variance,
)
from .manifest import (
    SHRINKAGE_TARGET,
    covariance_manifest,
    covariance_manifest_entry,
    run_manifest,
    write_run_manifest,
)

__all__ = [
    "ood_score_from_openood_confidence",
    "REPORTING_TIERS",
    "SHRINKAGE_ABLATION_TIER",
    "validate_reporting_tier",
    "NORM_EPS",
    "CovarianceFit",
    "fit_covariance",
    "l2_normalize",
    "ledoit_wolf_shrinkage",
    "shrink_covariance",
    "GAUSSIAN_CONFIGURATIONS",
    "GaussianConfig",
    "GaussianScorer",
    "fit_all_scorers",
    "fit_scorer",
    "Scorer",
    "score_frame",
    "DEFAULT_K",
    "SWEEP_K",
    "KNN_CONFIGURATIONS",
    "KnnConfig",
    "KnnScorer",
    "fit_all_knn",
    "fit_knn",
    "average_ranks",
    "pearson_r",
    "spearman_rho",
    "PCA_RESIDUAL_CONFIGURATIONS",
    "DEGENERACY_RHO_THRESHOLD",
    "SUBSPACES",
    "SWEEP_D",
    "VARIANCE_THRESHOLD",
    "PcaResidualConfig",
    "PcaResidualScorer",
    "SubspaceFit",
    "degeneracy_report",
    "explained_variance_ratio",
    "fit_all_pca_residual",
    "fit_pca_residual",
    "select_dimension_by_variance",
    "SHRINKAGE_TARGET",
    "covariance_manifest",
    "covariance_manifest_entry",
    "run_manifest",
    "write_run_manifest",
]


def ood_score_from_openood_confidence(confidence: Any) -> Any:
    """**The** OpenOOD seam adapter. The only negation in the codebase.

    OpenOOD postprocessors return a confidence (higher = more in-distribution).
    Everything in this package returns higher = more OOD. This function is where
    the two conventions meet, and it is the only place a sign is flipped.

    Defined here as code rather than only in the docstring above so that the
    rule is importable: a second negation elsewhere is now a visible duplicate
    of an existing function rather than an invisible ad-hoc minus sign.
    """
    return -confidence
