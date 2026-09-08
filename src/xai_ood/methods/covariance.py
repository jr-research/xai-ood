"""The single covariance-estimation code path for the Gaussian family.

One fitting function, ``fit_covariance``, serves all four cells of the 2x2
factorial plus RMD's two terms. The two factors are arguments, not code paths:

* **marginal vs class-conditional** is whether ``labels`` is passed. Marginal
  fits one Gaussian to the whole ID cloud (labels ignored). Class-conditional
  fits K means with a single **pooled within-class** covariance estimated from
  the class-centred residuals.
* **diagonal vs full** is a *restriction applied to the fitted matrix*, not a
  separate estimator. The naive baseline is required to be a diagonal
  restriction of the shared path precisely so that
  ``tests/test_sign_convention_and_naive_baseline.py``'s equivalence assertion
  is a real check rather than a comparison of two unrelated implementations.

Scope note: the within-class covariance is **pooled**. A per-class covariance
is an optional third level of the factor and is deliberately not implemented
here.

Numerics, all four required
---------------------------
1. Estimation and inversion happen in ``float64``, whatever the input dtype.
2. Every fitted matrix is symmetrised with ``S = 0.5 * (S + S.T)`` before it is
   decomposed or factorised.
3. Eigendecomposition is ``numpy.linalg.eigh``, never ``eig``. ``eig`` on a
   symmetric matrix can return complex eigenpairs from numerical asymmetry.
4. Distances go through a **Cholesky factor**; the explicit inverse is never
   formed. See ``CovarianceFit.mahalanobis_sq``.

Diagnostics that must reach the run manifest
--------------------------------------------
Symmetry, condition number and the selected shrinkage intensity are computed at
fit time and carried on the fit object (``CovarianceFit.diagnostics``).
They are unavailable retroactively without refitting, and the condition numbers
are the measurement that finally settles the shrinkage-conditioning claim, which
is an input-level argument (~65 samples/dimension) rather than a
result. **None of them is ever a constant in a source file.**

On Ledoit-Wolf
--------------
The target is Ledoit & Wolf (2004), *Journal of Multivariate
Analysis* 88(2): a **scaled identity**, ``(1 - d) * S + d * mu * I`` with
``mu = trace(S) / p``. That is the estimator ``sklearn.covariance.LedoitWolf``
implements and cites. ``ledoit_wolf_shrinkage`` below reimplements
sklearn's own formula in numpy so that these tests run in an environment with
neither scipy nor scikit-learn installed, which is what keeps this family
testable without them. ``tests/test_covariance_estimator.py`` cross-checks against
scikit-learn when it is importable and skips otherwise, so the equivalence is
asserted automatically wherever scikit-learn is installed.

Shrinkage is an **ablation**, not an assumed necessity. On the four cells
(marginal and pooled within-class, both fit on all 50,000 train embeddings) it
is expected to be near-inert; it is load-bearing only for the optional per-class
variant. Report the number, do not argue the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "CovarianceFit",
    "fit_covariance",
    "ledoit_wolf_shrinkage",
    "shrink_covariance",
    "l2_normalize",
    "NORM_EPS",
]


#: Denominator guard for L2 normalisation. Note the placement: ``x / (n + eps)``,
#: **not** OpenOOD's ``x / n + eps``, which adds a constant after dividing.
#: Harmless at these magnitudes but wrong, and it is the documented explanation
#: for any third-decimal disagreement with their published numbers.
NORM_EPS: float = 1e-10


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation with a guarded denominator, in float64."""
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + NORM_EPS)


# --------------------------------------------------------------------------- #
# Ledoit-Wolf, scaled-identity target
# --------------------------------------------------------------------------- #


def ledoit_wolf_shrinkage(residuals: np.ndarray) -> float:
    """Ledoit-Wolf shrinkage intensity for the scaled-identity target.

    ``residuals`` are **already centred** rows: ``x_i - mu`` for the marginal
    model, ``x_i - mu_{k(i)}`` for the pooled within-class model. This is the
    ``assume_centered=True`` case, which is the only one this module needs,
    because centring is the caller's business (it is what distinguishes the two
    models).

    Equivalent to ``sklearn.covariance.ledoit_wolf_shrinkage``, and to the
    formulas in Ledoit & Wolf (2004, JMVA) section 3, which
    ``tests/test_covariance_estimator.py`` checks by an independent naive route
    that shares no algebra with this one. Reimplemented rather than imported so
    this module has no scipy or scikit-learn dependency.

    Written to allocate nothing of size ``(n, p)``. sklearn's version forms
    ``X**2`` and the Gram product ``X2.T @ X2``, which at (50,000 x 768) costs an
    extra 300 MB and an O(n p^2) product; sklearn blocks the loop to bound the
    memory. The identity that removes both is

        sum(X2.T @ X2) = sum_k ||x_k||^4

    since the (i, j) entry is ``sum_k x_ki^2 x_kj^2`` and summing over i and j
    factorises. So only the squared row norms are needed, which are O(n).
    Same value, one Gram product instead of two, no large temporary.

    Returns a scalar in ``[0, 1]``. The ``min(beta, delta)`` clamp is sklearn's
    and is load-bearing: without it the intensity can exceed 1 and invert the
    sign of the covariances.
    """
    x = np.asarray(residuals, dtype=np.float64)
    n_samples, n_features = x.shape

    sq_norms = np.einsum("ij,ij->i", x, x)  # ||x_k||^2, one per sample
    trace_s = float(sq_norms.sum()) / n_samples  # trace(S)
    mu = trace_s / n_features  # trace(S) / p

    # delta_ = ||S||_F^2 with S = X.T X / n
    gram = x.T @ x
    delta_ = float((gram**2).sum()) / n_samples**2

    beta_ = float((sq_norms**2).sum())
    beta_ = (beta_ / n_samples - delta_) / (n_features * n_samples)

    # delta = ||S - mu I||_F^2 / p, expanded so no p x p temporary is built
    delta = delta_ - 2.0 * mu * trace_s + n_features * mu**2
    delta /= n_features

    beta = min(beta_, delta)
    if beta <= 0.0 or delta <= 0.0:
        return 0.0
    return float(beta / delta)


def shrink_covariance(cov: np.ndarray, intensity: float) -> np.ndarray:
    """Apply scaled-identity shrinkage: ``(1 - d) * S + d * mu * I``.

    ``mu = trace(S) / p``, so the trace is preserved exactly for any intensity.
    That invariant is asserted in the tests and is the cheapest available check
    that the target is the identity-scaled one and not one of the other two
    Ledoit-Wolf targets (constant correlation, JPM 2004; single factor, JEF 2003).
    """
    cov = np.asarray(cov, dtype=np.float64)
    p = cov.shape[0]
    mu = float(np.trace(cov)) / p
    return (1.0 - intensity) * cov + intensity * mu * np.eye(p, dtype=np.float64)


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CovarianceFit:
    """A fitted Gaussian model: means, one covariance, and its diagnostics.

    ``class_means`` is ``None`` for the marginal model and ``(K, p)`` for the
    class-conditional one. ``mean`` is the global mean in both cases: the
    class-conditional model still needs it, because RMD's background term is
    scored against the marginal fit and because the PCA-residual scorer centres
    on ``mu_ID``.
    """

    mean: np.ndarray
    covariance: np.ndarray
    class_means: np.ndarray | None
    class_labels: np.ndarray | None
    diagonal: bool
    shrinkage: bool
    shrinkage_intensity: float
    n_samples: int
    name: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    _cholesky: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    @property
    def n_features(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def class_conditional(self) -> bool:
        return self.class_means is not None

    def whiten(self, x: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
        """Whiten each row of ``x``: ``L^-1 (x - mu)``, returned ``(n, p)``.

        Used by the PCA-residual scorer's whitening-before-projection
        variant, which is the pre-declared remedy if the class-mean
        residual turns out to be a rescaled embedding norm. It is the *same*
        transform ``mahalanobis_sq`` already applies -- that method is
        exactly ``sum(whiten(x) ** 2, axis=1)``, which
        ``tests/test_covariance_estimator.py`` pins so the two cannot drift.

        **This is a Cholesky whitening, not the symmetric one.** ``L^-1`` and
        ``Sigma^-1/2`` differ by a left orthogonal factor, since both satisfy
        ``W^T W = Sigma^-1``. That difference is invisible to what the PCA-residual
        variant does with it: fitting a PCA subspace and taking a projection
        residual are both equivariant under an orthogonal map applied to the
        fitting data and the scored data alike, so the residual norms are
        identical either way. Using the cached factor means the whitening costs
        a triangular solve rather than a second eigendecomposition, and it
        cannot disagree with the Mahalanobis distances computed from the same
        fit.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        mu = self.mean if mean is None else np.asarray(mean, dtype=np.float64)
        return np.linalg.solve(self._cholesky, (x - mu).T).T

    def mahalanobis_sq(self, x: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
        """Squared Mahalanobis distance of each row of ``x`` to ``mean``.

        Solved against the cached lower Cholesky factor ``L`` of the covariance:
        ``d^2 = ||L^-1 (x - mu)||^2``. The explicit inverse is never formed.
        ``numpy.linalg.solve`` is used on the triangular factor;
        ``scipy.linalg.solve_triangular`` would exploit the triangularity and is
        a safe drop-in, but is not used here so the module stays scipy-free and
        the suite runs with only numpy and pandas installed.

        Kept as its own implementation rather than ``(self.whiten(x) ** 2).sum``:
        the ``einsum`` below consumes the solve's ``(p, n)`` output in place,
        while ``whiten`` transposes it for callers that want the whitened
        vectors themselves. Same arithmetic, and the equality is asserted.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        mu = self.mean if mean is None else np.asarray(mean, dtype=np.float64)
        diff = x - mu
        y = np.linalg.solve(self._cholesky, diff.T)
        return np.einsum("ij,ij->j", y, y)

    def class_conditional_mahalanobis_sq(self, x: np.ndarray) -> np.ndarray:
        """Minimum squared Mahalanobis distance over the class means.

        This is the generative model behind LDA and is what Lee et al. (2018)
        and OpenOOD's ``MDSPostprocessor`` implement.
        """
        if self.class_means is None:
            raise ValueError(
                f"{self.name}: class_conditional_mahalanobis_sq needs a "
                f"class-conditional fit; this fit is marginal (labels were not passed)."
            )
        per_class = np.stack(
            [self.mahalanobis_sq(x, mean=mu_k) for mu_k in self.class_means], axis=1
        )
        return per_class.min(axis=1)


def _covariance_diagnostics(
    raw: np.ndarray,
    symmetrised: np.ndarray,
    final: np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    """Everything about a fitted covariance that is unavailable after the fact.

    Every fitted covariance must log ``np.allclose(S, S.T)`` and its condition
    number. Logging the symmetry check *after* symmetrising would be
    vacuous, so both sides are recorded: the raw asymmetry the estimator
    produced, and confirmation that symmetrisation fixed it.

    **Expect ``raw_max_asymmetry`` to be exactly 0.0 in every entry.** Measured at
    (400, 6), (5000, 192), (20000, 384) and the real (50000, 768),
    at one and four BLAS threads: ``R.T @ R`` is *bitwise* symmetric every time.
    Recording the raw asymmetry does not convert a vacuous check into a
    measurement on any tested BLAS: it converts one constant into another, so calling
    it a measurement overstates it. The field earns its place anyway: a
    non-zero value is the signal, this is a BLAS and threading property rather
    than a guarantee, and the point is that the signal would not be invisible.

    Note that a *hardcoded* ``0.0`` here would be indistinguishable from a
    measured one, exactly as for the symmetrisation itself; neither is
    behaviourally testable while the property holds. Both are guarded by
    inspection rather than by a test, and that is stated rather than implied.

    Eigenvalues come from ``eigh``, never ``eig`` -- that one *is* guarded, by
    ``tests/test_import_hygiene.py``'s AST scan for ``np.linalg.eig``.
    """
    asym = float(np.abs(raw - raw.T).max())
    eigvals = np.linalg.eigvalsh(final)
    lo, hi = float(eigvals.min()), float(eigvals.max())
    cond = float(hi / lo) if lo > 0 else float("inf")
    return {
        "name": name,
        "n_features": int(final.shape[0]),
        "raw_is_symmetric": bool(np.allclose(raw, raw.T)),
        "raw_max_asymmetry": asym,
        "is_symmetric": bool(np.allclose(symmetrised, symmetrised.T)),
        "eigenvalue_min": lo,
        "eigenvalue_max": hi,
        "condition_number": cond,
        "trace": float(np.trace(final)),
    }


def fit_covariance(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    diagonal: bool = False,
    shrinkage: bool = False,
    l2_normalize_first: bool = False,
    name: str = "covariance",
) -> CovarianceFit:
    """Fit one Gaussian model. The only covariance estimator here.

    Parameters
    ----------
    embeddings
        ``(n, p)`` ID *training* embeddings. Fit on train, always; evaluation is
        on test. Cast to float64 internally whatever comes in.
    labels
        ``None`` for the marginal model. ``(n,)`` integer class labels for the
        class-conditional model, which uses K means and one **pooled**
        within-class covariance.
    diagonal
        Restrict the fitted matrix to its diagonal. A restriction of this one
        path, not a separate estimator.
    shrinkage
        Apply Ledoit-Wolf scaled-identity shrinkage. Runtime flag, so on/off is
        one argument rather than a branch, and the selected intensity is
        recorded on the fit either way (0.0 when off).
    l2_normalize_first
        L2-normalise the embeddings before fitting. This is the RMD++ /
        Mahalanobis++ variant; the same normalisation must then be applied to
        anything scored against this fit, which
        ``GaussianScorer`` handles.

    Composition order when both ``diagonal`` and ``shrinkage`` are set
    -----------------------------------------------------------------
    **The order is immaterial, and it is not a decision.**

    Shrink-then-restrict and restrict-then-shrink produce **bit-identical**
    matrices from **bit-identical** intensities, for two independent reasons:

    * ``diag((1 - d) * S + d * mu * I) == (1 - d) * diag(S) + d * mu * I``,
      because the target is a scaled *identity* and ``mu = trace(S) / p`` is
      unchanged by the restriction: ``trace(diag(S)) == trace(S)`` exactly.
    * ``d`` itself is unchanged, because the intensity is estimated from
      ``residuals`` -- the ``(n, p)`` centred data -- and not from the fitted
      matrix. Nothing about that estimate can see the ``diagonal`` flag.

    Verified bitwise, ``max abs diff 0.0``, at three shapes.

    **The real decision, which is load-bearing, is the second bullet:** the
    intensity is estimated from the residual matrix rather than from the
    restricted covariance. *That* is what makes the order immaterial, and it is
    what makes the recorded intensities comparable across all four cells, so the
    shrinkage on/off ablation is a cleanly crossed factor rather than four
    unrelated one-offs. Estimating ``d`` from ``diag(S)`` instead would give a
    different number in the diagonal cells and the ordering would then matter.

    The code is written shrink-then-restrict because that keeps shrinkage a
    single flag applied at exactly one point on the shared path, which is what
    the design asks for ("one argument and not a branch"). That is a readability
    preference with no numerical consequence, not a methods choice.

    One thing worth stating rather than leaving implicit: the diagonal cells'
    matrix is not what ``LedoitWolf`` would return if handed a diagonal-only
    model. That object does not exist in this design -- diagonal is a
    restriction, not a model, so nothing is being approximated.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D (n, p) array, got shape {x.shape}")
    if l2_normalize_first:
        x = l2_normalize(x)

    n_samples, n_features = x.shape
    global_mean = x.mean(axis=0)

    if labels is None:
        class_means = None
        class_labels = None
        residuals = x - global_mean
    else:
        lab = np.asarray(labels)
        if lab.shape[0] != n_samples:
            raise ValueError(
                f"{name}: {lab.shape[0]} labels for {n_samples} embeddings"
            )
        class_labels = np.unique(lab)
        class_means = np.stack([x[lab == k].mean(axis=0) for k in class_labels])
        # Pooled within-class: one covariance from all class-centred residuals,
        # which is what draws on all 50,000 samples rather than 5,000 per class.
        residuals = np.empty_like(x)
        for mu_k, k in zip(class_means, class_labels):
            mask = lab == k
            residuals[mask] = x[mask] - mu_k

    # Divisor is n, not n - 1 and not n - K. This is the MLE, and it is what
    # sklearn's EmpiricalCovariance (hence Lee et al. 2018 and OpenOOD) uses.
    # It also matters beyond convention: a constant rescaling of a covariance is
    # AUROC-invariant within a single cell, but RMD subtracts two Mahalanobis
    # terms, so the two covariances must share a divisor or the difference is
    # scaled inconsistently.
    raw_cov = residuals.T @ residuals / n_samples

    symmetrised = 0.5 * (raw_cov + raw_cov.T)

    intensity = 0.0
    cov = symmetrised
    if shrinkage:
        intensity = ledoit_wolf_shrinkage(residuals)
        cov = shrink_covariance(cov, intensity)

    if diagonal:
        cov = np.diag(np.diag(cov))

    diagnostics = _covariance_diagnostics(raw_cov, symmetrised, cov, name=name)
    diagnostics.update(
        {
            "n_samples": int(n_samples),
            "samples_per_dimension": float(n_samples) / float(n_features),
            "model": "class_conditional_pooled" if labels is not None else "marginal",
            "n_classes": int(len(class_labels)) if class_labels is not None else 0,
            "diagonal": bool(diagonal),
            "shrinkage": bool(shrinkage),
            "shrinkage_intensity": float(intensity),
            "l2_normalized": bool(l2_normalize_first),
        }
    )

    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guard, not a path
        raise np.linalg.LinAlgError(
            f"{name}: fitted covariance is not positive definite "
            f"(condition number {diagnostics['condition_number']:.3e}, smallest "
            f"eigenvalue {diagnostics['eigenvalue_min']:.3e}). With n={n_samples} "
            f"and p={n_features} this should not happen on these cells; "
            f"if it does, a dimension has zero variance or shrinkage=True is "
            f"needed. Do not silently pseudo-invert."
        ) from exc

    return CovarianceFit(
        mean=global_mean,
        covariance=cov,
        class_means=class_means,
        class_labels=class_labels,
        diagonal=bool(diagonal),
        shrinkage=bool(shrinkage),
        shrinkage_intensity=float(intensity),
        n_samples=int(n_samples),
        name=name,
        diagnostics=diagnostics,
        _cholesky=chol,
    )
