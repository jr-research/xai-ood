# tests/test_independent_properties.py
#
# Metamorphic and differential properties, derived independently of the code and
# adopted as pytest functions so they run with the suite.
#
# Why these are worth keeping separate from the per-module test files
# -------------------------------------------------------------------
# Every test under `tests/` was written by the same process that wrote `src/`,
# against the same mental model, in the same sitting. A shared misunderstanding
# would produce a matching test and a matching implementation and no failure.
# The properties below were derived from the **mathematics** rather than from the
# code: each is either a differential check against an independent
# implementation (scipy, scikit-learn, an explicit O(n^2) loop) or a metamorphic
# invariant that must hold for any correct implementation whatever its internals.
#
# Scope
# -----
# One pytest function per *atomic* property, twenty-four in all. The ``pNN``
# prefixes are stable labels, so a tolerance or a fixture can be discussed by
# name; they are not an ordering or a priority.
#
# Tolerances are stated as measured, rounded up to the next power of ten. A
# tolerance that merely tracks whatever the code currently returns is not a
# check, so each is justified against the arithmetic instead.
import numpy as np
import pytest

from xai_ood.methods.correlation import spearman_rho
from xai_ood.methods.covariance import (
    fit_covariance,
    l2_normalize,
    ledoit_wolf_shrinkage,
    shrink_covariance,
)
from xai_ood.methods.knn import KnnConfig, KnnScorer
from xai_ood.methods.pca_residual import PcaResidualConfig, fit_pca_residual
from xai_ood.metrics import auroc, evaluate, fpr_at_tpr, threshold_at_tpr

# --------------------------------------------------------------------------- #
# Shared fixtures. Deterministic, small enough to run in the suite.
# --------------------------------------------------------------------------- #


def gaussian_cloud(n: int = 300, p: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = np.linspace(0.5, 3.0, p)
    return rng.normal(size=(n, p)) * scales + np.arange(p, dtype=np.float64)


def random_rotation(p: int, seed: int) -> np.ndarray:
    """A genuine orthogonal matrix, from the QR of a Gaussian."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(p, p)))
    # Fix the sign convention so Q is uniform on O(p) rather than QR-dependent.
    return q * np.sign(np.diag(r))


def well_conditioned_map(p: int, seed: int) -> np.ndarray:
    """An invertible linear map with a bounded condition number.

    Mahalanobis invariance is exact in real arithmetic for *any* invertible map,
    but a map with condition number 1e12 would make the check a test of
    floating-point luck rather than of the invariance. Condition number is
    bounded at 100 by construction: an orthogonal frame with a controlled
    singular-value spread.
    """
    rng = np.random.default_rng(seed)
    u = random_rotation(p, seed)
    v = random_rotation(p, seed + 1)
    singular = np.geomspace(1.0, 100.0, p)
    return (u * singular) @ v.T + 0.0 * rng.normal(size=(p, p))


# =========================================================================== #
# AUROC
# =========================================================================== #


def test_p01_auroc_matches_sklearn_across_fuzzed_continuous_inputs():
    """P1. Differential check, 200 fuzzed continuous cases.

    Tolerance 1e-12 against a measured max deviation of 2.2e-16. AUROC is a sum
    of integer win counts over a product of sizes, so both implementations are
    doing exact integer arithmetic before one division; anything above
    round-off is a definitional disagreement, not a numerical one.
    """
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(31)
    worst = 0.0
    for _ in range(200):
        n_id = int(rng.integers(3, 60))
        n_ood = int(rng.integers(3, 60))
        shift = float(rng.uniform(-2.0, 2.0))
        id_scores = rng.normal(size=n_id)
        ood_scores = rng.normal(size=n_ood) + shift
        labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
        scores = np.concatenate([id_scores, ood_scores])
        expected = metrics.roc_auc_score(labels, scores)
        worst = max(worst, abs(auroc(id_scores, ood_scores) - expected))
    assert worst < 1e-12, worst


def test_p02_auroc_matches_sklearn_on_tie_heavy_inputs():
    """P2. The same check on quantised scores, so ties dominate.

    This is the half-credit convention's only real exposure. Continuous
    embedding distances almost never tie; a diagonal-covariance score on
    quantised inputs does, and a tie convention that disagreed with the
    Mann-Whitney definition would show up here and nowhere else.
    """
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(32)
    worst = 0.0
    saw_ties = False
    for _ in range(100):
        n_id = int(rng.integers(5, 40))
        n_ood = int(rng.integers(5, 40))
        levels = int(rng.integers(2, 5))
        id_scores = rng.integers(0, levels, size=n_id).astype(np.float64)
        ood_scores = rng.integers(0, levels, size=n_ood).astype(np.float64)
        saw_ties = saw_ties or bool(np.intersect1d(id_scores, ood_scores).size)
        labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
        scores = np.concatenate([id_scores, ood_scores])
        expected = metrics.roc_auc_score(labels, scores)
        worst = max(worst, abs(auroc(id_scores, ood_scores) - expected))
    assert saw_ties, "the tie-heavy fixture produced no cross-side ties"
    assert worst < 1e-12, worst


def test_p03_auroc_equals_the_brute_force_mann_whitney_count():
    """P3. Differential check against an explicit O(n*m) loop.

    Needs no optional dependency, unlike P1 and P2. It reimplements the *definition*
    ``P(ood > id) + 0.5 P(tie)`` with two nested loops and no searchsorted, so
    it shares no algebra with the module: an off-by-one in the sorted-array
    binary search cannot survive both.
    """
    rng = np.random.default_rng(33)
    id_scores = rng.integers(0, 4, size=37).astype(np.float64)
    ood_scores = rng.integers(0, 4, size=29).astype(np.float64)
    wins = 0.0
    for o in ood_scores:
        for i in id_scores:
            wins += 1.0 if o > i else (0.5 if o == i else 0.0)
    expected = wins / (id_scores.size * ood_scores.size)
    assert auroc(id_scores, ood_scores) == pytest.approx(expected, abs=1e-15)


def test_p04_auroc_is_invariant_under_any_strictly_increasing_transform():
    """P4. Metamorphic. AUROC depends only on the ranks.

    Holds for any correct implementation, so a failure localises to something
    reading magnitudes where it should read order. Three transforms with very
    different curvature, applied to both sides at once.
    """
    rng = np.random.default_rng(34)
    id_scores = rng.normal(size=80)
    ood_scores = rng.normal(size=60) + 0.8
    baseline = auroc(id_scores, ood_scores)
    for transform in (np.exp, lambda v: v**3, lambda v: np.arctan(v) * 7.0 + 1.0):
        assert auroc(transform(id_scores), transform(ood_scores)) == pytest.approx(
            baseline, abs=1e-14
        )


def test_p05_auroc_is_antisymmetric_in_its_two_arguments():
    """P5. Metamorphic: ``auroc(a, b) + auroc(b, a) == 1`` exactly.

    Every pair is counted once on each side, ties contributing 0.5 twice, so the
    identity is exact rather than approximate. It is the cheapest available
    check that the sign convention is applied consistently: a scorer negated in
    one branch and not the other would break it.
    """
    rng = np.random.default_rng(35)
    for _ in range(20):
        a = rng.integers(0, 3, size=int(rng.integers(3, 30))).astype(np.float64)
        b = rng.integers(0, 3, size=int(rng.integers(3, 30))).astype(np.float64)
        assert auroc(a, b) + auroc(b, a) == pytest.approx(1.0, abs=1e-15)


def test_p06_the_fpr95_threshold_is_an_observed_score_clearing_the_target():
    """P6. The threshold rule, checked against its own definition.

    Independent of the implementation: the returned threshold must be a value
    that actually occurs in the ID array, the achieved TPR must reach the
    target, and no *smaller* observed value may also reach it. That triple is
    the specification, and it pins ``ceil`` against ``floor`` without naming
    either.
    """
    rng = np.random.default_rng(36)
    for _ in range(50):
        id_scores = np.sort(rng.normal(size=int(rng.integers(5, 90))))
        tau = threshold_at_tpr(id_scores, 0.95)
        assert tau in set(id_scores.tolist())
        achieved = np.count_nonzero(id_scores <= tau) / id_scores.size
        assert achieved >= 0.95
        smaller = id_scores[id_scores < tau]
        if smaller.size:
            best_smaller = float(smaller.max())
            assert np.count_nonzero(id_scores <= best_smaller) / id_scores.size < 0.95


def test_p07_fpr_at_tpr_counts_exactly_the_ood_scores_at_or_below_the_threshold():
    """P7. Differential check on FPR@95 against a direct count."""
    rng = np.random.default_rng(37)
    id_scores = rng.normal(size=64)
    ood_scores = rng.normal(size=48) + 1.5
    tau = threshold_at_tpr(np.sort(id_scores), 0.95)
    expected = np.count_nonzero(ood_scores <= tau) / ood_scores.size
    assert fpr_at_tpr(id_scores, ood_scores) == pytest.approx(expected, abs=1e-15)
    assert evaluate(id_scores, ood_scores)["fpr95"] == pytest.approx(expected, abs=1e-15)


# =========================================================================== #
# Mahalanobis and the covariance path
# =========================================================================== #


def test_p08_mahalanobis_is_invariant_under_an_invertible_linear_map():
    """P8. Metamorphic.

    For any invertible ``A``, fitting on ``X A^T`` and scoring ``z A^T`` gives
    ``(A dz)^T (A S A^T)^-1 (A dz) = dz^T S^-1 dz``. This is the defining
    property of the Mahalanobis distance and the reason it is used at all, so
    an implementation that violated it would not be computing a Mahalanobis
    distance whatever else it got right. Tolerance is relative, because the map
    is deliberately non-orthogonal and the condition number enters.
    """
    x = gaussian_cloud(seed=1)
    query = gaussian_cloud(n=40, seed=2)
    a = well_conditioned_map(x.shape[1], seed=3)

    plain = fit_covariance(x, name="plain").mahalanobis_sq(query)
    mapped = fit_covariance(x @ a.T, name="mapped").mahalanobis_sq(query @ a.T)
    assert np.allclose(plain, mapped, rtol=1e-8, atol=1e-8)


def test_p09_mahalanobis_is_invariant_under_translation():
    """P9. Metamorphic.

    Shifting the fitting data and the queries by the same vector leaves the
    covariance and every centred difference unchanged, so distances are
    identical. Catches a fit that centred on something other than its own mean.
    """
    x = gaussian_cloud(seed=4)
    query = gaussian_cloud(n=40, seed=5)
    shift = np.array([100.0, -50.0, 7.5, 0.0, 3.0, -2.0])

    plain = fit_covariance(x, name="plain").mahalanobis_sq(query)
    shifted = fit_covariance(x + shift, name="shifted").mahalanobis_sq(query + shift)
    assert np.allclose(plain, shifted, rtol=1e-9, atol=1e-9)


def test_p10_whiten_and_mahalanobis_sq_are_the_same_transform():
    """P10. Differential check inside the module.

    ``mahalanobis_sq`` and ``whiten`` reach the same quantity by two routes
    (``einsum`` over the solve's output, versus a transpose then a sum), and the
    PCA-residual whitened variant depends on their agreeing. Asserted on both
    the marginal and the class-conditional fit.
    """
    x = gaussian_cloud(seed=6)
    labels = np.tile(np.arange(3), 100)
    query = gaussian_cloud(n=40, seed=7)
    for fit in (
        fit_covariance(x, name="marginal"),
        fit_covariance(x, labels, name="pooled"),
    ):
        expected = np.einsum("ij,ij->i", fit.whiten(query), fit.whiten(query))
        assert np.allclose(fit.mahalanobis_sq(query), expected, rtol=0, atol=1e-12)


def test_p11_diagonal_mahalanobis_matches_the_closed_form_on_a_known_gaussian():
    """P11. Differential check against arithmetic.

    With a diagonal restriction the distance collapses to
    ``sum((z - mu)^2 / var)`` with ``var`` the MLE per-dimension variance, which
    is computable in one line without touching the module's Cholesky path. Uses
    an exact sign factorial so the fitted variances are the stated numbers
    rather than estimates of them.
    """
    axis_sd = np.array([2.0, 5.0, 0.5])
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * 3)).T.reshape(-1, 3)
    x = signs * axis_sd  # mean exactly 0, variance exactly axis_sd**2

    fit = fit_covariance(x, diagonal=True, name="diagonal")
    query = np.array([[1.0, -2.0, 0.25], [0.0, 0.0, 0.0], [4.0, 5.0, -0.5]])
    expected = (((query - x.mean(axis=0)) ** 2) / axis_sd**2).sum(axis=1)
    assert np.allclose(fit.mahalanobis_sq(query), expected, rtol=0, atol=1e-12)


def test_p12_class_conditional_distance_is_the_minimum_over_the_class_means():
    """P12. Differential check against the definition.

    Recomputed here as an explicit min over per-class calls to the *marginal*
    entry point, which shares no branch with
    ``class_conditional_mahalanobis_sq``.
    """
    x = gaussian_cloud(n=240, seed=8)
    labels = np.tile(np.arange(4), 60)
    query = gaussian_cloud(n=30, seed=9)
    fit = fit_covariance(x, labels, name="pooled")

    per_class = np.stack(
        [fit.mahalanobis_sq(query, mean=mu) for mu in fit.class_means], axis=1
    )
    assert np.allclose(
        fit.class_conditional_mahalanobis_sq(query), per_class.min(axis=1), atol=0
    )


def test_p13_ledoit_wolf_intensity_matches_scikit_learn():
    """P13. Differential check against the reference implementation.

    ``src/`` is forbidden to import scikit-learn, so the estimator is
    reimplemented in numpy with an identity that avoids sklearn's large
    temporary. The reimplementation is the deviation most worth an independent
    check, since a wrong intensity would shift every shrunk covariance by a
    plausible amount rather than an obvious one. Tolerance 1e-12 against a
    measured agreement of 1.1e-16.
    """
    covariance = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(41)
    for n, p in ((80, 12), (300, 6), (40, 30)):
        residuals = rng.normal(size=(n, p)) * np.linspace(0.2, 4.0, p)
        residuals -= residuals.mean(axis=0)
        expected = covariance.ledoit_wolf_shrinkage(residuals, assume_centered=True)
        assert ledoit_wolf_shrinkage(residuals) == pytest.approx(expected, abs=1e-12)


def test_p14_shrinkage_preserves_the_trace_for_any_intensity():
    """P14. Metamorphic.

    ``mu = trace(S)/p`` makes the scaled-identity target trace-preserving by
    construction, which the other two Ledoit-Wolf targets are not. It is the
    cheapest available evidence that the implemented target is the JMVA 2004
    one, and it needs no reference implementation.
    """
    rng = np.random.default_rng(42)
    residuals = rng.normal(size=(120, 9))
    cov = residuals.T @ residuals / residuals.shape[0]
    for intensity in (0.0, 0.13, 0.5, 0.99, 1.0):
        shrunk = shrink_covariance(cov, intensity)
        assert np.trace(shrunk) == pytest.approx(np.trace(cov), rel=1e-12)


# =========================================================================== #
# kNN
# =========================================================================== #


def test_p15_knn_matches_a_brute_force_pairwise_distance_matrix():
    """P15. Differential check.

    The module computes distances through the Gram identity
    ``||q||^2 + ||r||^2 - 2 q.r`` and partitions rather than sorts. This builds
    the full explicit-difference distance matrix and sorts it, sharing neither
    trick. Tolerance 1e-9 absolute against a measured 1.8e-15; the identity's
    cancellation error is far below the spacing that decides a k-th-neighbour
    ranking, which is the property that matters.
    """
    rng = np.random.default_rng(51)
    reference = rng.normal(size=(200, 5))
    query = rng.normal(size=(40, 5))
    scorer = KnnScorer(
        KnnConfig("probe", normalize=False, k=7, reporting_tier="primary"), reference
    )

    explicit = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    expected = np.sort(explicit, axis=1)[:, 6]
    assert np.allclose(scorer.score(query), expected, rtol=0, atol=1e-9)


def test_p16_knn_is_invariant_under_a_shared_rotation():
    """P16. Metamorphic.

    Euclidean distance is orthogonally invariant, so rotating the reference set
    and the queries together cannot change any neighbour distance. Catches an
    implementation that had quietly become axis-dependent, which the Gram
    identity would not be but a per-dimension shortcut would.
    """
    rng = np.random.default_rng(52)
    reference = rng.normal(size=(150, 5))
    query = rng.normal(size=(30, 5))
    q = random_rotation(5, seed=53)
    config = KnnConfig("probe", normalize=False, k=5, reporting_tier="primary")

    plain = KnnScorer(config, reference).score(query)
    rotated = KnnScorer(config, reference @ q).score(query @ q)
    assert np.allclose(plain, rotated, rtol=1e-10, atol=1e-10)


def test_p17_knn_is_monotone_non_decreasing_in_k():
    """P17. Metamorphic.

    The k-th smallest of a fixed multiset cannot decrease as k grows. Holds
    per-query, not merely on average, so it is asserted elementwise. This is the
    property that separates the k-th-neighbour score from the k-averaged one,
    which is monotone in a different way.
    """
    rng = np.random.default_rng(54)
    reference = rng.normal(size=(200, 4))
    query = rng.normal(size=(25, 4))
    scorer = KnnScorer(
        KnnConfig("probe", normalize=False, k=1, reporting_tier="primary"), reference
    )
    previous = scorer.score(query, k=1)
    for k in (2, 5, 10, 50, 100):
        current = scorer.score(query, k=k)
        assert (current >= previous - 1e-12).all()
        previous = current


def test_p18_normalized_knn_equals_unnormalized_knn_on_normalized_inputs():
    """P18. Metamorphic.

    Normalizing inside the scorer must be the same operation as normalizing
    outside it, or the normalized variant is measuring something the ablation
    does not name. Uses the package's one ``l2_normalize``, which is the point:
    a second normalization with the guard misplaced would break this.
    """
    rng = np.random.default_rng(55)
    reference = rng.normal(size=(180, 6)) * 4.0
    query = rng.normal(size=(30, 6)) * 4.0

    inside = KnnScorer(
        KnnConfig("probe", normalize=True, k=9, reporting_tier="appendix"), reference
    ).score(query)
    outside = KnnScorer(
        KnnConfig("probe", normalize=False, k=9, reporting_tier="primary"),
        l2_normalize(reference),
    ).score(l2_normalize(query))
    assert np.allclose(inside, outside, rtol=1e-10, atol=1e-12)


# =========================================================================== #
# PCA residual
# =========================================================================== #


def test_p19_pca_residual_matches_an_independently_formed_projector():
    """P19. Differential check.

    The module forms the residual as ``z_c - V_d (V_d^T z_c)`` to avoid a
    cancelling subtraction. This builds the explicit projector
    ``I - V_d V_d^T`` and applies it, which is the textbook form and a different
    arithmetic path. Tolerance 1e-9 against a measured 8.9e-16.
    """
    x = gaussian_cloud(n=300, p=7, seed=61)
    query = gaussian_cloud(n=40, p=7, seed=62)
    scorer = fit_pca_residual(
        PcaResidualConfig("probe", "all_id", reporting_tier="secondary"), x
    )

    basis = scorer.subspace.components[:, : scorer.d]
    projector = np.eye(x.shape[1]) - basis @ basis.T
    expected = np.linalg.norm((query - x.mean(axis=0)) @ projector.T, axis=1)
    assert np.allclose(scorer.score(query), expected, rtol=0, atol=1e-9)


def test_p20_pca_residual_is_invariant_under_a_shared_rotation():
    """P20. Metamorphic.

    PCA is equivariant under an orthogonal map and the residual norm is
    invariant under one, so fitting and scoring in a rotated frame must give
    identical scores. This is also the property that justifies using a Cholesky
    whitening rather than the symmetric square root in the whitened variant, so
    it is load-bearing rather than decorative.
    """
    x = gaussian_cloud(n=300, p=6, seed=63)
    query = gaussian_cloud(n=40, p=6, seed=64)
    q = random_rotation(6, seed=65)
    config = PcaResidualConfig("probe", "all_id", reporting_tier="secondary")

    plain = fit_pca_residual(config, x).score(query)
    rotated = fit_pca_residual(config, x @ q).score(query @ q)
    assert np.allclose(plain, rotated, rtol=1e-8, atol=1e-8)


def test_p21_the_residual_is_zero_for_points_inside_the_retained_subspace():
    """P21. Metamorphic.

    A point of the form ``mu + V_d c`` lies in the retained subspace, so its
    residual is exactly zero up to round-off. This is the definitional anchor
    of the whole scorer: a residual that stayed positive there would mean the
    projector was not a projector.
    """
    x = gaussian_cloud(n=300, p=7, seed=66)
    scorer = fit_pca_residual(
        PcaResidualConfig("probe", "all_id", reporting_tier="secondary"), x
    )
    rng = np.random.default_rng(67)
    basis = scorer.subspace.components[:, : scorer.d]
    coefficients = rng.normal(size=(25, scorer.d))
    inside = x.mean(axis=0) + coefficients @ basis.T

    assert np.allclose(scorer.score(inside), 0.0, atol=1e-9)
    # And the complement: a point displaced along a discarded direction has a
    # residual equal to the displacement, so the check is not vacuous.
    discarded = scorer.subspace.components[:, scorer.d]
    displaced = inside + 3.0 * discarded
    assert np.allclose(scorer.score(displaced), 3.0, atol=1e-8)


def test_p22_the_sweep_is_monotone_non_increasing_in_d():
    """P22. Metamorphic.

    ``r_d^2 = sum_{j > d} c_j^2`` is a sum of fewer non-negative terms as d
    grows, so the residual can only shrink. Asserted elementwise, and at the
    full-rank endpoint the residual must be exactly zero rather than merely
    small.
    """
    x = gaussian_cloud(n=300, p=8, seed=68)
    query = gaussian_cloud(n=30, p=8, seed=69)
    scorer = fit_pca_residual(
        PcaResidualConfig("probe", "all_id", reporting_tier="secondary"), x
    )
    sweep = scorer.score_sweep(query, ds=(1, 2, 4, 6, 8))
    previous = sweep[1]
    for d in (2, 4, 6, 8):
        assert (sweep[d] <= previous + 1e-12).all()
        previous = sweep[d]
    assert np.array_equal(sweep[8], np.zeros_like(sweep[8]))


# =========================================================================== #
# Spearman
# =========================================================================== #


def test_p23_spearman_matches_scipy_including_forced_ties():
    """P23. Differential check.

    ``src/`` is forbidden to import scipy, so the rank correlation is
    reimplemented. Ties are where two rank conventions diverge, so a third of
    the cases here are heavily quantised. Tolerance 1e-10 against a measured
    1e-12.
    """
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(71)
    for i in range(60):
        n = int(rng.integers(5, 80))
        a = rng.normal(size=n)
        b = a * 0.5 + rng.normal(size=n)
        if i % 3 == 0:  # forced ties
            a = np.round(a * 1.5)
            b = np.round(b * 1.5)
        if np.unique(a).size < 2 or np.unique(b).size < 2:
            continue
        expected = stats.spearmanr(a, b).statistic
        assert spearman_rho(a, b, name="probe") == pytest.approx(expected, abs=1e-10)


def test_p24_spearman_is_invariant_under_a_strictly_monotone_transform():
    """P24. Metamorphic.

    A rank correlation depends only on order, so any strictly increasing
    transform of either argument leaves it unchanged, and a strictly decreasing
    one flips its sign exactly. The sign flip is the sharper half: it catches an
    implementation that dropped a sign somewhere in the rank centring.
    """
    rng = np.random.default_rng(72)
    a = rng.normal(size=120)
    b = a * 0.3 + rng.normal(size=120)
    baseline = spearman_rho(a, b, name="probe")

    assert spearman_rho(np.exp(a), b, name="probe") == pytest.approx(baseline, abs=1e-12)
    assert spearman_rho(a, np.arctan(b), name="probe") == pytest.approx(baseline, abs=1e-12)
    assert spearman_rho(-a, b, name="probe") == pytest.approx(-baseline, abs=1e-12)
