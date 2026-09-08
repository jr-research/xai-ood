# tests/test_covariance_estimator.py
#
# Covers the shared covariance code path in
# src/xai_ood/methods/covariance.py: the four numerics requirements,
# the Ledoit-Wolf estimator, and the shrink-then-restrict composition order.
#
# Closed-form or hand-computed expectations, no mocking, numpy/pandas/pytest
# only.
#
# The fixtures use exact factorial designs rather than random draws wherever an
# exact expectation is claimed. Four points at (+-a, +-b) have mean exactly zero
# and covariance exactly diag(a^2, b^2) under the 1/n divisor, with the
# off-diagonal cancelling term for term. That makes "hand-computed expected
# distance" literal rather than approximate.
import json

import numpy as np
import pytest

from xai_ood.methods.covariance import (
    NORM_EPS,
    fit_covariance,
    l2_normalize,
    ledoit_wolf_shrinkage,
    shrink_covariance,
)


# --------------------------------------------------------------------------- #
# Fixtures with exactly-known moments
# --------------------------------------------------------------------------- #


def exact_diagonal_design(a=2.0, b=3.0):
    """Four points whose sample mean is 0 and covariance is exactly diag(a^2, b^2)."""
    return np.array([[a, b], [a, -b], [-a, b], [-a, -b]], dtype=np.float64)


def two_cluster_design(sep=5.0, a=0.3, b=0.4):
    """Two exact designs offset to (-sep, 0) and (+sep, 0).

    Pooled within-class covariance is exactly diag(a^2, b^2); the marginal
    covariance is exactly diag(a^2 + sep^2, b^2), which is the law of total
    covariance with a between-class term of sep^2 in dimension 0.
    """
    base = exact_diagonal_design(a, b)
    left = base + np.array([-sep, 0.0])
    right = base + np.array([sep, 0.0])
    x = np.vstack([left, right])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return x, labels


# --------------------------------------------------------------------------- #
# Numerics: float64, symmetrise, eigh, Cholesky
# --------------------------------------------------------------------------- #


def test_estimation_is_float64_whatever_the_input_dtype():
    """Estimate and invert in float64. The cache is float32."""
    x = exact_diagonal_design().astype(np.float32)
    fit = fit_covariance(x)
    assert fit.covariance.dtype == np.float64
    assert fit.mean.dtype == np.float64
    assert fit.mahalanobis_sq(x).dtype == np.float64


def test_fitted_covariance_is_exactly_symmetric_and_symmetry_is_logged():
    """Symmetrise before decomposing, and log allclose(S, S.T)."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(400, 6))
    fit = fit_covariance(x, name="marginal")

    # Exactly, not approximately: 0.5 * (S + S.T) is symmetric bitwise.
    assert np.array_equal(fit.covariance, fit.covariance.T)

    d = fit.diagnostics
    assert d["is_symmetric"] is True
    # Both sides are recorded. Logging allclose only *after* symmetrising would
    # be vacuous, so the raw asymmetry the estimator produced is kept too.
    assert "raw_is_symmetric" in d and "raw_max_asymmetry" in d
    assert d["raw_max_asymmetry"] >= 0.0


def test_condition_number_is_real_and_matches_eigh():
    """Condition number logged per covariance, via eigh not eig."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 5)) @ np.diag([5.0, 3.0, 1.0, 0.5, 0.2])
    fit = fit_covariance(x)

    eigvals = np.linalg.eigvalsh(fit.covariance)
    assert np.isrealobj(eigvals)
    expected = float(eigvals.max() / eigvals.min())
    assert fit.diagnostics["condition_number"] == pytest.approx(expected, rel=1e-12)
    assert fit.diagnostics["condition_number"] > 1.0


def test_mahalanobis_matches_explicit_inverse_without_forming_one():
    """The Cholesky solve must agree with the inverse it is avoiding.

    A Cholesky solve is preferred over an explicit inverse. That is a
    numerical-hygiene preference, not a change of answer, so the two must agree
    on a well-conditioned fixture.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(500, 4)) @ np.array(
        [[2.0, 0.5, 0.0, 0.0], [0.0, 1.5, 0.3, 0.0], [0.0, 0.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.8]]
    )
    fit = fit_covariance(x)
    query = rng.normal(size=(20, 4))

    diff = query - fit.mean
    explicit = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(fit.covariance), diff)
    assert fit.mahalanobis_sq(query) == pytest.approx(explicit, rel=1e-9)


def test_singular_covariance_raises_a_message_that_names_the_remedy():
    """A zero-variance dimension makes the diagonal baseline divide by zero.

    The extraction checks assert all per-dimension variances are
    strictly positive for exactly this reason. If one ever is not, the failure
    should name shrinkage rather than silently pseudo-inverting.
    """
    x = np.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        fit_covariance(x, diagonal=True, name="degenerate")


# --------------------------------------------------------------------------- #
# Marginal vs class-conditional: the law of total covariance
# --------------------------------------------------------------------------- #


def test_marginal_covariance_is_exactly_the_known_diagonal():
    x = exact_diagonal_design(a=2.0, b=3.0)
    fit = fit_covariance(x)
    assert fit.mean == pytest.approx(np.zeros(2), abs=0)
    assert fit.covariance == pytest.approx(np.diag([4.0, 9.0]), rel=1e-15)


def test_pooled_within_class_covariance_excludes_between_class_structure():
    """Sigma_marginal = Sigma_within + Sigma_between, exactly, on this fixture.

    This is the identity the whole 2x2 rests on: the two rows of the table are
    different geometries, not the same geometry translated. The marginal
    covariance absorbs the between-class term; the pooled one excludes it by
    construction.
    """
    sep, a, b = 5.0, 0.3, 0.4
    x, labels = two_cluster_design(sep, a, b)

    marginal = fit_covariance(x, None, name="marginal")
    pooled = fit_covariance(x, labels, name="pooled")

    within = np.diag([a**2, b**2])
    between = np.diag([sep**2, 0.0])

    assert pooled.covariance == pytest.approx(within, rel=1e-12)
    assert marginal.covariance == pytest.approx(within + between, rel=1e-12)
    assert marginal.covariance == pytest.approx(pooled.covariance + between, rel=1e-12)

    assert pooled.class_means == pytest.approx(np.array([[-sep, 0.0], [sep, 0.0]]), abs=1e-12)
    assert marginal.class_means is None


def test_class_conditional_fit_still_carries_the_global_mean():
    """RMD's background term and the PCA centring both need mu_ID."""
    x, labels = two_cluster_design()
    pooled = fit_covariance(x, labels)
    assert pooled.mean == pytest.approx(np.zeros(2), abs=1e-12)


def test_marginal_fit_refuses_class_conditional_scoring():
    x = exact_diagonal_design()
    fit = fit_covariance(x, name="marginal")
    with pytest.raises(ValueError, match="class-conditional"):
        fit.class_conditional_mahalanobis_sq(x)


# --------------------------------------------------------------------------- #
# Diagonal as a restriction of the shared path
# --------------------------------------------------------------------------- #


def test_diagonal_is_the_restriction_of_the_full_fit_not_a_separate_estimator():
    """The naive baseline is a diagonal restriction of this path."""
    rng = np.random.default_rng(19)
    latent = rng.normal(size=(600, 3))
    x = latent @ np.array([[1.0, 0.7, 0.2], [0.0, 1.0, 0.6], [0.0, 0.0, 1.0]])

    full = fit_covariance(x, diagonal=False)
    diag = fit_covariance(x, diagonal=True)

    assert diag.covariance == pytest.approx(np.diag(np.diag(full.covariance)), rel=1e-15)
    # And the restriction is not a no-op on correlated data, so the test above
    # is checking something.
    assert not np.allclose(diag.covariance, full.covariance)


def test_diagonal_restriction_is_a_no_op_when_the_data_is_already_uncorrelated():
    x = exact_diagonal_design()
    full = fit_covariance(x, diagonal=False)
    diag = fit_covariance(x, diagonal=True)
    assert diag.covariance == pytest.approx(full.covariance, rel=1e-15)
    assert diag.mahalanobis_sq(x) == pytest.approx(full.mahalanobis_sq(x), rel=1e-12)


# --------------------------------------------------------------------------- #
# Ledoit-Wolf, scaled-identity target
# --------------------------------------------------------------------------- #


def test_shrinkage_intensity_is_in_the_unit_interval():
    rng = np.random.default_rng(5)
    for n in (20, 200, 5000):
        x = rng.normal(size=(n, 8)) @ np.diag([4.0, 3.0, 2.0, 1.0, 0.9, 0.5, 0.3, 0.1])
        delta = ledoit_wolf_shrinkage(x - x.mean(axis=0))
        assert 0.0 <= delta <= 1.0


def test_the_min_beta_delta_clamp_binds_on_isotropic_data():
    """The clamp the docstring calls load-bearing, pinned by a case that needs it.

    `ledoit_wolf_shrinkage` clamps `beta = min(beta_, delta)`; without it the
    intensity can exceed 1 and invert the sign of the covariances. Every
    fixture above is strongly anisotropic, where the clamp never binds -- so
    deleting `min(...)` passes the entire suite.

    Isotropic data is where it binds: for `N(0, I)` the sample covariance is
    already almost exactly the shrinkage target, `delta -> 0`, and the estimator
    saturates at exactly 1.0. That is the clamp firing, and it is not a corner
    case -- it is what the estimator is supposed to do when the target is right.

    Mutation killed: `beta = min(beta_, delta)` -> `beta = beta_`.
    """
    isotropic = np.random.default_rng(4).normal(size=(20000, 6))
    intensity = ledoit_wolf_shrinkage(isotropic - isotropic.mean(axis=0))
    assert intensity == pytest.approx(1.0, rel=1e-12)


def test_shrinkage_preserves_the_trace_exactly():
    """The defining property of the scaled-identity target.

    tr((1-d)S + d*mu*I) = (1-d)tr(S) + d*p*(tr(S)/p) = tr(S).

    This is the cheapest available discriminator between the JMVA-2004 target
    this thesis uses and the other two estimators sharing the name
    (constant correlation, JPM 2004; single factor, JEF 2003).
    """
    rng = np.random.default_rng(23)
    x = rng.normal(size=(120, 6)) @ np.diag([3.0, 2.0, 1.5, 1.0, 0.5, 0.2])
    cov = np.cov(x, rowvar=False, ddof=0)
    for delta in (0.0, 0.25, 0.5, 1.0):
        shrunk = shrink_covariance(cov, delta)
        assert np.trace(shrunk) == pytest.approx(np.trace(cov), rel=1e-12)


def test_shrinkage_at_intensity_one_is_the_scaled_identity():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(120, 5))
    cov = np.cov(x, rowvar=False, ddof=0)
    mu = np.trace(cov) / cov.shape[0]
    assert shrink_covariance(cov, 1.0) == pytest.approx(mu * np.eye(5), rel=1e-12)
    assert shrink_covariance(cov, 0.0) == pytest.approx(cov, rel=1e-12)


def test_shrinkage_improves_conditioning():
    """The mechanism the ablation is testing, isolated.

    Pulling eigenvalues toward their mean cannot increase the spread, so the
    condition number falls whenever the intensity is non-zero. This is what
    "shrinkage would earn its keep on the per-class variant" means concretely.
    """
    rng = np.random.default_rng(31)
    x = rng.normal(size=(60, 12)) @ np.diag(np.linspace(5.0, 0.05, 12))

    plain = fit_covariance(x, shrinkage=False)
    shrunk = fit_covariance(x, shrinkage=True)

    assert shrunk.shrinkage_intensity > 0.0
    assert shrunk.diagnostics["condition_number"] < plain.diagnostics["condition_number"]


def test_shrinkage_intensity_falls_as_samples_per_dimension_rises():
    """The input-level argument the thesis makes, turned into an assertion.

    ~65 samples per dimension on these cells is why shrinkage is
    expected to be near-inert there and load-bearing only on the optional
    per-class level. The direction of that dependence is testable without the
    cache; the actual intensities are not, and are not asserted here.
    """
    rng = np.random.default_rng(37)
    cov_root = np.diag(np.linspace(3.0, 0.2, 10))

    scarce = rng.normal(size=(15, 10)) @ cov_root
    plentiful = rng.normal(size=(20000, 10)) @ cov_root

    delta_scarce = fit_covariance(scarce, shrinkage=True).shrinkage_intensity
    delta_plentiful = fit_covariance(plentiful, shrinkage=True).shrinkage_intensity

    assert delta_scarce > delta_plentiful
    assert delta_plentiful < 0.05


def test_shrinkage_off_records_a_zero_intensity_rather_than_omitting_it():
    """The ablation needs both arms in the manifest, not one arm and a gap."""
    x = exact_diagonal_design()
    fit = fit_covariance(x, shrinkage=False)
    assert fit.shrinkage is False
    assert fit.shrinkage_intensity == 0.0
    assert fit.diagnostics["shrinkage_intensity"] == 0.0


def _ledoit_wolf_from_the_paper(centred):
    """Ledoit & Wolf (2004, JMVA) section 3, written out with explicit outer
    products and the paper's normalised Frobenius norm ||A||^2 = tr(A A^T) / p.

    Deliberately naive and O(n p^2): this is the *independent route*, so it must
    not share an algebraic step with the implementation it checks. sklearn's
    (and therefore covariance.py's) formulation is an algebraic rearrangement
    that avoids ever building an outer product; this one builds them.

        m     = tr(S) / p
        d^2   = ||S - m I||^2
        b^2   = min( (1/n^2) sum_k ||x_k x_k^T - S||^2 , d^2 )
        delta = b^2 / d^2
    """
    n, p = centred.shape
    s = centred.T @ centred / n
    m = np.trace(s) / p
    d2 = np.sum((s - m * np.eye(p)) ** 2) / p
    b2_bar = sum(np.sum((np.outer(x, x) - s) ** 2) / p for x in centred) / n**2
    b2 = min(b2_bar, d2)
    return b2 / d2 if d2 > 0 else 0.0


@pytest.mark.parametrize("n,p", [(15, 10), (40, 12), (200, 15), (500, 8)])
def test_ledoit_wolf_matches_the_paper_formulas_computed_naively(n, p):
    """Verification of the reimplementation that does not depend on sklearn.

    The sklearn cross-check below is the right check but it cannot run without
    scikit-learn, which would leave the estimator unverified wherever it is
    absent. This closes that gap: agreement to machine precision against the paper's own
    formulas, computed by a route that shares no algebra with the fast one.
    """
    rng = np.random.default_rng(1000 + n)
    x = rng.normal(size=(n, p)) @ np.diag(np.linspace(4.0, 0.1, p))
    centred = x - x.mean(axis=0)
    assert ledoit_wolf_shrinkage(centred) == pytest.approx(
        _ledoit_wolf_from_the_paper(centred), rel=1e-12
    )


def test_ledoit_wolf_matches_scikit_learn_when_it_is_available():
    """Cross-check of the numpy reimplementation against the cited estimator.

    Skipped where scikit-learn is absent, and fires automatically wherever it is
    installed, which is where the equivalence actually needs to hold. sklearn
    splits the Gram products into blocks, so agreement is to floating-point
    noise, not exact.
    """
    sklearn_cov = pytest.importorskip(
        "sklearn.covariance", reason="scikit-learn is not installed here"
    )
    rng = np.random.default_rng(41)
    x = rng.normal(size=(200, 15)) @ np.diag(np.linspace(4.0, 0.1, 15))
    centred = x - x.mean(axis=0)

    ours = ledoit_wolf_shrinkage(centred)
    theirs = sklearn_cov.ledoit_wolf_shrinkage(centred, assume_centered=True)
    assert ours == pytest.approx(theirs, rel=1e-10)

    ref = sklearn_cov.LedoitWolf(assume_centered=True).fit(centred)
    fit = fit_covariance(x, shrinkage=True)
    assert fit.covariance == pytest.approx(ref.covariance_, rel=1e-10)


# --------------------------------------------------------------------------- #
# Shrinkage composition: the intensity comes from the residuals
# --------------------------------------------------------------------------- #
#
# A composition-order "decision" does not exist here: shrink-then-restrict and
# restrict-then-shrink give bit-identical matrices from bit-identical
# intensities, so a test named for the ordering pins nothing. These two check
# the property that *makes* the order immaterial, which is a real decision.


def test_the_diagonal_shrunk_cell_has_the_documented_closed_form():
    """diagonal + shrinkage == diag((1 - d) * diag(S) + d * mu).

    The closed form of the diagonal-plus-shrinkage cell, spelled out
    independently of the intensity and the unshrunk matrix. This pins the
    *form* of the result; it deliberately does not claim to pin an ordering,
    because both orderings produce this same matrix.
    """
    rng = np.random.default_rng(43)
    latent = rng.normal(size=(80, 7))
    x = latent @ np.triu(np.ones((7, 7))) * np.linspace(1.0, 0.3, 7)

    both = fit_covariance(x, diagonal=True, shrinkage=True)
    full_shrunk = fit_covariance(x, diagonal=False, shrinkage=True)

    assert both.shrinkage_intensity == pytest.approx(full_shrunk.shrinkage_intensity, rel=1e-15)
    assert both.covariance == pytest.approx(
        np.diag(np.diag(full_shrunk.covariance)), rel=1e-15
    )

    # Spelled out independently from the intensity and the unshrunk matrix.
    plain = fit_covariance(x, diagonal=False, shrinkage=False).covariance
    d = both.shrinkage_intensity
    mu = np.trace(plain) / plain.shape[0]
    assert np.diag(both.covariance) == pytest.approx(
        (1.0 - d) * np.diag(plain) + d * mu, rel=1e-12
    )


def test_intensity_is_estimated_from_the_residuals_not_the_restricted_matrix():
    """The load-bearing decision, and the reason the composition order is moot.

    The intensity is estimated from `residuals` -- the (n, p) centred data --
    and never from the fitted matrix, so nothing about the estimate can see the
    `diagonal` flag. Two consequences:

      * The recorded intensity is a property of the *row* of the factorial, not
        of the cell, which is what makes the shrinkage on/off ablation a cleanly
        crossed factor rather than four unrelated one-offs.
      * `diag((1-d) S + d mu I) == (1-d) diag(S) + d mu I` already holds because
        the target is a scaled identity and `trace(diag(S)) == trace(S)`; with
        `d` also unchanged, the two composition orders coincide bitwise. The
        order is therefore not a decision, and this is the property that makes
        it not one.

    Mutation killed: estimating the intensity from the restricted covariance
    (`ledoit_wolf_shrinkage` fed `diag(S)`-derived residuals) instead, which
    gives the diagonal cells a different `d` from the full cells.
    """
    x, labels = two_cluster_design()
    rng = np.random.default_rng(47)
    x = x + rng.normal(scale=0.01, size=x.shape)  # break the exact design

    marg_full = fit_covariance(x, None, diagonal=False, shrinkage=True)
    marg_diag = fit_covariance(x, None, diagonal=True, shrinkage=True)
    cc_full = fit_covariance(x, labels, diagonal=False, shrinkage=True)
    cc_diag = fit_covariance(x, labels, diagonal=True, shrinkage=True)

    assert marg_full.shrinkage_intensity == pytest.approx(marg_diag.shrinkage_intensity, rel=1e-15)
    assert cc_full.shrinkage_intensity == pytest.approx(cc_diag.shrinkage_intensity, rel=1e-15)


# --------------------------------------------------------------------------- #
# L2 normalisation
# --------------------------------------------------------------------------- #


def test_l2_normalize_guards_the_denominator_not_the_quotient():
    """x / (norm + eps), never OpenOOD's x / norm + eps.

    Their placement adds a constant to every component after dividing. Harmless
    at these magnitudes, but it is the documented explanation for any
    third-decimal disagreement with their published numbers, so it must not be
    reproduced here.
    """
    x = np.array([[3.0, 4.0]])
    got = l2_normalize(x)
    expected = x / (5.0 + NORM_EPS)
    assert got == pytest.approx(expected, rel=1e-15)

    # The discrepancy is about 1e-10 in absolute terms, which is precisely why
    # their placement is harmless but wrong. rtol must be 0
    # here, or np.allclose's default relative tolerance swallows the difference
    # and the test passes for the wrong reason.
    openood_quirk = x / 5.0 + NORM_EPS
    assert not np.allclose(got, openood_quirk, rtol=0.0, atol=1e-12)
    # Exactly: quirk_i - ours_i = eps * (1 + x_i / ||x||^2 * ||x||) to first
    # order, so the gap is of order eps and no larger than 2 * eps here.
    gap = float(np.abs(openood_quirk - got).max())
    assert NORM_EPS <= gap <= 2.0 * NORM_EPS


def test_l2_normalize_handles_a_zero_row_without_dividing_by_zero():
    got = l2_normalize(np.zeros((1, 4)))
    assert np.all(np.isfinite(got))
    assert got == pytest.approx(np.zeros((1, 4)), abs=0)


def test_fitting_on_normalized_embeddings_is_the_same_as_normalizing_first():
    rng = np.random.default_rng(53)
    x = rng.normal(size=(200, 5)) * np.linspace(1.0, 6.0, 5)

    inline = fit_covariance(x, l2_normalize_first=True)
    manual = fit_covariance(l2_normalize(x), l2_normalize_first=False)

    assert inline.covariance == pytest.approx(manual.covariance, rel=1e-12)
    assert inline.diagnostics["l2_normalized"] is True


# --------------------------------------------------------------------------- #
# Diagnostics are runtime values, not constants
# --------------------------------------------------------------------------- #


def test_diagnostics_are_json_serialisable_and_carry_every_required_key():
    x, labels = two_cluster_design()
    rng = np.random.default_rng(59)
    x = x + rng.normal(scale=0.02, size=x.shape)
    fit = fit_covariance(x, labels, shrinkage=True, name="pooled")

    blob = json.dumps(fit.diagnostics)
    round_tripped = json.loads(blob)
    for key in (
        "is_symmetric",
        "raw_is_symmetric",
        "raw_max_asymmetry",
        "condition_number",
        "eigenvalue_min",
        "eigenvalue_max",
        "shrinkage",
        "shrinkage_intensity",
        "n_samples",
        "n_features",
        "samples_per_dimension",
        "model",
        "n_classes",
    ):
        assert key in round_tripped, key
    assert round_tripped["model"] == "class_conditional_pooled"
    assert round_tripped["n_classes"] == 2


def test_samples_per_dimension_is_computed_not_asserted():
    """The ~65 figure is a property of the data, so it is derived, never typed."""
    rng = np.random.default_rng(61)
    x = rng.normal(size=(650, 10))
    fit = fit_covariance(x)
    assert fit.diagnostics["samples_per_dimension"] == pytest.approx(65.0, rel=1e-12)
