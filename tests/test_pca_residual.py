# tests/test_pca_residual.py
#
# The two-subspace PCA projection residual.
#
# Closed-form unit tests on synthetic data, as for the six Gaussian
# configurations and the two kNN variants.
#
# Expectations are hand-computed in the docstring or closed-form, no mocking,
# numpy/pandas/pytest only. The central
# fixture is a **full sign-factorial**, so the fitted second-moment matrix is
# exactly diagonal with exactly the stated entries rather than an estimate of
# them -- the same trick the Gaussian tests use, and it is what makes "the residual is
# hypot(5, 12) = 13" an arithmetic claim rather than an approximate one.
#
# Two real numbers are deliberately absent: the integer d under the 95% rule and
# the degeneracy rho. Both can only be measured once the embeddings exist. What
# is tested here is the *function* that produces each, against fixtures whose
# answers are known by construction -- never against a remembered value.
import itertools

import numpy as np
import pandas as pd
import pytest

from xai_ood.methods.correlation import spearman_rho
from xai_ood.methods.covariance import fit_covariance, l2_normalize
from xai_ood.methods.dump import score_frame
from xai_ood.methods.gaussian import fit_scorer
from xai_ood.methods.manifest import run_manifest
from xai_ood.methods.pca_residual import (
    CONFIGURATIONS,
    DEGENERACY_RHO_THRESHOLD,
    SUBSPACES,
    SWEEP_D,
    VARIANCE_THRESHOLD,
    PcaResidualConfig as _PcaResidualConfig,
    PcaResidualScorer,
    config_by_name,
    degeneracy_report,
    explained_variance_ratio,
    fit_all_pca_residual,
    fit_pca_residual,
    select_dimension_by_variance,
)
from xai_ood.schema import CANONICAL_SORT_KEY, decode_hyperparams


def PcaResidualConfig(*args, reporting_tier: str = "secondary", **kwargs):
    """Test-local constructor that defaults the reporting tier.

    ``reporting_tier`` is a required keyword-only field, so a scorer added
    later cannot slip in untiered. The fixtures
    below build throwaway configurations whose tier is irrelevant to anything
    they assert, so the default lives here rather than at forty call sites. The
    five *reported* configurations declare their tiers explicitly in
    ``pca_residual.CONFIGURATIONS``, and ``tests/test_reporting_tiers.py`` is
    what checks those against the declaration.
    """
    return _PcaResidualConfig(*args, reporting_tier=reporting_tier, **kwargs)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

#: Per-axis standard deviations of the exact factorial fixture. Distinct, so the
#: eigenvectors are the coordinate axes uniquely up to sign; equal values would
#: leave a degenerate eigenspace and any rotation within it would be a valid
#: basis, which would make "the retained subspace is span(e0, e1)" false.
AXIS_SD = np.array([10.0, 5.0, 2.0, 1.0])


def exact_factorial() -> np.ndarray:
    """16 rows: every sign combination of AXIS_SD. Moments are exact, not estimated.

    Mean is exactly the zero vector: every coordinate takes each sign equally
    often. The second-moment matrix is exactly ``diag(100, 25, 4, 1)``: the
    diagonal because each coordinate is +/- sd in every row, and the
    off-diagonal because for any two coordinates the four sign pairs occur
    equally often and cancel. Verified in
    ``test_the_factorial_fixture_has_exactly_the_moments_it_claims``, which is
    not a tautology -- it is what every closed-form expectation below rests on.
    """
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(AXIS_SD))))
    return signs * AXIS_SD


def clustered(
    *,
    n_features: int,
    n_classes: int = 10,
    per_class: int = 120,
    separation: float = 3.0,
    noise: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A K-cluster ID cloud with isotropic within-class noise."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), per_class)
    means = rng.normal(size=(n_classes, n_features)) * separation
    x = means[labels] + rng.normal(size=(n_classes * per_class, n_features)) * noise
    return x, labels


def index_frame(n: int, split: str = "id_test") -> pd.DataFrame:
    """A canonical-key index frame for ``n`` rows, deliberately out of order.

    ``corruption`` and ``severity`` are null-valued rather than absent, per the
    schema, so the four-column sort key stays total on a non-cs-ID frame.
    """
    return pd.DataFrame(
        {
            "split": split,
            "image_id": [f"img_{i:04d}" for i in range(n)][::-1],
            "corruption": pd.Series([None] * n, dtype="string"),
            "severity": pd.Series([pd.NA] * n, dtype="Int64"),
        }
    )


def test_the_factorial_fixture_has_exactly_the_moments_it_claims():
    """Guards the guard: every closed-form expectation below depends on this."""
    x = exact_factorial()
    assert x.shape == (16, 4)
    assert np.array_equal(x.mean(axis=0), np.zeros(4))
    second_moment = x.T @ x / len(x)
    assert np.array_equal(second_moment, np.diag(AXIS_SD**2))


# --------------------------------------------------------------------------- #
# The variance rule -- the function that produces the integer d
# --------------------------------------------------------------------------- #


def test_explained_variance_ratio_is_a_cumulative_fraction():
    """[4, 3, 2, 1] totals 10, so the cumulative fractions are .4 .7 .9 1.0."""
    assert explained_variance_ratio([4.0, 3.0, 2.0, 1.0]) == pytest.approx(
        [0.4, 0.7, 0.9, 1.0]
    )


def test_explained_variance_ratio_is_invariant_to_the_covariance_divisor():
    """Scaling every eigenvalue leaves the ratios untouched.

    This is why it does not matter that ``fit_covariance`` divides by n (the
    MLE) rather than n - 1, and why ``_pca_basis`` may use the same divisor
    without affecting which d the rule selects.
    """
    spectrum = np.array([9.0, 4.0, 2.0, 1.0])
    for factor in (1e-6, 0.5, 1.0, 7.0, 1e6):
        assert explained_variance_ratio(spectrum * factor) == pytest.approx(
            explained_variance_ratio(spectrum)
        )


def test_variance_rule_picks_the_smallest_d_reaching_the_threshold():
    """[80, 12, 5, 3]: cumulative .80 .92 .97 1.00. First >= .95 is d = 3."""
    assert select_dimension_by_variance([80.0, 12.0, 5.0, 3.0]) == 3


def test_variance_rule_boundary_is_inclusive():
    """[95, 5] explains exactly 95% at d = 1, so d = 1, not 2.

    95/100 and the literal 0.95 are the same double, so this lands exactly on
    the boundary rather than near it, and it is the comparison operator being
    tested rather than a floating-point accident.
    """
    assert 95.0 / 100.0 == 0.95  # the boundary really is exact here
    assert select_dimension_by_variance([95.0, 5.0]) == 1


def test_variance_rule_just_below_the_boundary_takes_one_more_component():
    """[94.9, 5.1]: .949 < .95, so the first component is not enough."""
    assert select_dimension_by_variance([94.9, 5.1]) == 2


def test_variance_rule_returns_one_when_the_leading_component_dominates():
    assert select_dimension_by_variance([99.0, 0.5, 0.3, 0.2]) == 1


def test_variance_rule_returns_every_component_on_a_flat_spectrum():
    """Twenty equal eigenvalues: .05 each, so 19 give .95 exactly.

    An inclusive boundary makes this 19 rather than 20 -- the same operator as
    the boundary test above, checked where it has a visible consequence.
    """
    assert select_dimension_by_variance([1.0] * 20) == 19


def test_variance_rule_respects_a_different_threshold():
    """[4, 3, 2, 1] again: .4 .7 .9 1.0. At 0.7 the answer is 2, at 0.9 it is 3."""
    assert select_dimension_by_variance([4.0, 3.0, 2.0, 1.0], threshold=0.7) == 2
    assert select_dimension_by_variance([4.0, 3.0, 2.0, 1.0], threshold=0.9) == 3
    assert select_dimension_by_variance([4.0, 3.0, 2.0, 1.0], threshold=1.0) == 4


def test_variance_rule_rejects_a_threshold_outside_zero_to_one():
    for bad in (0.0, -0.1, 1.5, 95.0):
        with pytest.raises(ValueError, match="threshold must be in"):
            select_dimension_by_variance([4.0, 3.0, 2.0, 1.0], threshold=bad)


def test_variance_rule_rejects_a_degenerate_spectrum():
    with pytest.raises(ValueError, match="total variance is zero"):
        select_dimension_by_variance([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="negative eigenvalue"):
        select_dimension_by_variance([1.0, -0.5])
    with pytest.raises(ValueError, match="no eigenvalues"):
        select_dimension_by_variance([])


def test_the_committed_threshold_is_the_ninety_five_percent_rule():
    """The rule is pre-declared."""
    assert VARIANCE_THRESHOLD == 0.95


def test_d_is_selected_from_the_fitted_spectrum_and_not_hardcoded():
    """The same code selects different d for different data. That is the point.

    d can only be measured once the embeddings exist. What can be tested
    without them is that it is a *function of the eigenspectrum*, so
    two fixtures with deliberately different spectra must give different
    answers. A constant would pass every other test in this file.
    """
    x = exact_factorial()
    # Spectrum (100, 25, 4, 1), total 130. Cumulative .769 .961 -> d = 2.
    assert fit_pca_residual(PcaResidualConfig("t", "all_id"), x).d == 2

    # Flatten it: (16, 9, 4, 1), total 30. Cumulative .533 .833 .967 -> d = 3.
    flatter = np.array(list(itertools.product([-1.0, 1.0], repeat=4))) * np.array(
        [4.0, 3.0, 2.0, 1.0]
    )
    assert fit_pca_residual(PcaResidualConfig("t", "all_id"), flatter).d == 3


# --------------------------------------------------------------------------- #
# The score, closed form
# --------------------------------------------------------------------------- #


def test_the_all_id_subspace_is_the_leading_coordinate_axes():
    """With variances (100, 25, 4, 1) the principal axes are e0..e3 in order."""
    fit = fit_pca_residual(PcaResidualConfig("t", "all_id"), exact_factorial()).subspace
    assert fit.eigenvalues == pytest.approx(AXIS_SD**2)
    # Columns are eigenvectors; signs are arbitrary, so compare absolute values.
    assert np.abs(fit.components) == pytest.approx(np.eye(4), abs=1e-12)


def test_residual_is_the_norm_of_the_discarded_coordinates():
    """Fixture: axes e0..e3, mean 0. Query z = (3, 4, 5, 12).

    r_2(z) discards e2 and e3, leaving hypot(5, 12) = 13 exactly.
    r_1(z) discards e1, e2, e3: norm(4, 5, 12) = sqrt(16 + 25 + 144) = sqrt(185).
    r_3(z) discards e3 alone: 12.
    All three are exact, because the subspace is exactly the coordinate axes.
    """
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), exact_factorial())
    z = np.array([[3.0, 4.0, 5.0, 12.0]])
    assert scorer.score(z, d=1) == pytest.approx([np.sqrt(185.0)])
    assert scorer.score(z, d=2) == pytest.approx([13.0])
    assert scorer.score(z, d=3) == pytest.approx([12.0])


def test_the_fitted_default_d_is_the_one_score_uses():
    """d = 2 under the 95% rule on this fixture, so score() == score(d=2)."""
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), exact_factorial())
    z = np.array([[3.0, 4.0, 5.0, 12.0]])
    assert scorer.d == 2
    assert scorer.score(z) == pytest.approx(scorer.score(z, d=2))


def test_retaining_every_component_gives_exactly_zero_residual():
    """At d = p the projector is the identity, so (I - P_d) annihilates everything.

    The top of SWEEP_D is exactly this point. It is a legitimate end of the
    sensitivity curve -- the score carries no information there -- and it is
    computed as an exact zero rather than as a cancellation of large numbers,
    which is why score() forms the residual directly.
    """
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), exact_factorial())
    z = np.array([[3.0, 4.0, 5.0, 12.0], [-1.0, 0.0, 7.0, 2.0]])
    assert scorer.score(z, d=4) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_the_residual_is_centred_on_mu_id_not_on_the_origin():
    """Shift the whole fixture; residuals must not move.

    r_d(z) = ||(I - P_d)(z - mu_ID)||, so translating the training data and the
    query by the same vector leaves every score unchanged. A scorer that forgot
    to centre, or that centred on zero, would fail this.
    """
    x = exact_factorial()
    z = np.array([[3.0, 4.0, 5.0, 12.0]])
    shift = np.array([100.0, -7.0, 0.5, 33.0])
    plain = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    shifted = fit_pca_residual(PcaResidualConfig("t", "all_id"), x + shift)
    assert shifted.mean == pytest.approx(shift)
    assert shifted.score(z + shift) == pytest.approx(plain.score(z))


def test_the_residual_never_increases_as_more_components_are_retained():
    """Nested subspaces: P_d subspace contains P_{d-1}'s, so r_d <= r_{d-1}.

    Monotonicity is a structural property of a *nested* PCA basis. It would
    break if the components were ever reordered, or if d indexed into the
    ascending eigh output without the reversal.
    """
    x, _ = clustered(n_features=12, seed=1)
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    query = x[:50]
    previous = scorer.score(query, d=1)
    for d in range(2, 13):
        current = scorer.score(query, d=d)
        assert np.all(current <= previous + 1e-12)
        previous = current


def test_residual_of_a_point_in_the_retained_subspace_is_zero():
    """A query on span(e0, e1) has nothing left after projection."""
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), exact_factorial())
    assert scorer.score(np.array([[6.0, -2.0, 0.0, 0.0]]), d=2) == pytest.approx(
        [0.0], abs=1e-12
    )


def test_a_one_dimensional_query_is_accepted_and_returns_one_score():
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), exact_factorial())
    assert scorer.score(np.array([3.0, 4.0, 5.0, 12.0]), d=2) == pytest.approx([13.0])


# --------------------------------------------------------------------------- #
# Sign convention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=lambda c: c.name)
def test_every_configuration_scores_a_far_away_point_as_most_ood(config):
    """Higher = more OOD, for every reported variant. No exceptions.

    The OOD point is pushed along a *minor-variance* direction, which is what a
    residual scorer is built to catch: displacing it along a retained principal
    direction would leave the residual untouched, and rightly so.
    """
    x, labels = clustered(n_features=20, seed=2)
    scorer = fit_pca_residual(config, x, labels)
    minor = scorer.subspace.components[:, -1]
    # The L2 variants normalise, so a displacement must survive normalisation:
    # scale it relative to the cloud rather than adding a fixed offset.
    ood = (x[:1] + 40.0 * np.linalg.norm(x[0]) * minor)
    assert scorer.score(ood)[0] > scorer.score(x).max()


def test_scores_are_non_negative_because_they_are_norms():
    x, labels = clustered(n_features=20, seed=3)
    for config in CONFIGURATIONS:
        scores = fit_pca_residual(config, x, labels).score(x)
        assert np.all(scores >= 0.0)
        assert scores.dtype == np.float64


# --------------------------------------------------------------------------- #
# The class-mean subspace
# --------------------------------------------------------------------------- #


def test_class_mean_subspace_retains_k_minus_one_components():
    """K = 10 -> d = 9, every available component. Nothing to sweep."""
    x, labels = clustered(n_features=30, n_classes=10, seed=4)
    scorer = fit_pca_residual(PcaResidualConfig("t", "class_mean"), x, labels)
    assert scorer.d == 9
    assert scorer.subspace.n_components_available == 9
    assert scorer.diagnostics()["n_residual_directions"] == 30 - 9


@pytest.mark.parametrize("n_classes", [2, 3, 5, 10])
def test_the_centred_class_mean_matrix_has_rank_at_most_k_minus_one(n_classes):
    """Rank <= K - 1 for **any** class balance, not only a balanced one.

    Because mu_ID is the global mean, sum_k n_k (mu_k - mu_ID) = 0 is a linear
    dependency among the rows of the centred class-mean matrix whatever the n_k
    are. This is why d = K - 1 is the whole span rather than an approximation of
    it, and the unbalanced case is included because CIFAR-10 happens to be
    exactly balanced and would hide a rule that only worked there.
    """
    rng = np.random.default_rng(5)
    sizes = [40 + 17 * k for k in range(n_classes)]  # deliberately unequal
    labels = np.repeat(np.arange(n_classes), sizes)
    means = rng.normal(size=(n_classes, 25)) * 4.0
    x = means[labels] + rng.normal(size=(sum(sizes), 25)) * 0.5

    scorer = fit_pca_residual(PcaResidualConfig("t", "class_mean"), x, labels)
    assert scorer.d == n_classes - 1
    assert scorer.subspace.n_components_available <= n_classes - 1


def test_class_mean_subspace_requires_labels():
    x, _ = clustered(n_features=12, seed=6)
    with pytest.raises(ValueError, match="labels are required"):
        fit_pca_residual(PcaResidualConfig("t", "class_mean"), x)


def test_class_mean_subspace_needs_at_least_two_classes():
    x, labels = clustered(n_features=12, n_classes=10, seed=7)
    with pytest.raises(ValueError, match="at least 2 classes"):
        fit_pca_residual(PcaResidualConfig("t", "class_mean"), x, np.zeros_like(labels))


def test_the_all_id_subspace_ignores_labels():
    """The all-ID component is label-free, like kNN and the marginal cells."""
    x, labels = clustered(n_features=12, seed=8)
    with_labels = fit_pca_residual(PcaResidualConfig("t", "all_id"), x, labels)
    without = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    assert with_labels.d == without.d
    assert with_labels.score(x) == pytest.approx(without.score(x))


def test_the_two_subspaces_are_genuinely_different_scores():
    """Reported separately and never combined; they must not coincide."""
    x, labels = clustered(n_features=30, seed=9)
    all_id = fit_pca_residual("pca_residual_all_id", x, labels).score(x)
    class_mean = fit_pca_residual("pca_residual_class_mean", x, labels).score(x)
    assert not np.allclose(all_id, class_mean)


def test_the_all_id_subspace_refuses_whitening():
    """Whitening the all-ID cloud destroys the subspace it would then project onto.

    Whitening maps the ID cloud's covariance to *exactly the identity*, so the
    "principal" subspace of that cloud is degenerate: every eigenvalue is 1, and
    ``eigh`` returns a basis determined by floating-point noise rather than by
    the data. Measured before this guard existed, at p = 40: spectrum max/min
    1.000000/1.000000 with std 3e-15, and permuting the input rows -- a no-op
    for PCA on a cloud -- changed the scores so completely that the refit
    correlated **0.17 with itself**. The shipped ``class_mean_whitened`` variant
    under the same permutation agrees to 1.15e-14.

    So "five reported configurations, not six" is not merely a consequence of
    whitening being pre-declared for the class-mean case. It is mathematically
    forced, and the combination has to raise rather than quietly return noise
    shaped like a score.
    """
    with pytest.raises(ValueError, match="identity"):
        PcaResidualConfig("pca_residual_all_id_whitened", "all_id", whiten=True)


def test_whitening_the_class_mean_subspace_is_still_allowed():
    """The guard above must not touch the pre-declared remedy."""
    config = PcaResidualConfig("t", "class_mean", whiten=True)
    assert config.whiten is True
    assert config.subspace == "class_mean"


class _CombinedSubspaceScorer:
    """The mutant the "never combined" guardrail names: one number from both.

    Not a ``PcaResidualConfig`` -- a combining variant would most plausibly
    arrive as a scorer summing two fitted residuals rather than as a new flag,
    which is exactly why a guard that inspects only ``CONFIGURATIONS``'
    ``subspace`` strings cannot see it.
    """

    def __init__(self, all_id, class_mean):
        self.all_id = all_id
        self.class_mean = class_mean
        self.config = all_id.config

    @property
    def name(self):
        return "pca_residual_combined"

    def score(self, x):
        return self.all_id.score(x) + self.class_mean.score(x)


def test_a_combining_scorer_is_rejected():
    """Guard-the-guard: the mutant must actually be caught.

    This is the shape of a test that passes for a reason other than the stated
    one. A guard asserting only
    that both subspaces appear among ``CONFIGURATIONS`` and that each config's
    subspace is in ``SUBSPACES`` cannot fail: the second is already enforced by
    ``PcaResidualConfig.__post_init__``, and the first only says both subspaces
    appear *somewhere*. A scorer summing the two residuals into one number --
    precisely what "reported separately and never combined" forbids -- passes it
    untouched.
    """
    x, labels = clustered(n_features=30, seed=9)
    combined = _CombinedSubspaceScorer(
        fit_pca_residual(PcaResidualConfig("a", "all_id"), x, labels),
        fit_pca_residual(PcaResidualConfig("b", "class_mean"), x, labels),
    )
    # It produces real, plausible-looking scores; nothing about it is obviously
    # broken, which is what makes it the right mutant.
    assert np.all(np.isfinite(combined.score(x)))

    with pytest.raises(AssertionError, match="never combined"):
        _assert_each_scorer_matches_exactly_one_subspace([combined], x, labels)


def test_no_configuration_combines_the_two_subspaces():
    """A guardrail turned into a test: the two components are never merged.

    "Reported separately rather than combined into a single number" is a
    declared decision, and this asserts the property that would actually break
    if they were ever merged: **every reported scorer's score must rank
    the samples exactly as one of the two subspaces does, and not as the other.**
    A combining scorer matches neither, and is caught -- see
    ``test_a_combining_scorer_is_rejected``, which drives this same check over
    the mutant.
    """
    x, labels = clustered(n_features=30, seed=9)
    _assert_each_scorer_matches_exactly_one_subspace(
        [fit_pca_residual(c, x, labels) for c in CONFIGURATIONS], x, labels
    )


def _assert_each_scorer_matches_exactly_one_subspace(scorers, x, labels):
    """Two exact-value properties, neither assuming the score is order-preserving.

    1. **A scorer is exactly what its own config says it is.** Refitting from
       ``scorer.config`` alone must reproduce its scores. Anything drawing on a
       second fitted subspace cannot survive a refit from a config naming one.
    2. **No scorer is a blend of the two.** Its scores must not coincide with
       the sum or the mean of the two subspaces' residuals.

    Ordering equality was tried first and is *wrong*: the ``normalize`` ablation
    legitimately reorders samples relative to its own unnormalized subspace, so
    a rank check fails for ``pca_residual_all_id_l2`` for reasons that have
    nothing to do with combining. That failure is why this compares values.
    """
    residuals = {
        subspace: fit_pca_residual(
            PcaResidualConfig(f"ref_{subspace}", subspace), x, labels
        ).score(x)
        for subspace in SUBSPACES
    }
    blends = {
        "the sum of both subspaces": residuals["all_id"] + residuals["class_mean"],
        "the mean of both subspaces": 0.5
        * (residuals["all_id"] + residuals["class_mean"]),
    }

    for scorer in scorers:
        scores = scorer.score(x)

        refitted = fit_pca_residual(scorer.config, x, labels).score(x)
        assert scores == pytest.approx(refitted), (
            f"{scorer.name}: refitting from its own config does not reproduce "
            f"its scores, so the scorer draws on something its config does not "
            f"name -- a second subspace being the case 'reported separately, "
            f"never combined' forbids."
        )

        for description, blend in blends.items():
            assert not np.allclose(scores, blend), (
                f"{scorer.name} scores exactly as {description}. The two "
                f"components are reported separately and never combined."
            )


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def test_sweep_agrees_with_individual_scores():
    """One projection pass must give what eight separate calls would.

    The two take different routes on purpose: score() forms the residual vector
    directly, the sweep sums squared coefficients from the tail of the spectrum.
    Equality is therefore to floating-point tolerance rather than bitwise, and
    the tolerance is relative because the residuals span orders of magnitude
    across the grid.
    """
    x, _ = clustered(n_features=32, per_class=60, seed=10)
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    grid = [1, 2, 4, 8, 16, 31, 32]
    swept = scorer.score_sweep(x, grid)
    for d in grid:
        assert swept[d] == pytest.approx(scorer.score(x, d=d), rel=1e-9, abs=1e-9)


def test_sweep_block_size_does_not_change_the_scores():
    """Blocking is a memory knob, exactly as in knn._QUERY_BLOCK."""
    x, _ = clustered(n_features=20, per_class=40, seed=11)
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    small = scorer.score_sweep(x, [4, 8], block=7)
    large = scorer.score_sweep(x, [4, 8], block=10_000)
    for d in (4, 8):
        assert small[d] == pytest.approx(large[d])


def test_sweep_is_monotone_non_increasing_in_d():
    x, _ = clustered(n_features=24, per_class=50, seed=12)
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    grid = [1, 2, 4, 8, 16, 24]
    swept = scorer.score_sweep(x, grid)
    for lower, higher in zip(grid, grid[1:]):
        assert np.all(swept[higher] <= swept[lower] + 1e-9)


def test_sweep_at_full_rank_is_exactly_zero():
    x, _ = clustered(n_features=16, per_class=40, seed=13)
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    assert scorer.score_sweep(x, [16])[16] == pytest.approx(np.zeros(len(x)), abs=1e-12)


def test_the_committed_sweep_grid_is_the_declared_one():
    """Pre-declared, and never used to reselect d."""
    assert SWEEP_D == (16, 32, 64, 128, 256, 384, 512, 768)


def test_sweep_rejects_a_dimension_larger_than_the_embedding():
    x, _ = clustered(n_features=20, per_class=40, seed=14)
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    with pytest.raises(ValueError, match="exceeds the embedding dimension"):
        scorer.score_sweep(x, SWEEP_D)


def test_score_rejects_an_out_of_range_d():
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), exact_factorial())
    with pytest.raises(ValueError, match="must be at least 1"):
        scorer.score(np.zeros((1, 4)), d=0)
    with pytest.raises(ValueError, match="exceeds the embedding dimension"):
        scorer.score(np.zeros((1, 4)), d=5)


def test_score_rejects_a_dimension_mismatch():
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), exact_factorial())
    with pytest.raises(ValueError, match="input has 3 dimensions"):
        scorer.score(np.zeros((2, 3)))


# --------------------------------------------------------------------------- #
# The L2 ablation
# --------------------------------------------------------------------------- #


def test_the_l2_variant_uses_the_shared_normalization():
    """Not a second implementation: the scorer must agree with l2_normalize.

    The kNN tests make this exact assertion, and for the same reason -- a
    private reimplementation would pass a "does it look normalised" test while
    silently differing from RMD++ in the tenth decimal. Here the check is that
    scoring raw embeddings under the L2 config equals scoring the shared
    function's output under the unnormalised one.
    """
    x, labels = clustered(n_features=18, seed=15)
    normalized_config = fit_pca_residual("pca_residual_all_id_l2", x, labels)
    on_prenormalized = fit_pca_residual(
        PcaResidualConfig("t", "all_id"), l2_normalize(x), labels
    )
    assert normalized_config.d == on_prenormalized.d
    assert normalized_config.score(x) == pytest.approx(on_prenormalized.score(l2_normalize(x)))


def test_the_l2_ablation_is_a_separate_variant_not_a_substitution():
    """Unnormalized is primary. Both ship, under different names, with different scores."""
    x, labels = clustered(n_features=18, seed=16)
    primary = fit_pca_residual("pca_residual_all_id", x, labels)
    ablation = fit_pca_residual("pca_residual_all_id_l2", x, labels)
    assert primary.name != ablation.name
    assert primary.config.normalize is False
    assert ablation.config.normalize is True
    assert not np.allclose(primary.score(x), ablation.score(x))


def test_the_l2_variant_scores_are_scale_invariant():
    """L2 normalisation discards magnitude, so scaling a query cannot move its score.

    The mirror of the finding that the normalized kNN variant is blind
    to a pure-norm outlier: the same blindness is present here by construction,
    and it is the mechanism to look at first if the primary and the ablation
    diverge on real embeddings.
    """
    x, labels = clustered(n_features=18, seed=17)
    scorer = fit_pca_residual("pca_residual_class_mean_l2", x, labels)
    query = x[:20]
    assert scorer.score(query * 7.5) == pytest.approx(scorer.score(query))


# --------------------------------------------------------------------------- #
# Whitening: the pre-declared remedy, as a separately labeled variant
# --------------------------------------------------------------------------- #


def test_whiten_is_the_same_transform_as_the_mahalanobis_solve():
    """CovarianceFit.whiten and .mahalanobis_sq must not drift apart.

    ``whiten`` exists in covariance.py for this scorer. It is only safe to have
    because it is provably the same triangular solve the Mahalanobis distance
    already performs -- ``mahalanobis_sq`` is exactly
    the squared row norms of ``whiten``. Asserted here rather than in
    test_covariance_estimator.py because the method belongs to this family and
    the covariance file is left as it was.
    """
    x, _ = clustered(n_features=15, seed=18)
    fit = fit_covariance(x, None, name="t")
    whitened = fit.whiten(x)
    assert whitened.shape == x.shape
    assert np.einsum("ij,ij->i", whitened, whitened) == pytest.approx(fit.mahalanobis_sq(x))


def test_whitening_makes_the_id_cloud_isotropic():
    """The defining property: the whitened training data has identity covariance."""
    x, _ = clustered(n_features=15, seed=19)
    whitened = fit_covariance(x, None, name="t").whiten(x)
    assert whitened.mean(axis=0) == pytest.approx(np.zeros(15), abs=1e-10)
    assert whitened.T @ whitened / len(x) == pytest.approx(np.eye(15), abs=1e-9)


def test_cholesky_whitening_gives_the_same_residuals_as_symmetric_whitening():
    """The docstring claim, checked rather than asserted in prose.

    L^-1 and Sigma^-1/2 differ by a left orthogonal factor, and fitting a PCA
    subspace then taking a projection residual is equivariant under an
    orthogonal map applied to the fitting data and the scored data alike. So
    using the cached Cholesky factor rather than forming Sigma^-1/2 -- which
    would cost a second eigendecomposition -- cannot change a single score.
    """
    x, labels = clustered(n_features=16, seed=20)
    cholesky_variant = fit_pca_residual(
        PcaResidualConfig("t", "class_mean", whiten=True), x, labels
    )

    # Symmetric whitening, built here in the test only.
    fit = fit_covariance(x, None, name="t")
    eigenvalues, eigenvectors = np.linalg.eigh(fit.covariance)
    inverse_root = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    symmetric = (x - fit.mean) @ inverse_root.T

    reference = fit_pca_residual(PcaResidualConfig("t", "class_mean"), symmetric, labels)
    assert reference.score(symmetric) == pytest.approx(cholesky_variant.score(x), rel=1e-8)


def test_whitening_ships_as_a_separate_variant_never_a_silent_substitution():
    """The primary keeps its name, its configuration and its scores.

    The guardrail is that whitening-before-projection is reported as a
    separately labeled variant, not substituted in when the degeneracy fires.
    So: distinct names, distinct scores, and the primary's ``whiten`` flag is
    False whatever the degeneracy report says.
    """
    x, labels = clustered(n_features=24, seed=21)
    primary = fit_pca_residual("pca_residual_class_mean", x, labels)
    remedy = fit_pca_residual("pca_residual_class_mean_whitened", x, labels)
    assert primary.name == "pca_residual_class_mean"
    assert remedy.name == "pca_residual_class_mean_whitened"
    assert primary.config.whiten is False
    assert remedy.config.whiten is True
    assert primary.whitening_fit is None
    assert remedy.whitening_fit is not None
    assert not np.allclose(primary.score(x), remedy.score(x))


def test_a_whitened_config_without_a_fit_and_the_reverse_are_both_rejected():
    """Constructing the scorer by hand cannot produce a mismatched pair."""
    x, labels = clustered(n_features=12, seed=22)
    fitted = fit_pca_residual(PcaResidualConfig("t", "class_mean", whiten=True), x, labels)
    with pytest.raises(ValueError, match="needs a whitening fit"):
        PcaResidualScorer(
            PcaResidualConfig("t", "class_mean", whiten=True),
            fitted.subspace,
            9,
            mean=fitted.mean,
        )
    with pytest.raises(ValueError, match="would silently ignore"):
        PcaResidualScorer(
            PcaResidualConfig("t", "class_mean"),
            fitted.subspace,
            9,
            mean=fitted.mean,
            whitening_fit=fitted.whitening_fit,
        )


# --------------------------------------------------------------------------- #
# Baseline 1: the degeneracy control
# --------------------------------------------------------------------------- #


def test_the_degeneracy_threshold_is_the_pre_declared_one():
    assert DEGENERACY_RHO_THRESHOLD == 0.95


def test_degeneracy_fires_when_the_residual_is_essentially_the_centred_norm():
    """Barely-separated classes in 300 dimensions: rho lands above 0.95.

    This is the failure mode the control exists to detect, built deliberately.
    The mechanism is geometric rather than incidental: the class-mean subspace
    removes only K - 1 = 9 directions, so with p large the residual retains
    291/300 of the centred vector and its norm tracks the centred norm almost
    exactly. **At the real p = 768 that ratio is 759/768**, which is why this is
    worth measuring on the cache rather than assuming it away.

    The rho asserted here is a property of this fixture, not a prediction of
    the real number -- which is unknown until the run.
    """
    x, labels = clustered(
        n_features=300, per_class=120, separation=0.05, noise=1.0, seed=23
    )
    scorer = fit_pca_residual("pca_residual_class_mean", x, labels)
    report = degeneracy_report(scorer, x)
    assert report["rho_centred_norm"] > DEGENERACY_RHO_THRESHOLD
    assert report["degenerate"] is True
    assert report["verdict"] == "rescaled_norm_score"
    assert report["remedy"] == "whitening_before_projection"


def test_degeneracy_does_not_fire_when_the_subspace_carries_real_structure():
    """Well-separated classes and low noise: the residual is not the norm."""
    x, labels = clustered(
        n_features=60, per_class=120, separation=20.0, noise=0.5, seed=24
    )
    scorer = fit_pca_residual("pca_residual_class_mean", x, labels)
    report = degeneracy_report(scorer, x)
    assert report["rho_centred_norm"] < DEGENERACY_RHO_THRESHOLD
    assert report["degenerate"] is False
    assert report["verdict"] == "subspace_structure"
    assert report["remedy"] is None


def test_whitening_relieves_a_degeneracy_caused_by_id_anisotropy():
    """The remedy works on the mechanism it targets, on a fixture built for it.

    Class means live in the leading coordinates; the ID cloud has one enormous
    nuisance direction outside the class-mean subspace. Unwhitened, both the
    residual and the centred norm are dominated by that direction, so the
    control fires. Whitening rescales it to unit variance and the residual goes
    back to reflecting subspace structure.

    **This is not a claim that whitening will relieve the real degeneracy.**
    It demonstrates that the remedy acts on the stated mechanism. What the real
    embeddings do is a run-time measurement, and if whitening does not help
    there, that is a finding to report rather than a bug here.

    The fixture's constants are chosen structurally, not by seed-shopping. The
    class separation must survive the estimation noise the nuisance direction
    injects into the class means: at ``per_class`` samples that noise is
    ``nuisance / sqrt(per_class)``, here 40 / sqrt(500) = 1.8 against a
    separation of 5. Below that margin the fitted class-mean subspace starts
    absorbing the nuisance direction, the residual then *excludes* it, and the
    degeneracy stops appearing at all -- which is a real effect and not one to
    hide behind a lucky seed. Measured across seeds 0-7 at these constants:
    0.977-0.989 unwhitened, 0.085-0.116 whitened.
    """
    rng = np.random.default_rng(25)
    n_features, n_classes, per_class = 80, 10, 500
    labels = np.repeat(np.arange(n_classes), per_class)
    means = np.zeros((n_classes, n_features))
    means[:, : n_classes - 1] = rng.normal(size=(n_classes, n_classes - 1)) * 5.0
    scale = np.full(n_features, 0.4)
    scale[-1] = 40.0  # the nuisance direction
    x = means[labels] + rng.normal(size=(n_classes * per_class, n_features)) * scale

    plain = degeneracy_report(fit_pca_residual("pca_residual_class_mean", x, labels), x)
    whitened = degeneracy_report(
        fit_pca_residual("pca_residual_class_mean_whitened", x, labels), x
    )
    assert plain["degenerate"] is True
    assert whitened["degenerate"] is False
    assert whitened["rho_centred_norm"] < plain["rho_centred_norm"]


def test_degeneracy_report_uses_the_centred_norm_for_its_verdict():
    """The centred norm is the sharper control and is what the threshold applies to.

    Both correlations are reported -- the plain-norm one is the weaker check --
    but the pre-declared rule names the centred one, and the report says so in a
    field rather than only in prose.
    """
    x, labels = clustered(n_features=60, per_class=100, seed=26)
    report = degeneracy_report(fit_pca_residual("pca_residual_class_mean", x, labels), x)
    assert report["threshold_applies_to"] == "rho_centred_norm"
    assert set(report) >= {"rho_plain_norm", "rho_centred_norm", "threshold", "verdict"}


def test_degeneracy_report_is_a_function_of_the_data_not_a_stored_number():
    """Two fixtures, two different rhos, from the same code.

    rho is one of the three runtime-measured quantities: a function that returns
    a number, not a number. That is what this distinguishes -- a constant would
    pass the threshold tests above on one fixture and fail on the other.
    """
    degenerate, labels_a = clustered(
        n_features=300, per_class=120, separation=0.05, seed=27
    )
    structured, labels_b = clustered(
        n_features=60, per_class=120, separation=20.0, noise=0.5, seed=28
    )
    rho_a = degeneracy_report(
        fit_pca_residual("pca_residual_class_mean", degenerate, labels_a), degenerate
    )["rho_centred_norm"]
    rho_b = degeneracy_report(
        fit_pca_residual("pca_residual_class_mean", structured, labels_b), structured
    )["rho_centred_norm"]
    assert rho_a > rho_b + 0.5


def test_rho_rises_as_fewer_components_are_retained():
    """The degeneracy is a function of d, monotonically, and that is the mechanism.

    On isotropic data the residual keeps ``p - d`` of ``p`` directions, so the
    smaller ``d`` is, the closer ``r_d`` comes to the centred norm and the
    higher rho climbs. Measured here at p = 100: rho is 0.977 at d = 1, 0.922 at
    d = 4, and 0.36 at d = 50.

    This is the whole reason the class-mean component is the one under
    suspicion and the all-ID component is not. The class-mean subspace is stuck
    at d = K - 1 = 9 out of 768 by the rank of the centred class-mean matrix,
    which is the far left of this curve; the all-ID subspace takes whatever the
    95% rule gives, which will be far larger.
    """
    rng = np.random.default_rng(29)
    x = rng.normal(size=(600, 100))
    scorer = fit_pca_residual(PcaResidualConfig("t", "all_id"), x)
    centred_norm = np.linalg.norm(x - scorer.mean, axis=1)

    rhos = [
        spearman_rho(scorer.score(x, d=d), centred_norm) for d in (1, 2, 4, 8, 50)
    ]
    assert rhos[0] > 0.95
    assert rhos[-1] < 0.5
    assert rhos == sorted(rhos, reverse=True)


def test_degeneracy_report_rejects_a_one_dimensional_input():
    x, labels = clustered(n_features=12, seed=30)
    scorer = fit_pca_residual("pca_residual_class_mean", x, labels)
    with pytest.raises(ValueError, match="expected a 2-D"):
        degeneracy_report(scorer, x[0])


# --------------------------------------------------------------------------- #
# Configuration surface and the manifest
# --------------------------------------------------------------------------- #


def test_the_reported_configurations_are_the_five_committed_ones():
    """Unnormalized primary per subspace, an L2 ablation of each, one remedy."""
    assert [c.name for c in CONFIGURATIONS] == [
        "pca_residual_all_id",
        "pca_residual_class_mean",
        "pca_residual_all_id_l2",
        "pca_residual_class_mean_l2",
        "pca_residual_class_mean_whitened",
    ]


def test_config_names_are_unique():
    names = [c.name for c in CONFIGURATIONS]
    assert len(set(names)) == len(names)


def test_config_by_name_round_trips_and_rejects_an_unknown_name():
    for config in CONFIGURATIONS:
        assert config_by_name(config.name) is config
    with pytest.raises(KeyError, match="unknown PCA-residual configuration"):
        config_by_name("pca_residual_combined")


def test_an_unknown_subspace_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown subspace"):
        PcaResidualConfig("t", "both")


def test_fit_all_returns_every_reported_configuration():
    x, labels = clustered(n_features=25, seed=31)
    scorers = fit_all_pca_residual(x, labels)
    assert set(scorers) == {c.name for c in CONFIGURATIONS}
    for name, scorer in scorers.items():
        assert scorer.name == name


def test_diagnostics_carry_the_runtime_d_and_reach_the_run_manifest():
    """d is recorded where it belongs: in the manifest, read at runtime.

    ``manifest.run_manifest`` calls ``diagnostics()`` on every scorer, so this
    family joins the manifest without manifest.py being edited -- which is what
    the design asked for and what this asserts.
    """
    x, labels = clustered(n_features=25, seed=32)
    scorers = fit_all_pca_residual(x, labels)
    fragment = run_manifest(
        scorers,
        seed=0,
        repo_commit="deadbeef",
        openood_commit="8d44375e",
        # Declared by PcaResidualScorer.MANIFEST_REQUIRES; without it this call
        # raises. See test_a_scorer_declaring_a_required_key_is_refused_without_it.
        extra={"degeneracy": degeneracy_report(scorers["pca_residual_class_mean"], x)},
    )
    for name, scorer in scorers.items():
        entry = fragment["scorers"][name]
        assert entry["d"] == scorer.d
        assert entry["subspace"] == scorer.config.subspace
        assert entry["n_residual_directions"] == scorer.n_features - scorer.d
        assert 0.0 <= entry["explained_variance_at_d"] <= 1.0 + 1e-12
        assert decode_hyperparams(entry["hyperparams"])["d"] == scorer.d


def test_the_all_id_diagnostics_record_the_sweep_and_the_never_reselect_rule():
    x, labels = clustered(n_features=25, seed=33)
    all_id = fit_pca_residual("pca_residual_all_id", x, labels).diagnostics()
    class_mean = fit_pca_residual("pca_residual_class_mean", x, labels).diagnostics()
    assert all_id["sweep_d"] == list(SWEEP_D)
    assert all_id["sweep_is_reported_never_reselected"] is True
    assert all_id["d_rule"] == "explained_variance_0.95"
    # Nothing to sweep on the class-mean side; the grid must not appear there.
    assert "sweep_d" not in class_mean
    assert class_mean["d_rule"] == "k_minus_1"


def test_the_whitened_variant_records_its_whitening_covariance():
    """A condition number is required for every fitted covariance.

    The whitening fit is one, so its diagnostics travel with the scorer rather
    than being dropped because it is not a Gaussian-family covariance.
    """
    x, labels = clustered(n_features=20, seed=34)
    entry = fit_pca_residual("pca_residual_class_mean_whitened", x, labels).diagnostics()
    assert "condition_number" in entry["whitening_covariance"]
    assert entry["whitening_covariance"]["shrinkage"] is False


def test_the_scorer_does_not_expose_an_attribute_named_primary():
    """manifest.covariance_manifest filters on `primary` and then reads `background`.

    A PCA-residual scorer exposing a `primary` would pass that filter and die on
    the next line -- a failure mode manifest.py's own comment anticipates. The
    whitening fit is `whitening_fit`, and the covariance fragment simply skips
    this family.
    """
    x, labels = clustered(n_features=20, seed=35)
    for scorer in fit_all_pca_residual(x, labels).values():
        assert not hasattr(scorer, "primary")
        assert not hasattr(scorer, "background")


def test_hyperparams_record_the_fitting_split_and_the_decomposition():
    """Fit on train not val, centred on mu_ID, eigh not eig -- all three recorded.

    The switch away from OpenOOD's val-fitting is documented rather than silent,
    and the manifest is where "documented" has to mean something checkable.
    """
    x, labels = clustered(n_features=20, seed=36)
    params = decode_hyperparams(
        fit_pca_residual("pca_residual_all_id", x, labels).diagnostics()["hyperparams"]
    )
    assert params["fitting_split"] == "cifar10_train"
    assert params["centre"] == "mu_id"
    assert params["eigendecomposition"] == "eigh"


# --------------------------------------------------------------------------- #
# Baseline 2 and the shared dump
# --------------------------------------------------------------------------- #


def test_the_scorer_drops_into_the_shared_score_frame():
    """Satisfies the Scorer protocol, so the one dump function handles it.

    ``score_frame`` lives in its own module precisely so this family would not
    write a second one. The dump ends in ``assert_canonical_order``,
    which is what makes the shared bootstrap index set safe.
    """
    x, labels = clustered(n_features=20, per_class=15, seed=37)
    scorers = fit_all_pca_residual(x, labels)
    frame = score_frame(index_frame(len(x)), x, scorers)

    assert list(frame.columns) == list(CANONICAL_SORT_KEY) + list(scorers)
    assert len(frame) == len(x)
    assert frame["image_id"].is_monotonic_increasing
    for name in scorers:
        assert frame[name].notna().all()


def test_a_mixed_dump_with_the_comparative_baseline_is_aligned_for_free():
    """The headline claim is the paired difference against marginal_diagonal.

    That comparison is only meaningful if row i means the same image in both
    arrays. Dumping the naive cell and the PCA-residual scorers in **one** call
    makes the alignment structural rather than something to verify afterwards --
    which is the reason there is one dump function rather than one per family.
    """
    x, labels = clustered(n_features=20, per_class=15, seed=38)
    baseline = fit_scorer("marginal_diagonal", x, labels)
    scorers = [baseline, *fit_all_pca_residual(x, labels).values()]
    frame = score_frame(index_frame(len(x)), x, scorers)

    assert "marginal_diagonal" in frame.columns
    assert "pca_residual_all_id" in frame.columns
    # A paired difference is now a column subtraction, with no reindexing.
    difference = frame["pca_residual_all_id"] - frame["marginal_diagonal"]
    assert len(difference) == len(x)
    assert difference.notna().all()


def test_the_dump_permutes_scores_with_the_sort_rather_than_after_it():
    """The silent failure assert_canonical_order exists to prevent.

    ``index_frame`` hands in reversed image_ids, so the dump must sort. If the
    embeddings were not permuted alongside, every score would belong to a
    different row -- producing wrong paired differences without ever raising.
    Checked by scoring the sorted embeddings independently.
    """
    x, labels = clustered(n_features=20, per_class=15, seed=39)
    scorer = fit_pca_residual("pca_residual_class_mean", x, labels)
    index = index_frame(len(x))
    frame = score_frame(index, x, [scorer])

    order = np.argsort(index["image_id"].to_numpy(), kind="stable")
    assert frame["pca_residual_class_mean"].to_numpy() == pytest.approx(
        scorer.score(x[order])
    )
