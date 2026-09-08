# tests/test_metrics.py
#
# Closed-form coverage for the two metrics the whole thesis is reported in.
#
# Every expectation below is arithmetic written out in the test: a fixture whose
# answer can be computed by hand, not a regression value captured from a previous
# run. Three cross-checks against scikit-learn sit behind `pytest.importorskip`
# and un-skip wherever it is installed.

import numpy as np
import pytest

from xai_ood.metrics import (
    METRICS,
    TPR_TARGET,
    as_scores,
    auroc,
    auroc_against_sorted_id,
    evaluate,
    fpr_at_tpr,
    metric_definitions,
    threshold_at_tpr,
)


# --------------------------------------------------------------------------- #
# AUROC, by hand
# --------------------------------------------------------------------------- #


def test_perfect_separation_is_one():
    """Every OOD score above every ID score: 6 of 6 pairs won."""
    assert auroc([0.0, 1.0, 2.0], [3.0, 4.0]) == 1.0


def test_perfect_inversion_is_zero():
    """The same fixture with the sign convention violated. This is the shape of
    the failure the package's one-negation rule exists to prevent: an AUROC of
    roughly 1 - true, which reads as a plausible number rather than a crash."""
    assert auroc([3.0, 4.0], [0.0, 1.0, 2.0]) == 0.0


def test_a_single_tie_gets_half_credit():
    """id = [0, 1, 2], ood = [1].

    Pairs: (1 > 0) won, (1 = 1) half, (1 < 2) lost.
    wins = 1 + 0.5 = 1.5, over 3 x 1 pairs -> 0.5 exactly.
    """
    assert auroc([0.0, 1.0, 2.0], [1.0]) == 0.5


def test_hand_computed_mixed_case():
    """id = [1, 2, 3, 4], ood = [2, 5].

    ood = 2: beats 1, ties 2, loses to 3 and 4 -> 1 + 0.5 = 1.5
    ood = 5: beats all four                    -> 4.0
    total 5.5 over 4 x 2 = 8 pairs -> 0.6875 exactly.
    """
    assert auroc([1.0, 2.0, 3.0, 4.0], [2.0, 5.0]) == 0.6875


def test_all_scores_identical_is_one_half():
    """Every pair a tie. A constant scorer is exactly uninformative."""
    assert auroc(np.ones(7), np.ones(4)) == 0.5


def test_auroc_matches_a_brute_force_pairwise_count():
    """The searchsorted formulation against the definition it implements.

    The fast path exists because the bootstrap calls it 52,000 times; this pins
    that the optimisation did not change the quantity. Deliberately uses a
    coarse grid so ties are common rather than hypothetical.
    """
    rng = np.random.default_rng(20260826)
    id_scores = rng.integers(0, 8, size=60).astype(float)
    ood_scores = rng.integers(0, 8, size=45).astype(float)
    wins = 0.0
    for o in ood_scores:
        for i in id_scores:
            wins += 1.0 if o > i else (0.5 if o == i else 0.0)
    expected = wins / (id_scores.size * ood_scores.size)
    assert auroc(id_scores, ood_scores) == pytest.approx(expected, rel=1e-15)


def test_auroc_is_invariant_to_any_monotone_rescaling():
    """AUROC is a rank statistic, which is why kNN can return
    distances unsquared and the Gaussian family can return Mahalanobis distances
    squared without the two being on different footings."""
    rng = np.random.default_rng(7)
    id_scores = rng.normal(size=200)
    ood_scores = rng.normal(1.0, 1.0, size=150)
    plain = auroc(id_scores, ood_scores)
    for transform in (np.exp, lambda x: 3.0 * x + 11.0, lambda x: x ** 3):
        assert auroc(transform(id_scores), transform(ood_scores)) == pytest.approx(
            plain, rel=1e-12
        )


def test_swapping_the_arguments_gives_one_minus_the_auroc():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=80), rng.normal(0.7, size=90)
    assert auroc(a, b) + auroc(b, a) == pytest.approx(1.0, rel=1e-15)


def test_auroc_against_sorted_id_needs_a_sorted_array_and_the_caller_supplies_it():
    """The fast path takes the sort as a precondition; this documents it.

    Handing it an unsorted array does not raise -- searchsorted has no way to
    know -- so the contract is enforced by there being exactly two call sites,
    both of which sort. Pinned here so a third call site has something to fail.
    """
    id_scores = np.array([5.0, 1.0, 3.0])
    unsorted = auroc_against_sorted_id(id_scores, np.array([2.0]))
    correct = auroc_against_sorted_id(np.sort(id_scores), np.array([2.0]))
    assert correct == pytest.approx(1.0 / 3.0)
    assert unsorted != correct


# --------------------------------------------------------------------------- #
# The 95%-TPR threshold, by hand
# --------------------------------------------------------------------------- #


def test_threshold_is_the_ceil_rank_order_statistic():
    """n = 100, target 0.95 -> ceil(95) = 95th smallest = the value 95."""
    id_sorted = np.arange(1.0, 101.0)
    assert threshold_at_tpr(id_sorted, 0.95) == 95.0


def test_threshold_rounds_up_so_the_target_is_cleared_not_missed():
    """n = 20, target 0.95 -> ceil(19.0) = 19 -> exactly 0.95 achieved.
    n = 10, target 0.95 -> ceil(9.5) = 10 -> 1.0 achieved, which is >= 0.95.

    Rounding down would accept 9 of 10 and report an FPR at a 0.90 operating
    point under a column header that says 95.
    """
    assert threshold_at_tpr(np.arange(1.0, 21.0), 0.95) == 19.0
    assert threshold_at_tpr(np.arange(1.0, 11.0), 0.95) == 10.0


def test_threshold_is_an_observed_score_never_an_interpolated_one():
    """A deliberately gappy ID set. Interpolating between order statistics
    would return 47.5, a value no sample ever took."""
    id_sorted = np.array([1.0, 2.0, 3.0, 45.0, 50.0])
    tau = threshold_at_tpr(id_sorted, 0.95)
    assert tau in set(id_sorted.tolist())
    assert tau == 50.0  # ceil(0.95 * 5) = 5 -> the 5th smallest


def test_the_achieved_tpr_is_reported_because_it_is_not_always_the_target():
    """37 ID samples: ceil(0.95 * 37) = 36, so 36/37 = 0.973 is accepted, not
    0.950. An FPR quoted against an unstated 0.973 is a different number."""
    report = evaluate(np.arange(1.0, 38.0), np.array([100.0]))
    assert report["achieved_tpr"] == pytest.approx(36.0 / 37.0)
    assert report["threshold"] == 36.0


def test_fpr_at_tpr_by_hand():
    """id = 1..100 -> tau = 95. ood = [90, 96, 200]: one at or below tau."""
    assert fpr_at_tpr(np.arange(1.0, 101.0), [90.0, 96.0, 200.0]) == pytest.approx(
        1.0 / 3.0
    )


def test_fpr_is_zero_when_every_ood_score_clears_the_threshold():
    assert fpr_at_tpr(np.arange(1.0, 101.0), [1000.0, 2000.0]) == 0.0


def test_fpr_is_one_when_no_ood_score_clears_it():
    assert fpr_at_tpr(np.arange(1.0, 101.0), [-5.0, 0.0]) == 1.0


def test_ties_at_the_threshold_count_as_accepted():
    """An OOD score exactly equal to tau is classified ID, matching the
    "at or below" rule the threshold is defined by. Stated as a test because
    the strict/non-strict choice is a real third-decimal disagreement between
    implementations."""
    assert fpr_at_tpr(np.arange(1.0, 101.0), [95.0]) == 1.0


# --------------------------------------------------------------------------- #
# Input hygiene
# --------------------------------------------------------------------------- #


def test_nan_scores_raise_rather_than_propagate():
    """np.sort puts NaN last and searchsorted against a NaN-terminated array
    returns counts that are wrong without raising, so a single NaN would move
    an AUROC silently."""
    with pytest.raises(ValueError, match="non-finite"):
        auroc([1.0, np.nan, 3.0], [2.0])
    with pytest.raises(ValueError, match="non-finite"):
        auroc([1.0, 2.0], [np.inf])


def test_empty_sides_raise():
    with pytest.raises(ValueError, match="empty"):
        auroc([], [1.0])
    with pytest.raises(ValueError, match="empty"):
        auroc([1.0], [])


def test_as_scores_flattens_and_casts():
    out = as_scores([[1, 2], [3, 4]], name="x")
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_an_out_of_range_tpr_target_raises():
    with pytest.raises(ValueError, match="tpr must be"):
        threshold_at_tpr(np.arange(10.0), 1.5)
    with pytest.raises(ValueError, match="tpr must be"):
        threshold_at_tpr(np.arange(10.0), 0.0)


# --------------------------------------------------------------------------- #
# Scale and self-description
# --------------------------------------------------------------------------- #


def test_metrics_return_fractions_not_percentages():
    """The 0-100 conversion happens once, in xai_ood.bootstrap.tables._display, where
    a metric enters the results table. Nothing here scales anything."""
    rng = np.random.default_rng(0)
    value = auroc(rng.normal(size=50), rng.normal(2.0, size=50))
    assert 0.0 <= value <= 1.0
    assert value > 0.5  # separable fixture, so this is not vacuously in range


def test_the_metric_set_is_the_two_the_thesis_reports():
    assert METRICS == ("auroc", "fpr95")
    assert TPR_TARGET == 0.95


def test_metric_definitions_record_the_two_conventions_that_could_differ():
    definitions = metric_definitions()
    assert "0.5 P(tie)" in definitions["auroc"]
    assert definitions["fpr_at_tpr_target"] == 0.95
    assert definitions["fpr_threshold_recomputed_per_replicate"] is True
    assert "no interpolation" in definitions["fpr_threshold_rule"]


def test_evaluate_returns_both_metrics_and_the_counts_behind_them():
    report = evaluate([1.0, 2.0, 3.0, 4.0], [2.0, 5.0])
    assert report["auroc"] == 0.6875
    assert report["n_id"] == 4 and report["n_ood"] == 2


# --------------------------------------------------------------------------- #
# Cross-checks against scikit-learn. Skipped when it is absent, un-skipped when
# it is installed.
# --------------------------------------------------------------------------- #


def test_auroc_matches_sklearn_roc_auc_score():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    id_scores = rng.normal(size=500)
    ood_scores = rng.normal(0.8, 1.3, size=400)
    labels = np.concatenate([np.zeros(500), np.ones(400)])
    scores = np.concatenate([id_scores, ood_scores])
    assert auroc(id_scores, ood_scores) == pytest.approx(
        sk.roc_auc_score(labels, scores), rel=1e-12
    )


def test_auroc_matches_sklearn_with_heavy_ties():
    """The tie convention is where two defensible implementations diverge, so
    the cross-check is run on a fixture built to produce ties rather than on
    continuous scores where it would pass either way."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(12)
    id_scores = rng.integers(0, 5, size=300).astype(float)
    ood_scores = rng.integers(0, 5, size=200).astype(float)
    labels = np.concatenate([np.zeros(300), np.ones(200)])
    scores = np.concatenate([id_scores, ood_scores])
    assert auroc(id_scores, ood_scores) == pytest.approx(
        sk.roc_auc_score(labels, scores), rel=1e-12
    )


def test_auroc_matches_sklearn_at_phase_2_shape():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(13)
    id_scores = rng.normal(size=9_000)
    ood_scores = rng.normal(0.5, size=9_000)
    labels = np.concatenate([np.zeros(9_000), np.ones(9_000)])
    scores = np.concatenate([id_scores, ood_scores])
    assert auroc(id_scores, ood_scores) == pytest.approx(
        sk.roc_auc_score(labels, scores), rel=1e-12
    )
