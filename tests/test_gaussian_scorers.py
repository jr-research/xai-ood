# tests/test_gaussian_scorers.py
#
# Closed-form synthetic-Gaussian assertions for all six configurations: known
# mean, known diagonal covariance, hand-computed expected distance, plus the sign
# convention, the RMD decomposition identity, and canonical ordering of the score
# dump.
#
# Every expectation below is arithmetic done by hand in the docstring, not a
# value copied out of a previous run. Fixtures are exact factorial designs so
# the fitted moments are the stated ones rather than estimates of them.
#
# numpy / pandas / pytest only. No GPU, no embedding cache, no real embedding.
import numpy as np
import pandas as pd
import pytest

from xai_ood.methods import ood_score_from_openood_confidence
from xai_ood.methods.covariance import l2_normalize
from xai_ood.methods.gaussian import (
    CONFIGURATIONS,
    config_by_name,
    fit_all_scorers,
    fit_scorer,
    score_frame,
)
from xai_ood.schema import assert_canonical_order, decode_hyperparams
from xai_ood.visualization import style


# --------------------------------------------------------------------------- #
# Fixtures with exactly-known moments
# --------------------------------------------------------------------------- #

#: Mean exactly (0, 0); covariance exactly [[2.5, 0.5], [0.5, 1.0]].
#: Built as {p, -p, q, -q} with p = (2, 1), q = (1, -1):
#:   S = (2 p p^T + 2 q q^T) / 4 = ([[4,2],[2,1]] + [[1,-1],[-1,1]]) / 2
CORRELATED_DESIGN = np.array([[2.0, 1.0], [-2.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
CORRELATED_COV = np.array([[2.5, 0.5], [0.5, 1.0]])

SEP, A, B = 4.0, 0.5, 0.25


def two_cluster_design():
    """Two exact designs at (-4, 0) and (+4, 0), spread (a, b) = (0.5, 0.25).

    Every number here is exactly representable in binary floating point, so the
    fitted moments are exact rather than approximate:

      class means           (-4, 0) and (+4, 0)
      pooled within-class   diag(0.25, 0.0625)
      global mean           (0, 0)
      marginal covariance   diag(0.25 + 16, 0.0625) = diag(16.25, 0.0625)

    The last line is the law of total covariance: the marginal covariance
    absorbs the between-class term (sep^2 = 16 in dimension 0), the pooled one
    excludes it by construction.
    """
    base = np.array([[A, B], [A, -B], [-A, B], [-A, -B]])
    x = np.vstack([base + [-SEP, 0.0], base + [SEP, 0.0]])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return x, labels


WITHIN = np.diag([A**2, B**2])  # diag(0.25, 0.0625)
MARGINAL = np.diag([A**2 + SEP**2, B**2])  # diag(16.25, 0.0625)


def noisy_two_clusters(seed=101, n=400, sep=5.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[-sep, 0.0], scale=[0.6, 0.5], size=(n, 2))
    b = rng.normal(loc=[sep, 0.0], scale=[0.6, 0.5], size=(n, 2))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


# --------------------------------------------------------------------------- #
# Closed form, cell by cell
# --------------------------------------------------------------------------- #


def test_marginal_full_hand_computed_distance():
    """With S = [[2.5, 0.5], [0.5, 1.0]], det S = 2.25 and

        S^-1 = (1 / 2.25) * [[1.0, -0.5], [-0.5, 2.5]]

    so MD^2(z) = (z0^2 - z0 z1 + 2.5 z1^2) / 2.25. Hence

        (3, 0) -> 9 / 2.25            = 4
        (0, 3) -> 22.5 / 2.25         = 10
        (3, 3) -> (9 - 9 + 22.5)/2.25 = 10
        (0, 0) -> 0
    """
    scorer = fit_scorer("marginal_full", CORRELATED_DESIGN)
    assert scorer.primary.covariance == pytest.approx(CORRELATED_COV, rel=1e-15)

    z = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 3.0], [0.0, 0.0]])
    assert scorer.score(z) == pytest.approx([4.0, 10.0, 10.0, 0.0], rel=1e-12)


def test_marginal_diagonal_hand_computed_distance():
    """The diagonal restriction of the same fit is diag(2.5, 1.0), so

        MD^2(z) = z0^2 / 2.5 + z1^2

        (3, 0) -> 3.6
        (0, 3) -> 9.0
        (3, 3) -> 12.6

    Different from the full cell above on the same data, which is what makes
    the "diagonal is a restriction, not a separate estimator" test meaningful
    rather than tautological.
    """
    scorer = fit_scorer("marginal_diagonal", CORRELATED_DESIGN)
    assert scorer.primary.covariance == pytest.approx(np.diag([2.5, 1.0]), rel=1e-15)

    z = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
    assert scorer.score(z) == pytest.approx([3.6, 9.0, 12.6], rel=1e-12)


def test_class_conditional_cells_hand_computed_distance():
    """Pooled within-class is diag(0.25, 0.0625), class means (+-4, 0). So

        MD^2_cc(z) = min_k [ (z0 -+ 4)^2 / 0.25 + z1^2 / 0.0625 ]

        (0, 0)      -> 16 / 0.25                    = 64
        (-4, 0.25)  -> 0 + 0.0625 / 0.0625          = 1
        (-4.5, 0.25)-> 0.25/0.25 + 0.0625/0.0625    = 2
        (4.5, -0.25)-> 2, by symmetry

    The pooled covariance is already diagonal on this fixture, so the diagonal
    and full class-conditional cells coincide here; that equality is asserted
    too, since it is the check that the restriction is applied to the fitted
    matrix rather than changing the estimator.
    """
    x, labels = two_cluster_design()
    full = fit_scorer("class_conditional_full", x, labels)
    diag = fit_scorer("class_conditional_diagonal", x, labels)

    assert full.primary.covariance == pytest.approx(WITHIN, rel=1e-12)
    assert full.primary.class_means.ravel() == pytest.approx(
        np.array([[-SEP, 0.0], [SEP, 0.0]]).ravel(), abs=1e-12
    )

    z = np.array([[0.0, 0.0], [-4.0, 0.25], [-4.5, 0.25], [4.5, -0.25]])
    expected = [64.0, 1.0, 2.0, 2.0]
    assert full.score(z) == pytest.approx(expected, rel=1e-12)
    assert diag.score(z) == pytest.approx(expected, rel=1e-12)


def test_marginal_full_on_the_two_cluster_fixture_hand_computed():
    """Marginal covariance is diag(16.25, 0.0625) and the global mean is (0, 0):

        MD^2(z) = z0^2 / 16.25 + z1^2 / 0.0625

        (0, 0)       -> 0
        (-4.5, 0.25) -> 20.25/16.25 + 1 = 2.2461538...
        (-3.5, 0.25) -> 12.25/16.25 + 1 = 1.7538461...
    """
    x, labels = two_cluster_design()
    scorer = fit_scorer("marginal_full", x)
    assert scorer.primary.covariance == pytest.approx(MARGINAL, rel=1e-12)

    z = np.array([[0.0, 0.0], [-4.5, 0.25], [-3.5, 0.25]])
    expected = [0.0, 20.25 / 16.25 + 1.0, 12.25 / 16.25 + 1.0]
    assert scorer.score(z) == pytest.approx(expected, rel=1e-12)


def test_rmd_hand_computed_distance():
    """RMD(z) = MD^2_cc(z) - MD^2_marginal(z), from the two cells above:

        (0, 0)       ->  64      - 0            =  64
        (-4.5, 0.25) ->   2      - 2.2461538... =  -0.2461538...
        (-3.5, 0.25) ->   2      - 1.7538461... =   0.2461538...

    Negative values are expected and must not be clipped: RMD is the *excess*
    of class-specific deviation over generic unusualness, not a distance.
    """
    x, labels = two_cluster_design()
    rmd = fit_scorer("rmd", x, labels)

    z = np.array([[0.0, 0.0], [-4.5, 0.25], [-3.5, 0.25]])
    expected = [64.0, 2.0 - (20.25 / 16.25 + 1.0), 2.0 - (12.25 / 16.25 + 1.0)]
    got = rmd.score(z)
    assert got == pytest.approx(expected, rel=1e-10)
    assert got[1] < 0.0


def test_rmd_is_exactly_the_difference_of_its_two_reported_cells():
    """The decomposition claim the whole 2x2 rests on, asserted as an identity.

    The two cells of the Full column are literally RMD's two terms, so reporting
    them separately is a decomposition of a scorer already in the design rather
    than an
    added experiment. If this ever fails, the factorial no longer explains RMD.
    """
    x, labels = noisy_two_clusters()
    rng = np.random.default_rng(202)
    z = rng.normal(scale=6.0, size=(50, 2))

    scorers = fit_all_scorers(x, labels)
    difference = scorers["class_conditional_full"].score(z) - scorers["marginal_full"].score(z)
    assert scorers["rmd"].score(z) == pytest.approx(difference, rel=1e-12)


def test_rmd_pp_is_rmd_on_l2_normalized_embeddings():
    """RMD++ is a one-line preprocessing variant, applied to fit and query alike."""
    x, labels = noisy_two_clusters(seed=303)
    rng = np.random.default_rng(404)
    z = rng.normal(scale=6.0, size=(40, 2))

    rmd_pp = fit_scorer("rmd_pp", x, labels)
    rmd_on_normalized = fit_scorer("rmd", l2_normalize(x), labels)

    assert rmd_pp.score(z) == pytest.approx(rmd_on_normalized.score(l2_normalize(z)), rel=1e-10)
    # And it is genuinely a different scorer from plain RMD on this data.
    assert not np.allclose(rmd_pp.score(z), fit_scorer("rmd", x, labels).score(z))


# --------------------------------------------------------------------------- #
# Sign convention: higher = more OOD, all six, no exceptions
# --------------------------------------------------------------------------- #


def test_every_configuration_scores_a_far_away_point_as_most_ood():
    """The sign convention, applied to the real scorers rather than test-local ones.

    A mixed sign convention yields an AUROC of roughly one minus its true value,
    which reads as a plausible number rather than a crash. This is the "no
    exceptions" rule turned into code.
    """
    x, labels = noisy_two_clusters()
    far_ood = np.array([[0.0, 25.0]])

    for scorer in fit_all_scorers(x, labels).values():
        id_scores = scorer.score(x)
        ood_score = scorer.score(far_ood)
        assert ood_score[0] > id_scores.max(), scorer.name


def test_rmd_flags_the_void_between_clusters_where_the_marginal_cell_fails():
    """The mechanism the 2x2 exists to isolate, as a pre-specified assertion.

    A point sitting in the void between two ID clusters is close to the global
    mean along the highest-variance direction, so the marginal model scores it
    as maximally in-distribution. The class-conditional model does not. This is
    the standard failure of a unimodal Gaussian fitted to multimodal data, and
    near-OOD samples are the population most likely to land there.

    On the exact fixture: the void point (0, 0) *is* the global mean, so
    marginal MD^2 = 0, the smallest value attainable, while class-conditional
    MD^2 = 64 against a maximum of 2 over the ID points.
    """
    x, labels = two_cluster_design()
    void = np.array([[0.0, 0.0]])
    scorers = fit_all_scorers(x, labels)

    for name in ("class_conditional_full", "class_conditional_diagonal", "rmd"):
        assert scorers[name].score(void)[0] > scorers[name].score(x).max(), name

    for name in ("marginal_full", "marginal_diagonal"):
        assert scorers[name].score(void)[0] < scorers[name].score(x).min(), name


def test_marginal_diagonal_ranks_identically_to_the_diagonal_gaussian_nll():
    """Closes the loop the sign-convention test file's header asks for.

    ``test_sign_convention_and_naive_baseline.py`` proves the equivalence using
    test-local implementations. This asserts the same property of the *shipped*
    scorer: the diagonal Gaussian negative log-likelihood is
    ``0.5 * MD^2_diag + 0.5 * sum(log var)``, a strictly increasing affine
    function of the score, so the two orderings and therefore the two AUROCs
    coincide. It is a real check, not a tautology: it fails if the
    log-determinant term is ever made sample-dependent.
    """
    x, labels = noisy_two_clusters(seed=505)
    rng = np.random.default_rng(606)
    z = np.vstack([x, rng.normal(scale=9.0, size=(60, 2))])

    scorer = fit_scorer("marginal_diagonal", x)
    var = np.diag(scorer.primary.covariance)
    nll = 0.5 * np.sum((z - scorer.primary.mean) ** 2 / var, axis=1) + 0.5 * np.sum(np.log(var))

    assert np.array_equal(np.argsort(scorer.score(z)), np.argsort(nll))


def test_the_openood_seam_adapter_is_the_only_negation():
    """The adapter exists as importable code so a second negation is visible."""
    confidence = np.array([2.0, -1.0, 0.0])
    assert ood_score_from_openood_confidence(confidence) == pytest.approx([-2.0, 1.0, 0.0])


# --------------------------------------------------------------------------- #
# Configuration bookkeeping
# --------------------------------------------------------------------------- #


def test_there_are_exactly_six_configurations_with_the_expected_names():
    assert [c.name for c in CONFIGURATIONS] == [
        "marginal_diagonal",
        "marginal_full",
        "class_conditional_diagonal",
        "class_conditional_full",
        "rmd",
        "rmd_pp",
    ]


def test_every_configuration_has_a_stable_colour_in_the_thesis_palette():
    """A scorer must keep one colour across every figure in the thesis.

    ``series_color`` has a deterministic fallback for unlisted names, so an
    unregistered scorer would not crash; it would just get an arbitrary colour
    that no one chose. Checking membership rather than the return value is what
    makes that visible.
    """
    for config in CONFIGURATIONS:
        assert config.name in style._SERIES_COLOR, config.name


def test_hyperparams_round_trip_through_the_schema_encoder():
    config = config_by_name("rmd_pp")
    decoded = decode_hyperparams(config.encoded_hyperparams(shrinkage=True))
    assert decoded == {
        "class_conditional": True,
        "covariance": "full",
        "relative": True,
        "l2_normalize": True,
        "shrinkage": True,
        "shrinkage_target": "ledoit_wolf_scaled_identity",
        "pooling": "pooled_within_class",
    }


def test_shrinkage_off_records_a_null_target_rather_than_naming_one():
    decoded = decode_hyperparams(config_by_name("rmd").encoded_hyperparams(shrinkage=False))
    assert decoded["shrinkage"] is False
    assert decoded["shrinkage_target"] is None


def test_class_conditional_configuration_refuses_to_fit_without_labels():
    """Silently falling back to the marginal model would collapse two cells."""
    x, _ = two_cluster_design()
    with pytest.raises(ValueError, match="labels are required"):
        fit_scorer("class_conditional_full", x, None)


def test_rmd_background_fit_is_marginal_and_full_never_diagonal():
    x, labels = two_cluster_design()
    rmd = fit_scorer("rmd", x, labels)
    assert rmd.background is not None
    assert rmd.background.class_means is None
    assert rmd.background.diagonal is False


def test_unknown_configuration_name_lists_the_six():
    with pytest.raises(KeyError, match="rmd_pp"):
        config_by_name("rmd_plus_plus")


def test_shrinkage_flag_reaches_every_covariance_behind_a_relative_scorer():
    x, labels = noisy_two_clusters(seed=707)
    scorer = fit_scorer("rmd", x, labels, shrinkage=True)
    assert scorer.primary.shrinkage is True
    assert scorer.background.shrinkage is True
    assert scorer.primary.shrinkage_intensity > 0.0


# --------------------------------------------------------------------------- #
# The score dump, and canonical order
# --------------------------------------------------------------------------- #


def sample_index():
    """Four rows spanning both split roles, deliberately out of canonical order.

    ``corruption`` and ``severity`` are null-valued, not absent, on the non-cs-ID
    rows, so the sort key stays total.
    """
    return pd.DataFrame(
        {
            "split": ["id_test", "id_test", "csid", "csid"],
            "image_id": ["img_0002", "img_0001", "img_0001", "img_0001"],
            "corruption": [pd.NA, pd.NA, "fog", "fog"],
            "severity": [pd.NA, pd.NA, 3, 1],
        }
    )


def test_score_frame_returns_canonical_order():
    x, labels = noisy_two_clusters(seed=808)
    scorers = fit_all_scorers(x, labels)
    embeddings = np.array([[1.0, 2.0], [3.0, 0.5], [-2.0, 1.0], [0.0, -4.0]])

    out = score_frame(sample_index(), embeddings, scorers)

    assert_canonical_order(out)  # raises if not; belt and braces on the return
    assert list(out["split"]) == ["csid", "csid", "id_test", "id_test"]
    assert list(out["image_id"]) == ["img_0001", "img_0001", "img_0001", "img_0002"]
    # cs-ID rows sort by severity numerically; the clean rows keep their nulls,
    # which the comparator maps to a sentinel for sorting only, never on disk.
    severities = out["severity"].astype("Float64")
    assert list(severities[:2]) == [1, 3]
    assert bool(severities[2:].isna().all())
    assert bool(out["corruption"][2:].isna().all())
    for name in scorers:
        assert name in out.columns


def test_scores_travel_with_their_rows_through_the_sort():
    """The failure ``assert_canonical_order`` exists to prevent, tested directly.

    A misaligned score array produces wrong paired differences without ever
    raising, so the check has to be that a *particular* embedding's score lands
    on that embedding's row, not merely that the frame is sorted.
    """
    x, labels = noisy_two_clusters(seed=909)
    scorer = fit_scorer("marginal_full", x)
    index = sample_index()
    embeddings = np.array([[1.0, 2.0], [3.0, 0.5], [-2.0, 1.0], [0.0, -4.0]])

    out = score_frame(index, embeddings, [scorer])

    for i in range(len(index)):
        row = index.iloc[i]
        match = out[
            (out["split"] == row["split"])
            & (out["image_id"] == row["image_id"])
            & (out["corruption"].isna() if pd.isna(row["corruption"]) else out["corruption"] == row["corruption"])
            & (out["severity"].isna() if pd.isna(row["severity"]) else out["severity"] == row["severity"])
        ]
        assert len(match) == 1
        expected = scorer.score(embeddings[i : i + 1])[0]
        assert float(match["marginal_full"].iloc[0]) == pytest.approx(expected, rel=1e-12)


def test_shuffling_the_input_does_not_change_the_dump():
    """Canonical order makes the dump independent of arrival order.

    This is the precondition for the bootstrap applying one index set to
    every scorer's array: two scorers whose inputs arrived in different orders
    must still produce row-identical frames.
    """
    x, labels = noisy_two_clusters(seed=111)
    scorers = fit_all_scorers(x, labels)
    index = sample_index()
    embeddings = np.array([[1.0, 2.0], [3.0, 0.5], [-2.0, 1.0], [0.0, -4.0]])

    straight = score_frame(index, embeddings, scorers)

    order = np.array([2, 0, 3, 1])
    shuffled = score_frame(
        index.iloc[order].reset_index(drop=True), embeddings[order], scorers
    )

    pd.testing.assert_frame_equal(straight, shuffled)


def test_score_frame_rejects_a_row_count_mismatch():
    x, labels = noisy_two_clusters(seed=222)
    scorer = fit_scorer("marginal_full", x)
    with pytest.raises(ValueError, match="row for row"):
        score_frame(sample_index(), np.zeros((3, 2)), [scorer])


def test_score_frame_rejects_duplicate_sort_keys():
    """A duplicated key makes the order non-total, so a shared resample index
    would refer to different rows in different scorers' arrays."""
    x, labels = noisy_two_clusters(seed=333)
    scorer = fit_scorer("marginal_full", x)
    index = pd.DataFrame(
        {
            "split": ["id_test", "id_test"],
            "image_id": ["img_0001", "img_0001"],
            "corruption": [pd.NA, pd.NA],
            "severity": [pd.NA, pd.NA],
        }
    )
    with pytest.raises(ValueError, match="share a sort key"):
        score_frame(index, np.zeros((2, 2)), [scorer])


def test_all_six_run_end_to_end_on_synthetic_data():
    """The Gaussian family's "done when", as one test.

    All six configurations fit and score against synthetic data, producing
    finite, canonically-ordered per-sample arrays. No real embedding is touched.
    """
    rng = np.random.default_rng(444)
    x, labels = noisy_two_clusters(seed=555, n=250)
    n_eval = 60
    z = np.vstack([rng.normal(loc=[-5, 0], scale=0.6, size=(n_eval // 2, 2)),
                   rng.normal(loc=[0, 12], scale=1.0, size=(n_eval // 2, 2))])

    index = pd.DataFrame(
        {
            "split": ["id_test"] * (n_eval // 2) + ["far_ood"] * (n_eval // 2),
            "image_id": [f"img_{i:05d}" for i in range(n_eval)],
            "corruption": [pd.NA] * n_eval,
            "severity": [pd.NA] * n_eval,
        }
    )

    for shrinkage in (False, True):
        scorers = fit_all_scorers(x, labels, shrinkage=shrinkage)
        assert len(scorers) == 6
        out = score_frame(index, z, scorers)
        assert len(out) == n_eval
        for name in scorers:
            assert np.all(np.isfinite(out[name].to_numpy()))
        assert_canonical_order(out)
