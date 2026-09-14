# tests/test_knn.py
#
# The kNN family, k-th nearest-neighbour distance (Sun et al., 2022). Covers the
# normalization quirk not to copy, and the closed-form unit tests for this
# family.
#
# The central fixture is a ruler: references at (1, 0), (2, 0), ... (10, 0) and
# the query at the origin, so the distance to the i-th nearest reference is
# exactly i. That makes every expectation below an integer rather than an
# approximation, and it separates k-th from k-averaged unambiguously --
# the k-th is k, the k-average is (k + 1) / 2.
#
# numpy / pandas / pytest only. No GPU, no embedding cache, no faiss.
import numpy as np
import pandas as pd
import pytest

from xai_ood.methods.covariance import NORM_EPS, l2_normalize
from xai_ood.methods.dump import score_frame
from xai_ood.methods.knn import (
    DEFAULT_K,
    SWEEP_K,
    CONFIGURATIONS,
    KnnConfig,
    KnnScorer,
    config_by_name,
    fit_all_knn,
    fit_knn,
)
from xai_ood.schema import assert_canonical_order
from xai_ood.visualization import style


def ruler(n=10):
    """References at (1, 0) ... (n, 0). Distance from the origin to the i-th is i."""
    return np.array([[float(i), 0.0] for i in range(1, n + 1)])


ORIGIN = np.zeros((1, 2))


def gaussian_reference(seed=17, n=400, p=5):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)) @ np.diag(np.linspace(2.0, 0.5, p))


# --------------------------------------------------------------------------- #
# Closed form: k-th, and specifically not k-averaged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_kth_nearest_distance_is_exactly_k_on_the_ruler(k):
    scorer = fit_knn("knn_unnormalized", ruler(), k=k)
    assert scorer.score(ORIGIN) == pytest.approx([float(k)], rel=1e-15)


@pytest.mark.parametrize("k", [2, 3, 5, 10])
def test_kth_is_not_k_averaged(k):
    """Sun et al. report the two as empirically similar; the thesis reports the
    k-th, on their own theoretical argument. A fixture where the two differ
    by construction is what stops "similar" from becoming "interchangeable" in
    the implementation.

    On the ruler the k-average is (1 + 2 + ... + k) / k = (k + 1) / 2.
    """
    scorer = fit_knn("knn_unnormalized", ruler(), k=k)
    assert scorer.score(ORIGIN) == pytest.approx([float(k)], rel=1e-15)
    assert scorer.score_k_averaged(ORIGIN) == pytest.approx([(k + 1) / 2.0], rel=1e-15)
    assert scorer.score(ORIGIN)[0] != scorer.score_k_averaged(ORIGIN)[0]


def test_k_averaged_is_not_one_of_the_reported_variants():
    """It exists to be measured on request, not to be reported."""
    assert {c.name for c in CONFIGURATIONS} == {"knn_normalized", "knn_unnormalized"}
    for config in CONFIGURATIONS:
        assert config.hyperparams()["aggregation"] == "kth"


def test_closed_form_distance_on_a_grid():
    """A second closed form that is not collinear, so the metric itself is checked.

    References at the four points (+-3, 0) and (0, +-4); query at the origin.
    Sorted distances are 3, 3, 4, 4, so r_1 = r_2 = 3 and r_3 = r_4 = 4.
    """
    ref = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 4.0], [0.0, -4.0]])
    scorer = fit_knn("knn_unnormalized", ref, k=1)
    got = scorer.score_sweep(ORIGIN, [1, 2, 3, 4])
    assert [float(got[k][0]) for k in (1, 2, 3, 4)] == pytest.approx([3.0, 3.0, 4.0, 4.0], rel=1e-15)


# --------------------------------------------------------------------------- #
# k as a default argument, not a magic number
# --------------------------------------------------------------------------- #


def test_default_k_is_fifty_and_lives_in_one_place():
    assert DEFAULT_K == 50
    for config in CONFIGURATIONS:
        assert config.k == DEFAULT_K


def test_k_is_identical_across_the_two_variants():
    """Normalization must stay a clean single factor.

    If the two variants used different k, any normalized-vs-unnormalized
    difference would confound two changes, which is the exact mistake the 2x2
    factorial exists to avoid making in the Gaussian family.
    """
    scorers = fit_all_knn(gaussian_reference())
    assert len({s.config.k for s in scorers.values()}) == 1


def test_the_sweep_calls_the_same_function_rather_than_a_second_implementation():
    """The appendix sweep varies k. It must not fork the scorer to do it."""
    ref = gaussian_reference()
    scorer = fit_knn("knn_normalized", ref)
    rng = np.random.default_rng(88)
    q = rng.normal(size=(30, ref.shape[1]))

    swept = scorer.score_sweep(q, SWEEP_K[:4])
    for k in SWEEP_K[:4]:
        assert swept[k] == pytest.approx(scorer.score(q, k=k), rel=1e-15)


def test_sweep_grid_is_the_pre_declared_one():
    assert SWEEP_K == (1, 10, 20, 50, 100, 200, 500, 1000)
    assert DEFAULT_K in SWEEP_K  # the headline value sits inside the reported grid


def test_k_beyond_the_reference_set_raises_rather_than_clipping():
    ref = ruler(10)
    with pytest.raises(ValueError, match="exceeds the reference set size"):
        fit_knn("knn_unnormalized", ref, k=11)
    scorer = fit_knn("knn_unnormalized", ref, k=3)
    with pytest.raises(ValueError, match="exceeds the reference set size"):
        scorer.score(ORIGIN, k=11)
    with pytest.raises(ValueError, match="at least 1"):
        scorer.score(ORIGIN, k=0)


# --------------------------------------------------------------------------- #
# Normalization: the flag, and the quirk not to copy
# --------------------------------------------------------------------------- #


def test_normalization_uses_the_one_shared_implementation():
    """The normalisation guard is on the denominator, `x / (norm + eps)`.

    This asserts the *scorer* holds exactly what the shared `l2_normalize`
    produces, rather than re-testing the guard here. That is the point: there is
    one L2 normalization in the codebase, so kNN and RMD++ cannot drift apart.
    A second implementation in this module would make this test pass while the
    two families quietly disagreed.
    """
    ref = gaussian_reference()
    scorer = fit_knn("knn_normalized", ref)
    assert scorer.reference == pytest.approx(l2_normalize(ref), rel=1e-15)

    openood_quirk = ref / np.linalg.norm(ref, axis=1, keepdims=True) + NORM_EPS
    assert not np.allclose(scorer.reference, openood_quirk, rtol=0.0, atol=1e-12)


def test_normalization_is_applied_to_the_query_as_well_as_the_reference():
    """Normalizing one side only would compare points on different scales."""
    ref = gaussian_reference()
    scorer = fit_knn("knn_normalized", ref)
    rng = np.random.default_rng(99)
    q = rng.normal(size=(20, ref.shape[1])) * 7.0  # deliberately off-scale

    manual = fit_knn("knn_unnormalized", l2_normalize(ref))
    assert scorer.score(q) == pytest.approx(manual.score(l2_normalize(q)), rel=1e-12)


def test_the_two_variants_are_genuinely_different_scorers():
    ref = gaussian_reference()
    scorers = fit_all_knn(ref)
    rng = np.random.default_rng(123)
    q = rng.normal(size=(25, ref.shape[1])) * np.linspace(0.5, 3.0, ref.shape[1])
    assert not np.allclose(
        scorers["knn_normalized"].score(q), scorers["knn_unnormalized"].score(q)
    )


def test_unnormalized_variant_leaves_the_reference_untouched():
    ref = gaussian_reference()
    scorer = fit_knn("knn_unnormalized", ref)
    assert scorer.reference == pytest.approx(ref, rel=1e-15)


# --------------------------------------------------------------------------- #
# Sign convention
# --------------------------------------------------------------------------- #


def directional_fixture(seed=2024, n=300):
    """A tight ID cluster along +x, and OOD displaced in *direction*, not norm.

    The direction matters: see
    ``test_the_normalized_variant_is_blind_to_a_pure_norm_outlier`` below.
    """
    rng = np.random.default_rng(seed)
    ref = rng.normal(loc=[5.0, 0.0, 0.0], scale=0.3, size=(n, 3))
    id_query = rng.normal(loc=[5.0, 0.0, 0.0], scale=0.3, size=(100, 3))
    ood_query = rng.normal(loc=[0.0, 5.0, 0.0], scale=0.3, size=(20, 3))
    return ref, id_query, ood_query


@pytest.mark.parametrize("name", ["knn_normalized", "knn_unnormalized"])
def test_sign_convention_higher_distance_is_more_ood(name):
    """Higher = more OOD, with no sign flip.

    OpenOOD's kNN postprocessor returns a negative k-th distance, being a
    confidence. This scorer returns the distance itself, so the adapter at the
    OpenOOD seam is the only place a sign is ever flipped.
    """
    ref, id_query, ood_query = directional_fixture()
    scorer = fit_knn(name, ref, k=5)
    assert scorer.score(ood_query).min() > scorer.score(id_query).max()


def test_the_normalized_variant_is_blind_to_a_pure_norm_outlier():
    """A real property of the normalization factor, not a defect, pinned here.

    L2 normalization discards magnitude, so a point that differs from the ID
    cluster *only* in norm lands on top of it after normalization and the
    normalized variant scores it as maximally in-distribution. The unnormalized
    variant flags it easily.

    This is the same mechanism, read in the other direction, that Sun et al.
    (2022) invoke when they report normalization as worth 61 FPR95 points: ID
    features tend to have larger norms than OOD ones, so *removing* magnitude
    helps when magnitude is misleading and hurts when magnitude is the signal.
    Which way it falls on frozen DINOv2 CLS embeddings is exactly the empirical
    question this thesis reports both variants to answer, and it is why
    normalization is a factor applied across families rather than baked into one
    scorer.
    """
    rng = np.random.default_rng(31337)
    direction = np.array([1.0, 0.0, 0.0])
    ref = direction * rng.normal(loc=5.0, scale=0.3, size=(300, 1))
    ref = ref + rng.normal(scale=0.02, size=ref.shape)  # a little angular spread
    id_query = direction * rng.normal(loc=5.0, scale=0.3, size=(50, 1))
    id_query = id_query + rng.normal(scale=0.02, size=id_query.shape)
    norm_outlier = direction * 40.0  # same direction, eight times the norm

    unnormalized = fit_knn("knn_unnormalized", ref, k=5)
    normalized = fit_knn("knn_normalized", ref, k=5)

    assert unnormalized.score(norm_outlier[None, :])[0] > unnormalized.score(id_query).max()
    assert normalized.score(norm_outlier[None, :])[0] < normalized.score(id_query).max()


def test_scores_are_non_negative_distances():
    """The Gram identity can produce a small negative squared distance through
    cancellation; it is clipped, so a NaN can never reach `np.sqrt`."""
    ref = gaussian_reference()
    scorer = fit_knn("knn_normalized", ref, k=1)
    scores = scorer.score(ref)  # every query coincides with a reference point
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_querying_the_reference_set_with_itself_gives_zero_at_k_equals_one():
    """A documented gotcha, asserted so it is not mistaken for a bug later.

    Every point is its own nearest neighbour, so scoring the reference set
    against itself gives r_1 = 0 throughout. The protocol never does
    this -- the reference set is CIFAR-10 train and every evaluated split is
    disjoint -- but a sanity check that scored train against train would look
    broken without this written down.

    The tolerance is 1e-6, not 0, and that is the Gram identity's one real cost.
    ``d^2 = ||q||^2 + ||r||^2 - 2 q.r`` has all three terms of order ``scale^2``
    cancelling to zero, so the absolute error in ``d^2`` is around
    ``eps * scale^2``, and the square root turns that into roughly
    ``sqrt(eps) * scale`` -- about 1e-8 here. Irrelevant to a k-th-neighbour
    ranking at k = 50, where neighbours are separated by far more than that, and
    visible only at exact coincidence.
    """
    ref = gaussian_reference()
    scorer = fit_knn("knn_unnormalized", ref, k=1)
    scores = scorer.score(ref)
    assert scores == pytest.approx(np.zeros(len(ref)), abs=1e-6)
    assert scores.max() < 1e-6


# --------------------------------------------------------------------------- #
# Numerics: the Gram identity, and blocking
# --------------------------------------------------------------------------- #


def test_gram_distances_match_explicit_pairwise_norms():
    """The identity ||q||^2 + ||r||^2 - 2 q.r must agree with the real thing.

    An explicit (n_query, n_ref, p) difference array is unaffordable at
    9,000 x 50,000, so the identity is what ships; this bounds the error it costs on
    a fixture small enough to compute both ways.
    """
    rng = np.random.default_rng(555)
    ref = rng.normal(size=(120, 4))
    q = rng.normal(size=(20, 4))
    scorer = fit_knn("knn_unnormalized", ref, k=7)

    explicit = np.sort(np.linalg.norm(ref[None, :, :] - q[:, None, :], axis=2), axis=1)[:, 6]
    assert scorer.score(q) == pytest.approx(explicit, rel=1e-10)


@pytest.mark.parametrize("block", [1, 3, 7, 512, 10_000])
def test_block_size_does_not_change_the_scores(block):
    """Blocking is a memory knob and nothing else."""
    ref = gaussian_reference()
    scorer = fit_knn("knn_normalized", ref, k=9)
    rng = np.random.default_rng(777)
    q = rng.normal(size=(23, ref.shape[1]))
    assert scorer.score(q, block=block) == pytest.approx(scorer.score(q, block=512), rel=1e-15)


def test_dimension_mismatch_between_query_and_reference_raises():
    scorer = fit_knn("knn_unnormalized", gaussian_reference(), k=3)
    with pytest.raises(ValueError, match="dimensions"):
        scorer.score(np.zeros((4, 2)))


def test_a_single_query_may_be_passed_as_a_one_dimensional_array():
    scorer = fit_knn("knn_unnormalized", ruler(), k=4)
    assert scorer.score(np.zeros(2)) == pytest.approx([4.0], rel=1e-15)


def test_scoring_is_float64_whatever_the_input_dtype():
    ref = gaussian_reference().astype(np.float32)
    scorer = fit_knn("knn_normalized", ref, k=3)
    assert scorer.reference.dtype == np.float64
    assert scorer.score(ref.astype(np.float32)).dtype == np.float64


# --------------------------------------------------------------------------- #
# Bookkeeping and the shared dump
# --------------------------------------------------------------------------- #


def test_both_variants_have_a_stable_colour_in_the_thesis_palette():
    for config in CONFIGURATIONS:
        assert config.name in style._SERIES_COLOR, config.name


def test_hyperparams_record_k_and_the_aggregation_choice():
    """The results table must be able to say k = 50, k-th, from the row alone."""
    from xai_ood.schema import decode_hyperparams

    decoded = decode_hyperparams(config_by_name("knn_normalized").encoded_hyperparams())
    assert decoded == {
        "k": 50,
        "aggregation": "kth",
        "normalize": True,
        "metric": "euclidean",
        "reference_split": "cifar10_train",
    }


def test_diagnostics_record_the_reference_set_size():
    """Sun et al.'s k = 50 transfers because the reference set is the same size.

    Recording n_reference is what lets that claim be checked against what
    actually ran.
    """
    scorer = fit_knn("knn_normalized", gaussian_reference(n=400))
    d = scorer.diagnostics()
    assert d["n_reference"] == 400
    assert d["k"] == 50
    assert d["aggregation"] == "kth"


def test_unknown_configuration_name_lists_the_two_reported_ones():
    with pytest.raises(KeyError, match="knn_normalized"):
        config_by_name("knn")


def test_knn_uses_the_shared_score_frame_and_lands_in_canonical_order():
    """The kNN family's "done when": scores land in canonical order.

    Note this calls the *same* `score_frame` the Gaussian family uses, imported
    from `methods.dump`. A family-specific dump would have to re-implement the
    `assert_canonical_order` call, and the one that got it wrong would be silent.
    """
    ref = gaussian_reference()
    scorers = fit_all_knn(ref, k=5)
    embeddings = np.array([[1.0, 2.0, 0.5, -1.0, 0.0], [3.0, 0.5, 1.0, 0.0, 2.0],
                           [-2.0, 1.0, 0.0, 1.0, -1.0], [0.0, -4.0, 2.0, 0.5, 1.0]])
    index = pd.DataFrame(
        {
            "split": ["id_test", "id_test", "csid", "csid"],
            "image_id": ["img_0002", "img_0001", "img_0001", "img_0001"],
            "corruption": [pd.NA, pd.NA, "fog", "fog"],
            "severity": [pd.NA, pd.NA, 3, 1],
        }
    )

    out = score_frame(index, embeddings, scorers)
    assert_canonical_order(out)
    assert list(out["split"]) == ["csid", "csid", "id_test", "id_test"]
    for name in scorers:
        assert name in out.columns
        assert np.all(np.isfinite(out[name].to_numpy()))


def test_knn_and_gaussian_scorers_dump_together_in_one_aligned_frame():
    """Cross-family alignment for free, which is what the bootstrap needs.

    One dump covering both families is trivially aligned; two dumps would need
    `assert_aligned` to prove the same thing afterwards.
    """
    from xai_ood.methods.gaussian import fit_all_scorers

    rng = np.random.default_rng(4242)
    ref = np.vstack([rng.normal(loc=[-4, 0, 0], scale=0.6, size=(150, 3)),
                     rng.normal(loc=[4, 0, 0], scale=0.6, size=(150, 3))])
    labels = np.array([0] * 150 + [1] * 150)

    scorers = {**fit_all_scorers(ref, labels), **fit_all_knn(ref, k=5)}
    assert len(scorers) == 10

    n = 12
    z = rng.normal(scale=5.0, size=(n, 3))
    index = pd.DataFrame(
        {
            "split": ["id_test"] * n,
            "image_id": [f"img_{i:05d}" for i in range(n)],
            "corruption": [pd.NA] * n,
            "severity": [pd.NA] * n,
        }
    )
    out = score_frame(index, z, scorers)
    assert_canonical_order(out)
    assert set(scorers).issubset(out.columns)


def test_duplicate_scorer_names_are_refused_by_the_dump():
    """Two scorers sharing a name would silently overwrite one column."""
    ref = gaussian_reference()
    a = fit_knn("knn_normalized", ref, k=3)
    b = KnnScorer(KnnConfig("knn_normalized", False, 3, reporting_tier="appendix"), ref)
    index = pd.DataFrame(
        {"split": ["id_test"], "image_id": ["img_0001"], "corruption": [pd.NA], "severity": [pd.NA]}
    )
    with pytest.raises(ValueError, match="duplicate scorer name"):
        score_frame(index, np.zeros((1, ref.shape[1])), [a, b])


def test_a_scorer_named_after_a_key_column_is_refused_by_the_dump():
    """`out[scorer.name] = ...` would overwrite a canonical key column.

    `dump.py` already refuses duplicate scorer names because one column would
    silently overwrite another; this is the same failure one step over, against
    the four columns that define row identity.

    It is not hypothetical-only. On a cs-ID frame -- corruption populated, so the
    half-null guard cannot fire -- a scorer named `severity` returning values
    that happen to land in {1..5} replaces the severity column with its scores
    and the dump passes every downstream check: a four-row
    fog/severity-(1,2,1,2) frame comes back as severity (1,2,3,4), with no error
    raised anywhere.

    Mutation killed: removing the reserved-name check restores that silent
    corruption.
    """
    class NamedAfterAKeyColumn:
        name = "severity"

        def score(self, x):
            return np.array([1.0, 2.0, 3.0, 4.0])

    index = pd.DataFrame(
        {
            "split": ["cifar10c_csid"] * 4,
            "image_id": ["img_0001", "img_0001", "img_0002", "img_0002"],
            "corruption": ["fog"] * 4,
            "severity": [1, 2, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="collide with the canonical key column"):
        score_frame(index, np.zeros((4, 3)), [NamedAfterAKeyColumn()])
