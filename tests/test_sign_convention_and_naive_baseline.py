# tests/test_sign_convention_and_naive_baseline.py
#
# **This file is a specification, and stays one.** It deliberately imports
# nothing from xai_ood: it states what any scorer must satisfy without importing
# one, which is exactly the property that would be lost by re-pointing it at the
# shipped functions. It therefore exercises none of src/ -- the whole file
# passes with src/xai_ood/methods/ deleted -- and that is by design, not an
# oversight.
#
# What this means for coverage: the sign-convention requirement is earned by
# tests/test_gaussian_scorers.py, which runs the *real* scorers (and whose
# test_the_openood_seam_adapter_is_the_only_negation exercises the real
# adapter). This file is the statement of the requirement; that file is the
# evidence the shipped code meets it.
#
# Re-pointing these fixtures at the real scorer functions would destroy the only
# thing this file uniquely provides, so it is deliberately not done here. It is
# done in test_gaussian_scorers.py instead, alongside rather than in place of
# this specification.
#
# One deliberate divergence from the shipped estimator: fit_gaussian below uses
# np.cov's n-1 divisor, while xai_ood.methods.covariance.fit_covariance uses n
# (the MLE, forced by the Ledoit-Wolf target). The difference is a constant
# rescaling of the covariance, and every assertion here
# is about *ordering*, which is invariant to it. The spec is about sign and rank,
# not about the divisor; the divisor is specified in covariance.py.
#
# Two requirements: higher = more OOD for every scorer, and the naive baseline
# as a diagonal restriction of the shared path. The first is the single
# highest-cost-if-skipped check in the project: mixing a distance-as-score with
# OpenOOD's confidence-as-score gives an AUROC of roughly one minus its true
# value, which
# looks like a plausible result rather than a crash.
#
# Needs numpy + pytest only. No GPU, no embedding cache.
import numpy as np
import pytest

def make_fixture(seed=0):
    """Tight 2D ID cluster + one far-away point that should score as OOD.

    Each caller constructs its own generator from `seed`. There is deliberately
    no module-level `rng` shared across tests: that made every fixture depend on
    which tests had already run, so `pytest -k` and the whole-file run produced
    different data. A specification's inputs must not move with the selection.
    """
    rng = np.random.default_rng(seed)
    id_points = rng.normal(loc=[0, 0], scale=[1.0, 0.3], size=(500, 2))
    ood_point = np.array([[8.0, 8.0]])
    return id_points, ood_point


def fit_gaussian(x, diagonal):
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    if diagonal:
        cov = np.diag(np.diag(cov))
    return mean, cov


def mahalanobis_sq(x, mean, cov):
    diff = x - mean
    inv = np.linalg.inv(cov)
    return np.einsum("ij,jk,ik->i", diff, inv, diff)


def diagonal_gaussian_nll(x, mean, cov):
    var = np.diag(cov)
    diff = x - mean
    return 0.5 * np.sum(diff**2 / var, axis=1) + 0.5 * np.sum(np.log(var))


def knn_score(query, reference, k=5, normalize=False):
    ref = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-10) if normalize else reference
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10) if normalize else query
    dists = np.linalg.norm(ref[None, :, :] - q[:, None, :], axis=2)
    kth = np.sort(dists, axis=1)[:, k - 1]
    return kth  # convention under test: higher distance = more OOD, no sign flip needed here


@pytest.mark.parametrize("diagonal", [True, False])
def test_marginal_sign_convention(diagonal):
    """Higher score = more OOD, for every marginal-model variant."""
    id_points, ood_point = make_fixture()
    mean, cov = fit_gaussian(id_points, diagonal=diagonal)
    id_scores = mahalanobis_sq(id_points, mean, cov)
    ood_score = mahalanobis_sq(ood_point, mean, cov)
    assert ood_score > id_scores.max()


@pytest.mark.parametrize("diagonal", [True, False])
def test_class_conditional_sign_convention(diagonal):
    """Two synthetic ID clusters; class-conditional score = min distance to any class mean."""
    rng = np.random.default_rng(10)
    cluster_a = rng.normal(loc=[-3, 0], scale=0.5, size=(250, 2))
    cluster_b = rng.normal(loc=[3, 0], scale=0.5, size=(250, 2))
    id_points = np.vstack([cluster_a, cluster_b])
    ood_point = np.array([[0.0, 20.0]])

    pooled = np.cov(np.vstack([cluster_a - cluster_a.mean(0), cluster_b - cluster_b.mean(0)]), rowvar=False)
    if diagonal:
        pooled = np.diag(np.diag(pooled))
    means = [cluster_a.mean(axis=0), cluster_b.mean(axis=0)]

    def score(pts):
        return np.min([mahalanobis_sq(pts, m, pooled) for m in means], axis=0)

    id_scores = score(id_points)
    ood_score = score(ood_point)
    assert ood_score > id_scores.max()


def test_knn_sign_convention():
    id_points, ood_point = make_fixture()
    id_scores = knn_score(id_points, id_points, k=5)
    ood_score = knn_score(ood_point, id_points, k=5)
    assert ood_score > id_scores.max()


def test_openood_confidence_adapter_flips_sign():
    """The one seam adapter: OpenOOD confidence (higher=ID) -> OOD score (higher=OOD)."""

    def ood_score_from_openood_confidence(conf):
        return -conf

    id_like_confidence = np.array([5.0, 4.5, 6.0])   # OpenOOD says: confidently ID
    ood_like_confidence = np.array([-3.0])            # OpenOOD says: not ID

    id_ood_scores = ood_score_from_openood_confidence(id_like_confidence)
    ood_ood_scores = ood_score_from_openood_confidence(ood_like_confidence)
    assert ood_ood_scores.min() > id_ood_scores.max()


def test_naive_baseline_equals_diagonal_mahalanobis_up_to_constant():
    """Diagonal Gaussian NLL and squared diagonal Mahalanobis distance must give
    identical rankings (hence identical AUROC): they differ only by a constant
    that does not depend on the query point, for a fixed fitted covariance."""
    id_points, ood_point = make_fixture()
    mean, cov = fit_gaussian(id_points, diagonal=True)
    query = np.vstack([id_points, ood_point])

    nll = diagonal_gaussian_nll(query, mean, cov)
    md_sq = mahalanobis_sq(query, mean, cov)

    diff = nll - 0.5 * md_sq
    # Must be constant across every sample (this is the log-det term, sample-independent).
    assert np.allclose(diff, diff[0]), "log-det term became sample-dependent -- naive/MDS-diagonal equivalence broken"


# --------------------------------------------------------------------------- #
# The consequences, asserted directly
# --------------------------------------------------------------------------- #


def test_naive_baseline_and_diagonal_mahalanobis_give_identical_auroc():
    """The consequence the constant-offset test exists to buy, asserted directly.

    The requirement is that the two produce *identical AUROC*. The
    offset test proves the ranking is preserved; this one checks the metric that
    actually appears in the results table, computed without sklearn so the test
    stays dependency-free.
    """
    id_points, ood_point = make_fixture()
    ood_points = np.vstack(
        [ood_point, np.random.default_rng(20).normal(loc=[7, -7], scale=0.4, size=(50, 2))]
    )
    mean, cov = fit_gaussian(id_points, diagonal=True)

    def auroc(id_scores, ood_scores):
        # Rank-based AUROC with correct tie handling (Mann-Whitney U).
        combined = np.concatenate([id_scores, ood_scores])
        order = combined.argsort()
        ranks = np.empty_like(combined)
        ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
        # average ranks within ties
        for value in np.unique(combined):
            mask = combined == value
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        n_ood = len(ood_scores)
        n_id = len(id_scores)
        rank_sum_ood = ranks[n_id:].sum()
        return (rank_sum_ood - n_ood * (n_ood + 1) / 2) / (n_ood * n_id)

    auroc_nll = auroc(
        diagonal_gaussian_nll(id_points, mean, cov),
        diagonal_gaussian_nll(ood_points, mean, cov),
    )
    auroc_md = auroc(
        mahalanobis_sq(id_points, mean, cov),
        mahalanobis_sq(ood_points, mean, cov),
    )
    assert auroc_nll == pytest.approx(auroc_md, abs=1e-12)


def test_double_negation_canary_is_detectable():
    """The zero-variance canary, as a test rather than an eyeball.

    An accidental second negation anywhere upstream turns AUROC into its
    complement. Assert that the complement is genuinely far from the true value
    on this fixture, so logging both and eyeballing them is actually informative
    rather than a distinction without a difference near 0.5.
    """
    id_points, ood_point = make_fixture()
    ood_points = np.vstack(
        [ood_point, np.random.default_rng(20).normal(loc=[7, -7], scale=0.4, size=(50, 2))]
    )
    mean, cov = fit_gaussian(id_points, diagonal=False)

    id_scores = mahalanobis_sq(id_points, mean, cov)
    ood_scores = mahalanobis_sq(ood_points, mean, cov)

    # Correct convention: every OOD point outscores every ID point here.
    assert ood_scores.min() > id_scores.max()
    # Flipped convention: the ordering reverses completely, so the mistake is
    # loud rather than a few points of drift.
    assert (-ood_scores).max() < (-id_scores).min()


def test_fixtures_are_reproducible_regardless_of_which_tests_ran_first():
    """No shared generator state: the same seed must give the same fixture.

    This file used a single module-level `rng = np.random.default_rng(0)`
    consumed by every test in turn, so each test's data depended on which tests
    had already run. Nothing was broken by it -- the file passed whole and each
    test passed in isolation -- but the two runs used *different* fixtures, and
    a `pytest -k`, a `-p randomly`, or a new test inserted in the middle would
    silently change every downstream fixture. A sign-convention suite whose
    inputs move when the selection changes is not a fixed specification.

    Mutation killed: restoring a shared module-level generator, under which
    `make_fixture()` returns different data on each call.
    """
    first_id, first_ood = make_fixture()
    second_id, second_ood = make_fixture()
    assert np.array_equal(first_id, second_id)
    assert np.array_equal(first_ood, second_ood)

    assert not np.array_equal(make_fixture(seed=1)[0], make_fixture(seed=2)[0])
