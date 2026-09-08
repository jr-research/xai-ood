# tests/test_bootstrap.py
#
# The cluster bootstrap.
#
# THE FILE'S CENTRE OF GRAVITY is `test_the_full_spectrum_interval_does_not_
# shrink_when_duplicate_rows_are_added` and the two tests that follow it. They
# are the reason the module exists, and they are written mutation-first: each
# guard names the mutant it kills in its docstring, and the mutant is
# constructed and run in the file rather than asserted about in prose. A guard
# that does not fail on the bug it names is not a guard.
#
# Two mutants recur, both of which are the *same* defect arriving by different
# routes -- the resampling unit collapsing from the photograph to the row:
#
#   M1  row-level bootstrap. Every row is its own cluster. This is what the
#       obvious implementation does, and it shrinks the full-spectrum interval
#       by up to about sqrt(76).
#   M2  the image_id join silently failing. cs-ID rows carry a corrupted path
#       or a bare stem while the clean ID-test rows carry the source path, so
#       nothing matches, every cs-ID photograph becomes a singleton cluster,
#       and M1 happens without anyone writing a row-level bootstrap. This is
#       the failure `csid_imglist.py`'s docstring warns about by name.

import numpy as np
import pandas as pd
import pytest

from xai_ood.bootstrap import (
    B_DEFAULT,
    CI_LEVEL,
    CSID_SPLIT,
    DEFAULT_OOD_GROUPS,
    FPR95_FULL_SPECTRUM_CAVEAT,
    ID_TEST_SPLIT,
    PROTOCOL_ID_SPLITS,
    QUANTILE_METHOD,
    REQUIRED_PAIRWISE_CLAIMS,
    BootstrapPlan,
    ClusterIndex,
    Interval,
    bootstrap_manifest,
    build_cluster_index,
    build_plan,
    degradation_table,
    differences_table,
    percentile_interval,
    replicate_indices,
    results_table,
    run_bootstrap,
)
from xai_ood.csid_imglist import CORRUPTIONS, ROWS_PER_DRAWN_IMAGE, SEVERITIES
from xai_ood.metrics import auroc, fpr_at_tpr, threshold_at_tpr
from xai_ood.schema import PROTOCOLS, RESULTS_COLUMNS, SPLIT_ROLES, canonical_sort
from xai_ood.visualization.style import AUROC_SCALE

PROVENANCE = dict(
    seed=20260826,
    repo_commit="0" * 40,
    openood_commit="8d44375e4c695d03d2b97850b754f24fd4bda447",
    timestamp="2026-08-26T12:00:00Z",
)


# --------------------------------------------------------------------------- #
# Fixtures. Synthetic score arrays only; nothing here reads an embedding.
# --------------------------------------------------------------------------- #


def make_frame(
    *,
    n_photographs=150,
    n_subset=None,
    n_ood=12_000,
    seed=0,
    jitter=0.0,
    scorers=("s",),
    id_shift=0.0,
    ood_shift=1.5,
    ood_splits=("cifar100",),
    corruptions=CORRUPTIONS,
    severities=SEVERITIES,
    csid_image_id=None,
):
    """A per-sample score frame with the real cs-ID structure.

    ``n_subset`` photographs own ``1 + len(corruptions) * len(severities)`` rows;
    the rest own one. With ``jitter=0`` a corrupted copy carries *exactly* its
    clean row's score, which is the perfectly-correlated limit the sqrt(76)
    argument is stated in and the case where the non-shrinkage property is an
    exact equality rather than an approximation.

    ``csid_image_id`` lets a test break the join on purpose (mutant M2).
    """
    n_subset = n_photographs if n_subset is None else n_subset
    rng = np.random.default_rng(seed)
    ids = [f"cifar10/test/cat/{i:05d}.png" for i in range(n_photographs)]
    base = {name: rng.normal(id_shift, 1.0, size=n_photographs) for name in scorers}

    rows = []
    for i, image_id in enumerate(ids):
        row = dict(
            split=ID_TEST_SPLIT, image_id=image_id, corruption=None, severity=None
        )
        row.update({name: base[name][i] for name in scorers})
        rows.append(row)

    for i, image_id in enumerate(ids[:n_subset]):
        for corruption in corruptions:
            for severity in severities:
                noise = rng.normal(0.0, jitter) if jitter else 0.0
                row = dict(
                    split=CSID_SPLIT,
                    image_id=image_id if csid_image_id is None
                    else csid_image_id(image_id, corruption, severity),
                    corruption=corruption,
                    severity=severity,
                )
                row.update({name: base[name][i] + noise for name in scorers})
                rows.append(row)

    for split in ood_splits:
        for j in range(n_ood):
            row = dict(
                split=split, image_id=f"{split}/{j:05d}.png",
                corruption=None, severity=None,
            )
            row.update(
                {name: rng.normal(ood_shift, 1.0) for name in scorers}
            )
            rows.append(row)

    return canonical_sort(pd.DataFrame(rows), name="fixture")


def one_group(frame, **kwargs):
    splits = sorted(set(frame["split"]) - {ID_TEST_SPLIT, CSID_SPLIT})
    return build_plan(frame, ood_groups={"near_ood": tuple(splits)}, **kwargs)


def row_level_mutant(plan, protocol):
    """**Mutant M1**, one protocol's leg: every row is its own cluster.

    This is the obvious implementation -- resampling rows of the cached
    per-sample score array. Built as *data*
    rather than as a code branch, because that is what the bug actually is: a
    row-level bootstrap is this module driven by a degenerate cluster
    assignment, which is exactly what mutant M2 produces by accident. Nothing
    in ``src/`` can reach this state.

    Note it has to be built **one protocol at a time**, and that is itself
    informative rather than an inconvenience. The standard leg has 150 rows and
    the full-spectrum leg has 11,400, so a row-level procedure has two
    different resampling universes and cannot pair the legs at all: the same
    drawn ids on both legs is not merely violated, it is unavailable.
    ``BootstrapPlan.__post_init__`` refuses a plan that tries to have it both
    ways, which is the guard this helper originally tripped over.
    """
    original = plan.id_by_protocol[protocol]
    n = original.n_rows
    rows = ClusterIndex(
        name=f"row_level_mutant:{protocol}",
        labels=np.arange(n),
        starts=np.arange(n, dtype=np.int64),
        counts=np.ones(n, dtype=np.int64),
        positions=original.positions,
    )
    return BootstrapPlan(
        id_labels=np.arange(n),
        id_by_protocol={protocol: rows},
        ood_by_dataset=plan.ood_by_dataset,
        ood_groups=plan.ood_groups,
        n_rows=plan.n_rows,
        key_digest=plan.key_digest,
        diagnostics=plan.diagnostics,
    )


def width(result, scorer, protocol, group, metric="auroc"):
    low, high = percentile_interval(
        result.replicates(scorer, protocol, group, metric), result.level
    )
    return high - low


# --------------------------------------------------------------------------- #
# 1. THE TEST THIS MODULE EXISTS FOR
# --------------------------------------------------------------------------- #


def test_the_full_spectrum_interval_does_not_shrink_when_duplicate_rows_are_added():
    """**The property the whole procedure exists for.**

    Moving from the standard to the full-spectrum protocol adds no new
    photographs, only re-corrupted copies of photographs already present. A
    correct interval on the degradation must therefore not shrink.

    In the perfectly-correlated limit -- every corrupted copy carrying its
    clean row's exact score -- the statement is an *equality*, not an
    inequality. A drawn photograph contributes 76 identical scores instead of
    1, so the replicate's empirical ID distribution is unchanged and its AUROC
    is the same number. The two replicate arrays are equal element for element
    and the interval widths are identical.

    Mutant killed: **M1**, the row-level bootstrap. See the test below, which
    builds it and shows this assertion failing against it.
    """
    frame = make_frame(seed=1)
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=11, n_replicates=200, metrics=("auroc",))

    standard = result.replicates("s", "standard", "near_ood")
    full = result.replicates("s", "full_spectrum", "near_ood")

    # Element for element, not merely in distribution.
    assert np.array_equal(standard, full)

    w_std = width(result, "s", "standard", "near_ood")
    w_full = width(result, "s", "full_spectrum", "near_ood")
    assert w_std > 0.0
    assert w_full / w_std == pytest.approx(1.0, rel=1e-12)


def test_a_row_level_bootstrap_shrinks_the_interval_and_the_guard_above_catches_it():
    """**Mutant M1 constructed, run, and killed.**

    The same fixture and the same seed as the guard above, with the full-
    spectrum ID clusters collapsed to one row each. The interval falls to a
    fraction of the standard one -- toward the ``1 / sqrt(76) ~ 0.115`` the
    argument above predicts, bounded away from it only because the OOD
    side still contributes sampling noise that no ID-side change can remove.

    Measured on this fixture: ratio ~0.17 against the correct procedure's
    exactly 1.0. The guard's `approx(1.0)` assertion fails on it, which is what
    makes it a guard rather than a description.
    """
    frame = make_frame(seed=1)
    plan = one_group(frame)
    legs = {
        protocol: run_bootstrap(
            frame,
            row_level_mutant(plan, protocol),
            seed=11,
            n_replicates=200,
            metrics=("auroc",),
        )
        for protocol in ("standard", "full_spectrum")
    }

    w_std = width(legs["standard"], "s", "standard", "near_ood")
    w_full = width(legs["full_spectrum"], "s", "full_spectrum", "near_ood")
    ratio = w_full / w_std
    assert ratio < 0.5, f"the mutant did not shrink the interval (ratio {ratio})"

    # The guard above, applied to the mutant, must fail.
    with pytest.raises(AssertionError):
        assert ratio == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(AssertionError):
        assert np.array_equal(
            legs["standard"].replicates("s", "standard", "near_ood"),
            legs["full_spectrum"].replicates("s", "full_spectrum", "near_ood"),
        )


def test_a_plan_whose_two_legs_disagree_about_the_cluster_universe_is_refused():
    """The guard that the row-level mutant tripped when it was first written.

    One ``drawn`` array of cluster indices is applied to both protocol indexes.
    If the legs are over different universes the indices are still in
    range on one of them, so nothing raises and the full-spectrum leg quietly
    resamples the wrong rows -- which produced a wrong variance that looked
    plausible until it was compared against a hand-rolled row-level bootstrap.
    """
    frame = make_frame(seed=1, n_photographs=20, n_ood=30)
    plan = one_group(frame)
    mismatched = row_level_mutant(plan, "full_spectrum").id_by_protocol[
        "full_spectrum"
    ]
    with pytest.raises(ValueError, match="clusters but"):
        BootstrapPlan(
            id_labels=plan.id_labels,
            id_by_protocol={
                "standard": plan.id_by_protocol["standard"],
                "full_spectrum": mismatched,
            },
            ood_by_dataset=plan.ood_by_dataset,
            ood_groups=plan.ood_groups,
            n_rows=plan.n_rows,
            key_digest=plan.key_digest,
        )


def test_the_guard_still_holds_when_corruption_only_partly_correlates():
    """The realistic case: corrupted copies are noisy, not identical, and only
    a subset of photographs is corrupted at all -- 60 of 150 here, echoing the
    real 2,000 of 9,000.

    The equality of the previous test becomes an inequality, and what is
    asserted is the property the declared procedure actually commits to: the
    interval does not *shrink*. It is allowed to widen, and on this fixture it
    does slightly, because the jitter is real extra variation rather than
    duplicated data.
    """
    frame = make_frame(seed=5, n_subset=60, jitter=0.4)
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=11, n_replicates=200, metrics=("auroc",))
    ratio = width(result, "s", "full_spectrum", "near_ood") / width(
        result, "s", "standard", "near_ood"
    )
    assert ratio > 0.9, f"full-spectrum interval shrank to {ratio} of standard"


def test_a_replicate_carries_every_row_a_drawn_photograph_owns():
    """The structural form of the same property, independent of any interval.

    Under the cluster bootstrap a full-spectrum replicate contains exactly the
    photographs the standard one does, each appearing 76 times as often. That
    is checkable directly on the index arrays, with no statistics involved, and
    it is the sharpest statement of "the resampling unit is the photograph".

    Mutants killed: **M1** and **M2**, both of which break the 76:1 relation.
    """
    frame = make_frame(seed=2, n_photographs=40, n_ood=50)
    plan = one_group(frame)
    image_id = frame["image_id"].to_numpy()

    rng = np.random.default_rng(3)
    for id_rows, _ in replicate_indices(plan, rng, 5):
        standard = pd.Series(image_id[id_rows["standard"]]).value_counts()
        full = pd.Series(image_id[id_rows["full_spectrum"]]).value_counts()
        assert set(standard.index) == set(full.index)
        assert (full[standard.index] == ROWS_PER_DRAWN_IMAGE * standard).all()
        assert len(id_rows["standard"]) == plan.n_id_clusters


def test_cs_id_rows_whose_image_id_does_not_join_are_refused():
    """**Mutant M2 killed, at the plan rather than in the interval.**

    The cs-ID builder pins ``image_id`` to the *source* ID-test path exactly
    so a photograph's clean row and its 76 corrupted rows share a cluster. If a
    later dump uses the corrupted path instead, nothing about the frame is
    malformed: it is canonically ordered, its keys are unique, both per-frame
    checks pass. The clusters just silently become singletons and M1 happens by
    itself.

    ``build_plan`` refuses instead, because the ID cluster universe is derived
    from the ID-test rows alone and a cs-ID row that names no ID-test
    photograph has nowhere to go.
    """
    broken = make_frame(
        seed=1,
        n_photographs=20,
        n_ood=30,
        csid_image_id=lambda image_id, corruption, severity: (
            f"cifar10c/{corruption}/{severity}/{image_id.rsplit('/', 1)[-1]}"
        ),
    )
    with pytest.raises(ValueError, match="not in the cluster universe"):
        one_group(broken)


# --------------------------------------------------------------------------- #
# 2. Clusters
# --------------------------------------------------------------------------- #


def test_gather_returns_the_rows_of_the_drawn_clusters_in_order():
    """Hand-built: cluster 'a' owns rows [0, 1, 2], 'b' owns [3], 'c' owns
    [4, 5]. Drawing (c, a, c) must give [4, 5, 0, 1, 2, 4, 5]."""
    index = build_cluster_index(
        ["a", "a", "a", "b", "c", "c"],
        np.arange(6),
        universe=["a", "b", "c"],
        name="hand",
    )
    assert index.counts.tolist() == [3, 1, 2]
    assert index.gather(np.array([2, 0, 2])).tolist() == [4, 5, 0, 1, 2, 4, 5]


def test_gather_handles_an_empty_draw():
    index = build_cluster_index(["a"], np.arange(1), universe=["a"], name="hand")
    assert index.gather(np.array([], dtype=np.int64)).size == 0


def test_gather_is_insensitive_to_the_row_order_inside_a_cluster():
    """Rows of one photograph enter the replicate together; which order they
    are concatenated in cannot matter, because every metric downstream is a
    function of the multiset of scores."""
    scores = np.array([10.0, 20.0, 30.0, 40.0])
    a = build_cluster_index(["p", "p", "q", "q"], np.arange(4), universe=["p", "q"], name="a")
    b = build_cluster_index(["q", "p", "q", "p"], np.array([2, 0, 3, 1]), universe=["p", "q"], name="b")
    drawn = np.array([0, 1, 0])
    assert sorted(scores[a.gather(drawn)]) == sorted(scores[b.gather(drawn)])


def test_a_duplicate_cluster_label_in_the_universe_raises():
    with pytest.raises(ValueError, match="duplicate labels"):
        build_cluster_index(["a"], np.arange(1), universe=["a", "a"], name="hand")


def test_a_cluster_owning_no_rows_raises():
    with pytest.raises(ValueError, match="own no rows"):
        build_cluster_index(["a"], np.arange(1), universe=["a", "b"], name="hand")


def test_a_label_outside_the_universe_names_the_failure_it_causes():
    with pytest.raises(ValueError, match="row-level procedure"):
        build_cluster_index(["a", "z"], np.arange(2), universe=["a"], name="hand")


# --------------------------------------------------------------------------- #
# 3. The plan
# --------------------------------------------------------------------------- #


def test_the_cluster_universe_is_the_id_test_photographs():
    frame = make_frame(seed=1, n_photographs=30, n_subset=10, n_ood=40)
    plan = one_group(frame)
    assert plan.n_id_clusters == 30
    assert set(plan.id_labels) == set(
        frame.loc[frame["split"] == ID_TEST_SPLIT, "image_id"]
    )


def test_cluster_sizes_are_non_uniform_by_construction_and_need_no_correction():
    """2,000 photographs own 76 rows, 7,000 own 1 -- here 10 and 20. The
    procedure resamples ids and carries whatever rows each id owns, which is
    the whole reason no correction is applied."""
    frame = make_frame(seed=1, n_photographs=30, n_subset=10, n_ood=40)
    plan = one_group(frame)
    counts = plan.id_by_protocol["full_spectrum"].counts
    assert sorted(set(counts.tolist())) == [1, ROWS_PER_DRAWN_IMAGE]
    assert (counts == ROWS_PER_DRAWN_IMAGE).sum() == 10
    assert (plan.id_by_protocol["standard"].counts == 1).all()
    assert plan.diagnostics["cluster_size_range"]["full_spectrum"] == [
        1, ROWS_PER_DRAWN_IMAGE
    ]


def test_each_ood_dataset_is_its_own_universe():
    """Resampled within dataset, so group composition stays fixed."""
    frame = make_frame(seed=1, n_photographs=20, n_ood=30, ood_splits=("cifar100", "tin"))
    plan = build_plan(frame, ood_groups={"near_ood": ("cifar100", "tin")})
    assert set(plan.ood_by_dataset) == {"cifar100", "tin"}
    assert plan.ood_by_dataset["cifar100"].n_clusters == 30
    assert plan.ood_by_dataset["tin"].n_clusters == 30


def test_group_composition_is_fixed_across_replicates():
    """The point of within-dataset resampling, measured: every replicate
    contributes exactly 30
    cifar100 rows and 30 tin rows, so the near-OOD average is never an average
    over a group whose mixture moved."""
    frame = make_frame(seed=1, n_photographs=20, n_ood=30, ood_splits=("cifar100", "tin"))
    plan = build_plan(frame, ood_groups={"near_ood": ("cifar100", "tin")})
    split = frame["split"].to_numpy()
    rng = np.random.default_rng(0)
    for _, ood_rows in replicate_indices(plan, rng, 10):
        counts = pd.Series(split[ood_rows["near_ood"]]).value_counts()
        assert counts["cifar100"] == 30
        assert counts["tin"] == 30


def test_a_missing_cs_id_split_raises_rather_than_reporting_zero_degradation():
    """Without cs-ID rows the two protocols are the same set, so every scorer's
    degradation comes out as exactly 0.000 with a [0, 0] interval -- a result
    that looks like a finding and is a missing input."""
    frame = make_frame(seed=1, n_photographs=20, n_subset=0, n_ood=30)
    with pytest.raises(ValueError, match="degradation would come out as exactly zero"):
        one_group(frame)
    plan = one_group(frame, allow_missing_csid=True)
    assert plan.diagnostics["n_csid_rows"] == 0


def test_a_split_belonging_to_no_group_raises():
    """Forgetting one far-OOD dataset changes the far-OOD average without
    changing anything visible in the output."""
    frame = make_frame(seed=1, n_photographs=20, n_ood=25, ood_splits=("cifar100", "tin"))
    with pytest.raises(ValueError, match="belong to no protocol"):
        build_plan(frame, ood_groups={"near_ood": ("cifar100",)})
    plan = build_plan(
        frame, ood_groups={"near_ood": ("cifar100",)}, allow_unused_splits=True
    )
    assert plan.diagnostics["unused_splits"] == ["tin"]


def test_a_dataset_in_two_groups_raises():
    frame = make_frame(seed=1, n_photographs=20, n_ood=25, ood_splits=("cifar100",))
    with pytest.raises(ValueError, match="more than one OOD group"):
        build_plan(
            frame, ood_groups={"near_ood": ("cifar100",), "far_ood": ("cifar100",)}
        )


def test_a_named_dataset_with_no_rows_raises():
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    with pytest.raises(ValueError, match="has no rows"):
        build_plan(frame, ood_groups={"near_ood": ("cifar100", "svhn")})


def test_a_repeated_id_test_image_id_raises():
    """Two clean rows for one photograph would make it two clusters. In
    practice ``assert_canonical_order`` catches this first -- the two rows
    share a sort key -- so this is defence at the next layer down, for a frame
    that reached ``build_plan`` without going through a dump."""
    frame = make_frame(seed=1, n_photographs=20, n_subset=0, n_ood=25)
    rows = frame.copy()
    clean = rows.index[rows["split"] == ID_TEST_SPLIT]
    rows.loc[clean[1], "image_id"] = rows.loc[clean[0], "image_id"]
    with pytest.raises(ValueError, match="repeat an image_id"):
        one_group(rows, allow_missing_csid=True)


def test_the_split_names_are_schema_split_roles():
    """The plan's two ID split names are enumerated in the frozen schema rather
    than being strings this module invented."""
    assert ID_TEST_SPLIT in SPLIT_ROLES
    assert CSID_SPLIT in SPLIT_ROLES
    assert set(PROTOCOL_ID_SPLITS) == set(PROTOCOLS)
    assert PROTOCOL_ID_SPLITS["standard"] == (ID_TEST_SPLIT,)
    assert PROTOCOL_ID_SPLITS["full_spectrum"] == (ID_TEST_SPLIT, CSID_SPLIT)
    for group in DEFAULT_OOD_GROUPS:
        assert group in SPLIT_ROLES


# --------------------------------------------------------------------------- #
# 4. Shared resamples, and reproducibility
# --------------------------------------------------------------------------- #


def test_one_index_set_per_replicate_is_applied_to_every_scorer():
    """One index set per replicate, tested through a consequence of it.

    ``s`` and ``s_monotone`` are related by a strictly increasing transform, so
    on *any* fixed sample their AUROCs are identical -- AUROC is a rank
    statistic. If the two scorers were resampled independently, their replicate
    AUROCs would differ. Element-for-element equality across 200 replicates is
    therefore evidence about the index, not about the scores.
    """
    frame = make_frame(seed=4, n_photographs=40, n_ood=60)
    frame["s_monotone"] = np.exp(frame["s"].to_numpy() * 2.0 + 1.0)
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=9, n_replicates=200, metrics=("auroc",))
    for protocol in result.protocols:
        assert np.array_equal(
            result.replicates("s", protocol, "near_ood"),
            result.replicates("s_monotone", protocol, "near_ood"),
        )


def test_the_recorded_seed_reproduces_the_run_bit_for_bit():
    """What ``bootstrap_manifest`` claims when it records a seed instead of a
    thousand index arrays."""
    frame = make_frame(seed=6, n_photographs=40, n_ood=60)
    plan = one_group(frame)
    kwargs = dict(seed=20260826, n_replicates=50)
    first = run_bootstrap(frame, plan, **kwargs)
    second = run_bootstrap(frame, plan, **kwargs)
    for key, values in first.values.items():
        assert np.array_equal(values, second.values[key])
    assert bootstrap_manifest(first)["seed"] == 20260826


def test_a_different_seed_gives_a_different_run():
    frame = make_frame(seed=6, n_photographs=40, n_ood=60)
    plan = one_group(frame)
    a = run_bootstrap(frame, plan, seed=1, n_replicates=50, metrics=("auroc",))
    b = run_bootstrap(frame, plan, seed=2, n_replicates=50, metrics=("auroc",))
    assert not np.array_equal(
        a.replicates("s", "standard", "near_ood"),
        b.replicates("s", "standard", "near_ood"),
    )


def test_replicate_values_match_an_independent_recomputation():
    """Drives ``replicate_indices`` directly and recomputes every metric with
    the public ``xai_ood.metrics`` entry points, so the run loop's fast path
    (one sort per protocol, reused across groups) is checked against the slow
    definition rather than against itself."""
    frame = make_frame(seed=8, n_photographs=30, n_ood=40)
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=5, n_replicates=25)
    scores = frame["s"].to_numpy()

    rng = np.random.default_rng(5)
    for b, (id_rows, ood_rows) in enumerate(replicate_indices(plan, rng, 25)):
        for protocol in result.protocols:
            id_scores = scores[id_rows[protocol]]
            ood_scores = scores[ood_rows["near_ood"]]
            assert result.replicates("s", protocol, "near_ood")[b] == pytest.approx(
                auroc(id_scores, ood_scores), rel=1e-15
            )
            assert result.replicates("s", protocol, "near_ood", "fpr95")[
                b
            ] == pytest.approx(fpr_at_tpr(id_scores, ood_scores), rel=1e-15)


def test_the_fpr_threshold_is_recomputed_inside_each_replicate():
    """The per-replicate FPR threshold, as a mutation the test can distinguish.

    Mutant: fix the 95%-TPR threshold once from the full sample and reuse it in
    every replicate -- "otherwise the interval conditions on a quantity that is
    itself estimated". The mutant is computed here and asserted to *disagree*
    with the shipped values on a majority of replicates, so a change to the
    shipped code that froze the threshold would turn this red.
    """
    frame = make_frame(seed=9, n_photographs=60, n_ood=80)
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=4, n_replicates=40, metrics=("fpr95",))
    scores = frame["s"].to_numpy()

    fixed_tau = threshold_at_tpr(
        np.sort(scores[plan.id_by_protocol["standard"].positions])
    )
    rng = np.random.default_rng(4)
    disagreements = 0
    for b, (id_rows, ood_rows) in enumerate(replicate_indices(plan, rng, 40)):
        ood_scores = scores[ood_rows["near_ood"]]
        frozen = np.count_nonzero(ood_scores <= fixed_tau) / ood_scores.size
        shipped = result.replicates("s", "standard", "near_ood", "fpr95")[b]
        if frozen != pytest.approx(shipped, rel=1e-15):
            disagreements += 1
    assert disagreements > 20, (
        "the shipped FPR values are indistinguishable from a frozen-threshold "
        "implementation, so this test cannot detect the mutation it names"
    )


def test_point_estimates_come_from_the_data_not_from_the_replicates():
    """The percentile interval is placed around the full-sample estimate. Using
    the bootstrap mean instead would be a different (and slightly biased)
    estimator, and the difference is invisible in the table."""
    frame = make_frame(seed=10, n_photographs=40, n_ood=60)
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=2, n_replicates=100, metrics=("auroc",))
    scores = frame["s"].to_numpy()
    expected = auroc(
        scores[plan.id_by_protocol["standard"].positions],
        scores[plan.group_positions()["near_ood"]],
    )
    assert result.point[("s", "standard", "near_ood", "auroc")] == pytest.approx(
        expected, rel=1e-15
    )
    assert result.marginal("s", "standard", "near_ood").point == pytest.approx(
        expected, rel=1e-15
    )


# --------------------------------------------------------------------------- #
# 5. Input checks on the frames
# --------------------------------------------------------------------------- #


def test_misaligned_frames_are_caught_before_any_indexing():
    """``assert_aligned`` exists for this case specifically: two
    frames can each be perfectly sorted and still cover different row sets, and
    the per-frame check cannot see it."""
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    plan = one_group(frame)
    other = frame.iloc[:-1].reset_index(drop=True).rename(columns={"s": "t"})
    with pytest.raises(ValueError, match="rows, but"):
        run_bootstrap({"a": frame, "b": other}, plan, seed=1, n_replicates=5)


def test_two_frames_covering_different_rows_are_caught():
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    plan = one_group(frame)
    other = frame.rename(columns={"s": "t"}).copy()
    other.loc[0, "image_id"] = "cifar10/test/cat/99999.png"
    other = canonical_sort(other, name="other")
    with pytest.raises(ValueError, match="sort keys differ"):
        run_bootstrap({"a": frame, "b": other}, plan, seed=1, n_replicates=5)


def test_frames_that_do_not_match_the_plan_are_refused():
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    plan = one_group(frame)
    relabelled = frame.copy()
    relabelled["image_id"] = relabelled["image_id"].str.replace("cat", "dog")
    relabelled = canonical_sort(relabelled, name="relabelled")
    with pytest.raises(ValueError, match="do not match the plan"):
        run_bootstrap(relabelled, plan, seed=1, n_replicates=5)


def test_a_duplicate_score_column_across_frames_is_refused():
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    plan = one_group(frame)
    with pytest.raises(ValueError, match="more than one frame"):
        run_bootstrap({"a": frame, "b": frame.copy()}, plan, seed=1, n_replicates=5)


def test_non_finite_scores_are_refused():
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    plan = one_group(frame)
    frame.loc[3, "s"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        run_bootstrap(frame, plan, seed=1, n_replicates=5)


def test_an_unknown_metric_is_refused():
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    plan = one_group(frame)
    with pytest.raises(ValueError, match="unknown metric"):
        run_bootstrap(frame, plan, seed=1, n_replicates=5, metrics=("accuracy",))


# --------------------------------------------------------------------------- #
# 6. Paired differences
# --------------------------------------------------------------------------- #


def test_a_scorer_differs_from_itself_by_exactly_zero():
    frame = make_frame(seed=1, n_photographs=30, n_ood=40)
    frame["copy"] = frame["s"]
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=1, n_replicates=100, metrics=("auroc",))
    interval = result.difference("s", "copy", "standard", "near_ood")
    assert (interval.point, interval.low, interval.high) == (0.0, 0.0, 0.0)
    assert interval.excludes_zero is False


def test_a_genuinely_better_scorer_produces_an_interval_excluding_zero():
    """The fixture pair the bootstrap's "done when" asks for: two scorers, one
    strictly more separable than the other, and a difference interval that
    excludes zero."""
    rng = np.random.default_rng(21)
    frame = make_frame(seed=21, n_photographs=200, n_ood=400, scorers=("good", "weak"))
    # Make `weak` a noisier read of the same photographs: same ranking on
    # average, far more overlap between the ID and OOD score distributions.
    frame["weak"] = frame["good"] + rng.normal(0.0, 2.5, size=len(frame))
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=3, n_replicates=300, metrics=("auroc",))
    interval = result.difference("good", "weak", "standard", "near_ood")
    assert interval.point > 0.0
    assert interval.excludes_zero is True
    assert interval.low > 0.0


def test_pairing_is_not_the_same_as_two_independent_marginals():
    """Why the pairing exists, measured on the fixture.

    Because the two cells share test photographs and are positively correlated,
    ``Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)`` is strictly smaller than the
    independent case assumes. Destroying the pairing -- shuffling one scorer's
    replicate array against the other's -- widens the difference interval,
    which is exactly the understatement the marginal-overlap reading commits.
    """
    frame = make_frame(seed=22, n_photographs=200, n_ood=400, scorers=("a", "b"))
    frame["b"] = frame["a"] + np.random.default_rng(0).normal(0.0, 0.6, len(frame))
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=3, n_replicates=500, metrics=("auroc",))

    paired = result.difference("a", "b", "standard", "near_ood")
    left = result.replicates("a", "standard", "near_ood")
    right = result.replicates("b", "standard", "near_ood")
    unpaired = np.random.default_rng(1).permutation(left) - right
    low, high = percentile_interval(unpaired, CI_LEVEL)
    assert paired.width < (high - low)


def test_asking_a_marginal_interval_whether_it_excludes_zero_raises():
    """An error, not advice. The only reason to ask a marginal CI this question
    is to compare two of them by overlap."""
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=1, n_replicates=20, metrics=("auroc",))
    marginal = result.marginal("s", "standard", "near_ood")
    assert marginal.kind == "marginal"
    with pytest.raises(ValueError, match="overlap is explicitly NOT the test"):
        marginal.excludes_zero


def test_an_unknown_interval_kind_raises():
    with pytest.raises(ValueError, match="unknown interval kind"):
        Interval("x", "guess", 0.0, 0.0, 1.0, 0.95, 10)


def test_the_required_pairwise_claims_name_real_configurations():
    """Every pair in ``REQUIRED_PAIRWISE_CLAIMS`` must name configurations that
    actually shipped, or the required-claims check would raise on a complete
    run."""
    from xai_ood.methods import (
        GAUSSIAN_CONFIGURATIONS,
        KNN_CONFIGURATIONS,
        PCA_RESIDUAL_CONFIGURATIONS,
    )

    shipped = {
        c.name
        for group in (
            GAUSSIAN_CONFIGURATIONS, KNN_CONFIGURATIONS, PCA_RESIDUAL_CONFIGURATIONS
        )
        for c in group
    }
    named = {s for pair in REQUIRED_PAIRWISE_CLAIMS for s in pair}
    assert named <= shipped, sorted(named - shipped)
    # The four 2x2 cells give six pairs; both normalization ablations; both
    # PCA-residual components against the naive cell.
    assert len(REQUIRED_PAIRWISE_CLAIMS) == 10
    assert ("rmd_pp", "rmd") in REQUIRED_PAIRWISE_CLAIMS
    assert ("knn_normalized", "knn_unnormalized") in REQUIRED_PAIRWISE_CLAIMS


# --------------------------------------------------------------------------- #
# 7. Degradation
# --------------------------------------------------------------------------- #


def test_degradation_is_paired_across_protocols_from_the_same_drawn_ids():
    """With exact-duplicate cs-ID scores the two legs are the same number in
    every replicate, so the paired degradation is identically zero and its
    interval is [0, 0] -- not merely centred on zero.

    Two independent bootstraps of the two legs could not produce that: they
    would give a spread of differences around zero. This is the cross-protocol
    pairing as an equality, and it is the same structure the non-shrinkage guard
    rests on.
    """
    frame = make_frame(seed=1, n_photographs=100, n_ood=200)
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=1, n_replicates=200, metrics=("auroc",))
    interval = result.degradation("s", "near_ood")
    assert (interval.point, interval.low, interval.high) == (0.0, 0.0, 0.0)
    assert np.array_equal(
        result.degradation_replicates("s", "near_ood"), np.zeros(200)
    )


def test_degradation_is_positive_when_the_corrupted_copies_are_harder():
    frame = make_frame(seed=31, n_photographs=200, n_subset=200, n_ood=400)
    # Push the corrupted rows toward the OOD side: the full-spectrum ID set
    # overlaps the OOD set more, so full-spectrum AUROC falls.
    csid = frame["split"] == CSID_SPLIT
    frame.loc[csid, "s"] = frame.loc[csid, "s"] + 1.2
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=1, n_replicates=300, metrics=("auroc",))
    interval = result.degradation("s", "near_ood")
    assert interval.point > 0.0
    assert interval.excludes_zero is True


def test_cross_scorer_degradation_is_a_difference_of_differences():
    """Cross-scorer comparisons of the degradation are themselves paired
    differences, so all four AUROCs come from one resample."""
    frame = make_frame(seed=32, n_photographs=150, n_ood=300, scorers=("robust", "fragile"))
    csid = frame["split"] == CSID_SPLIT
    frame.loc[csid, "fragile"] = frame.loc[csid, "fragile"] + 1.5
    plan = one_group(frame)
    result = run_bootstrap(frame, plan, seed=1, n_replicates=300, metrics=("auroc",))

    interval = result.degradation_difference("robust", "fragile", "near_ood")
    manual = result.degradation_replicates(
        "robust", "near_ood"
    ) - result.degradation_replicates("fragile", "near_ood")
    low, high = percentile_interval(manual, result.level)
    assert (interval.low, interval.high) == pytest.approx((low, high))
    assert interval.point < 0.0  # `robust` degrades less
    assert interval.excludes_zero is True


def test_degradation_needs_both_protocols():
    frame = make_frame(seed=1, n_photographs=20, n_subset=0, n_ood=25)
    plan = build_plan(
        frame,
        ood_groups={"near_ood": ("cifar100",)},
        protocols=("standard",),
        allow_missing_csid=True,
    )
    result = run_bootstrap(frame, plan, seed=1, n_replicates=10, metrics=("auroc",))
    with pytest.raises(ValueError, match="needs both protocols"):
        degradation_table(result)


# --------------------------------------------------------------------------- #
# 8. Tables, and the single 0-1 -> 0-100 conversion point
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_result():
    frame = make_frame(seed=41, n_photographs=60, n_subset=30, n_ood=80, jitter=0.5)
    plan = one_group(frame)
    return run_bootstrap(frame, plan, seed=1, n_replicates=100)


def test_results_table_conforms_to_the_frozen_schema(small_result):
    table = results_table(small_result, **PROVENANCE)
    assert tuple(table.columns) == RESULTS_COLUMNS
    assert len(table) == len(small_result.scorers) * 2 * 1  # protocols x groups
    assert set(table["protocol"]) == set(PROTOCOLS)
    assert set(table["split_role"]) == {"near_ood"}


def test_the_display_scale_is_applied_exactly_once(small_result):
    """The display-scale rule: ``to_display_scale`` converts at one point, where a
    metric enters the results table, and never again downstream. So a table
    cell must be exactly ``AUROC_SCALE`` times the raw interval it came from --
    a second conversion would make it 10,000 times, a missing one 1 times."""
    table = results_table(small_result, **PROVENANCE)
    row = table[
        (table["scorer"] == "s") & (table["protocol"] == "standard")
    ].iloc[0]
    interval = small_result.marginal("s", "standard", "near_ood")
    assert float(row["auroc"]) == pytest.approx(AUROC_SCALE * interval.point, rel=1e-15)
    assert float(row["auroc_ci_low"]) == pytest.approx(AUROC_SCALE * interval.low, rel=1e-15)
    assert float(row["auroc_ci_high"]) == pytest.approx(AUROC_SCALE * interval.high, rel=1e-15)
    assert float(row["fpr95"]) == pytest.approx(
        AUROC_SCALE * small_result.point[("s", "standard", "near_ood", "fpr95")],
        rel=1e-15,
    )


def test_results_table_carries_the_provenance_it_was_given(small_result):
    table = results_table(small_result, **PROVENANCE)
    assert set(table["repo_commit"]) == {PROVENANCE["repo_commit"]}
    assert set(table["openood_commit"]) == {PROVENANCE["openood_commit"]}
    assert set(table["seed"]) == {PROVENANCE["seed"]}


def test_results_table_refuses_a_group_that_is_not_a_split_role():
    frame = make_frame(seed=1, n_photographs=20, n_ood=25)
    plan = build_plan(frame, ood_groups={"made_up": ("cifar100",)})
    result = run_bootstrap(frame, plan, seed=1, n_replicates=10, metrics=("auroc",))
    with pytest.raises(ValueError, match="not a split_role"):
        results_table(result, **PROVENANCE)


def test_differences_table_defaults_to_the_required_claims_and_raises_on_a_gap(
    small_result,
):
    """A claim resting on marginal CIs alone is the failure this refusal
    prevents."""
    with pytest.raises(ValueError, match="does not have"):
        differences_table(small_result)
    table = differences_table(small_result, pairs=[("s", "s")])
    assert set(table["comparison"]) == {"s - s"}
    assert (table["delta"] == 0.0).all()
    assert not table["excludes_zero"].any()


def test_differences_table_can_cover_a_subset_when_asked_explicitly(small_result):
    table = differences_table(small_result, require_all=False)
    assert table.empty  # none of the ten required pairs is in this fixture


def test_degradation_table_attaches_the_fpr95_caveat(small_result):
    table = degradation_table(small_result)
    fpr_rows = table[table["metric"] == "fpr95"]
    auroc_rows = table[table["metric"] == "auroc"]
    assert (fpr_rows["caveat"] == FPR95_FULL_SPECTRUM_CAVEAT).all()
    assert (auroc_rows["caveat"] == "").all()
    assert "cs-ID mass" in FPR95_FULL_SPECTRUM_CAVEAT


def test_degradation_table_reports_both_legs_and_their_difference(small_result):
    table = degradation_table(small_result, metrics=("auroc",))
    row = table.iloc[0]
    assert row["degradation"] == pytest.approx(
        row["standard"] - row["full_spectrum"], rel=1e-12
    )


# --------------------------------------------------------------------------- #
# 9. Manifest
# --------------------------------------------------------------------------- #


def test_the_manifest_records_every_committed_parameter(small_result):
    fragment = bootstrap_manifest(small_result)
    assert fragment["procedure"] == "cluster_bootstrap_over_source_photographs"
    assert fragment["interval"] == "percentile"
    assert fragment["bca"] is False
    assert fragment["level"] == CI_LEVEL
    assert fragment["quantile_method"] == QUANTILE_METHOD
    assert fragment["resamples_shared_across_scorers"] is True
    assert fragment["plan"]["resampling_unit"] == "source_photograph"
    assert fragment["metric_definitions"]["fpr_threshold_recomputed_per_replicate"]
    assert "not the test" in fragment["comparisons"]


def test_the_manifest_is_json_serialisable(small_result):
    import json

    json.loads(json.dumps(bootstrap_manifest(small_result)))


def test_the_committed_parameters_are_what_the_declaration_says():
    assert B_DEFAULT == 1_000
    assert CI_LEVEL == 0.95


# --------------------------------------------------------------------------- #
# 10. Percentile intervals
# --------------------------------------------------------------------------- #


def test_percentile_interval_by_hand():
    """values 0..100 inclusive, 101 of them. numpy's linear method puts the
    2.5% point at virtual index 100 * 0.025 = 2.5 -> 2.5, and the 97.5% point
    at 97.5."""
    low, high = percentile_interval(np.arange(101.0), 0.95)
    assert (low, high) == pytest.approx((2.5, 97.5), rel=1e-12)


def test_percentile_interval_level_is_two_sided():
    low, high = percentile_interval(np.arange(101.0), 0.90)
    assert (low, high) == pytest.approx((5.0, 95.0), rel=1e-12)


def test_percentile_interval_rejects_a_bad_level():
    with pytest.raises(ValueError, match="level must be"):
        percentile_interval(np.arange(10.0), 1.0)


def test_percentile_interval_rejects_no_replicates():
    with pytest.raises(ValueError, match="no replicates"):
        percentile_interval(np.array([]))


# --------------------------------------------------------------------------- #
# 11. End to end with real scorers from two families
# --------------------------------------------------------------------------- #


def test_paired_differences_for_a_fixture_pair_of_real_scorers():
    """End to end: two shipped scorers from different
    families, dumped through the one ``score_frame``, resampled by photograph,
    and compared by a within-replicate paired difference.

    Small and synthetic -- 8 dimensions, a few hundred rows -- because what is
    being checked is that the seam holds, not that any scorer is good.
    """
    from xai_ood.methods import fit_knn, fit_scorer, score_frame

    rng = np.random.default_rng(20260826)
    dim = 8
    train = rng.normal(size=(400, dim))
    labels = rng.integers(0, 4, size=400)

    gaussian = fit_scorer("marginal_diagonal", train, labels)
    knn = fit_knn("knn_unnormalized", train, k=5)

    n_photo, n_subset, n_ood = 40, 15, 60
    records, embeddings = [], []
    for i in range(n_photo):
        image_id = f"cifar10/test/cat/{i:05d}.png"
        clean = rng.normal(size=dim)
        records.append(
            dict(split=ID_TEST_SPLIT, image_id=image_id, corruption=None, severity=None)
        )
        embeddings.append(clean)
        if i < n_subset:
            for corruption in CORRUPTIONS[:3]:
                for severity in SEVERITIES[:2]:
                    records.append(
                        dict(
                            split=CSID_SPLIT, image_id=image_id,
                            corruption=corruption, severity=severity,
                        )
                    )
                    embeddings.append(clean + rng.normal(0.0, 0.3, size=dim))
    for j in range(n_ood):
        records.append(
            dict(split="cifar100", image_id=f"cifar100/{j:05d}.png",
                 corruption=None, severity=None)
        )
        embeddings.append(rng.normal(1.0, 1.4, size=dim))

    index = pd.DataFrame(records)
    frame = score_frame(index, np.asarray(embeddings), [gaussian, knn])

    plan = build_plan(frame, ood_groups={"near_ood": ("cifar100",)})
    result = run_bootstrap(frame, plan, seed=1, n_replicates=200)

    assert set(result.scorers) == {"marginal_diagonal", "knn_unnormalized"}
    difference = result.difference(
        "marginal_diagonal", "knn_unnormalized", "standard", "near_ood"
    )
    assert difference.kind == "difference"
    assert difference.low <= difference.point <= difference.high
    assert isinstance(difference.excludes_zero, bool)

    table = differences_table(
        result, pairs=[("marginal_diagonal", "knn_unnormalized")]
    )
    assert len(table) == 2 * 1 * 2  # protocols x groups x metrics
    degradation = degradation_table(result, metrics=("auroc",))
    assert len(degradation) == 2
    results_table(result, **PROVENANCE)


def test_a_shuffled_index_raises_rather_than_planning_the_wrong_rows():
    """The guard with no symptom, and the reason a digest cannot be it.

    ``build_plan`` maps clusters to positional rows, so an index in a different
    order than the score frames yields a plan that points at the wrong rows and
    raises nothing. The digest is a sum of per-row hashes and is therefore
    identical under any permutation, which is asserted here too so that a later
    change to a permutation-sensitive digest does not leave this test passing for
    a reason it no longer holds.
    """
    from xai_ood.bootstrap.plan import _key_digest
    from xai_ood.schema import CANONICAL_SORT_KEY

    frame = make_frame(seed=3, n_photographs=20, n_subset=5, n_ood=25)
    index = frame[list(CANONICAL_SORT_KEY)]
    groups = {"near_ood": ("cifar100",)}
    shuffled = index.iloc[np.random.default_rng(0).permutation(len(index))]
    shuffled = shuffled.reset_index(drop=True)

    assert _key_digest(index) == _key_digest(shuffled)
    with pytest.raises(ValueError, match="not in canonical order"):
        build_plan(shuffled, ood_groups=groups)
