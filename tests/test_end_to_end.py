# tests/test_end_to_end.py
#
# **The only test that crosses module seams.**
#
# The per-module tests cover their own units well; nothing else runs embeddings
# through the whole path:
#
#   fit -> Scorer.score -> score_frame -> assert_aligned -> build_plan
#        -> run_bootstrap -> results_table / differences_table / degradation_table
#
# Every seam in that chain was checked only by runtime assertions during the real
# run, which is the worst possible place to discover a misalignment: mid-run,
# against a cache that took hours to build, with no cheap way to bisect. This
# file runs the whole path on a synthetic dataset small enough to finish in a
# second and constructed so that the answers are known
# exactly rather than approximately.
#
# The fixture, and why it is built the way it is
# ----------------------------------------------
# Coordinate 0 is a **separating coordinate by construction**: ID rows take
# values on a grid inside [-1, 1] and OOD rows on a grid inside [10, 11], so
# every OOD value is strictly greater than every ID value. That makes
# `PerfectScorer`'s AUROC exactly 1.0 as an arithmetic fact, not with high
# probability -- which is what lets the display-scale assertion read `== 100.0`
# rather than `approx(100.0)`. The remaining coordinates carry seeded noise so
# the fitted covariances are non-singular and the real Gaussian path exercises
# the same seams.
#
# Cluster structure mirrors the real procedure in miniature: a minority of
# ID-test photographs carry corrupted copies and the rest carry one row each, so
# `cluster_size_range` under full-spectrum is genuinely non-uniform, exactly as
# it will be at 2,000-of-9,000.
import numpy as np
import pandas as pd
import pytest

from xai_ood.bootstrap import (
    CI_LEVEL,
    CSID_SPLIT,
    ID_TEST_SPLIT,
    build_plan,
    degradation_table,
    differences_table,
    results_table,
    run_bootstrap,
)
from xai_ood.methods.dump import score_frame
from xai_ood.methods.gaussian import fit_all_scorers
from xai_ood.methods.knn import fit_knn
from xai_ood.schema import RESULTS_COLUMNS, assert_aligned, validate_results_frame
from xai_ood.visualization.style import AUROC_SCALE

# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

N_PHOTOGRAPHS = 60  # ID-test photographs; the bootstrap's cluster universe
N_CORRUPTED = 15  # of those, how many carry cs-ID rows
CORRUPTIONS = ("gaussian_noise", "defocus_blur", "fog")
SEVERITIES = (1, 3, 5)
ROWS_PER_CORRUPTED = len(CORRUPTIONS) * len(SEVERITIES)  # 9
N_OOD_PER_DATASET = 50
N_FEATURES = 8
N_TRAIN = 400

#: Two datasets, one per group. Group *names* must be `split_role` values from
#: the frozen schema, because `results_table` writes them into that column;
#: dataset names are free.
OOD_GROUPS = {"near_ood": ("cifar100",), "far_ood": ("mnist",)}


def _separating_values(n: int, low: float, high: float) -> np.ndarray:
    """``n`` distinct values on a closed grid in ``[low, high]``."""
    return np.linspace(low, high, n)


def synthetic_evaluation() -> tuple[pd.DataFrame, np.ndarray]:
    """A canonical-key index and its embeddings, row for row, unsorted.

    Deliberately returned **unsorted** (cs-ID rows appended after all clean ID
    rows, OOD after that), so that `score_frame`'s canonical sort has real work
    to do and the permutation it applies to the embeddings is exercised rather
    than being the identity.
    """
    rng = np.random.default_rng(20260831)
    keys: list[dict[str, object]] = []
    first: list[float] = []

    id_ids = [f"id_{i:03d}" for i in range(N_PHOTOGRAPHS)]
    id_first = _separating_values(N_PHOTOGRAPHS, -1.0, 1.0)
    for image_id, value in zip(id_ids, id_first):
        keys.append(
            {
                "split": ID_TEST_SPLIT,
                "image_id": image_id,
                "corruption": None,
                "severity": None,
            }
        )
        first.append(float(value))

    # cs-ID: corrupted copies of the FIRST N_CORRUPTED photographs, keyed by the
    # same image_id. That join is the load-bearing one -- if the two sides
    # disagreed about what an image_id is, `build_cluster_index` would raise.
    csid_first = _separating_values(N_CORRUPTED * ROWS_PER_CORRUPTED, -0.99, 0.99)
    cursor = 0
    for image_id in id_ids[:N_CORRUPTED]:
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                keys.append(
                    {
                        "split": CSID_SPLIT,
                        "image_id": image_id,
                        "corruption": corruption,
                        "severity": severity,
                    }
                )
                first.append(float(csid_first[cursor]))
                cursor += 1

    for dataset in ("cifar100", "mnist"):
        ood_first = _separating_values(N_OOD_PER_DATASET, 10.0, 11.0)
        for i, value in enumerate(ood_first):
            keys.append(
                {
                    "split": dataset,
                    "image_id": f"{dataset}_{i:03d}",
                    "corruption": None,
                    "severity": None,
                }
            )
            first.append(float(value))

    index = pd.DataFrame(keys)
    embeddings = np.empty((len(index), N_FEATURES), dtype=np.float64)
    embeddings[:, 0] = np.asarray(first)
    embeddings[:, 1:] = rng.normal(size=(len(index), N_FEATURES - 1))
    return index, embeddings


def train_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """ID *train* embeddings and labels. Fit on train, score test, always."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(N_TRAIN, N_FEATURES))
    labels = np.tile(np.arange(4), N_TRAIN // 4)
    # Give the classes distinct means so the class-conditional cells are fitting
    # something rather than four copies of the marginal model.
    x[:, 1] += labels * 0.5
    return x, labels


class ConstantColumnScorer:
    """Scores by a single embedding coordinate. Higher = more OOD, as required.

    On the fixture above, coordinate 0 separates OOD from ID perfectly and
    strictly, so this scorer's AUROC is exactly 1.0.
    """

    def __init__(self, name: str, column: int = 0) -> None:
        self.name = name
        self.column = column

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)[:, self.column]


# --------------------------------------------------------------------------- #
# The whole path, once
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pipeline():
    """Run the entire chain once and hand every stage's output to the tests."""
    index, embeddings = synthetic_evaluation()
    x_train, labels = train_embeddings()

    gaussians = fit_all_scorers(x_train, labels)
    knn = fit_knn("knn_unnormalized", x_train, k=5)
    perfect_a = ConstantColumnScorer("perfect_a")
    perfect_b = ConstantColumnScorer("perfect_b")

    # Two frames rather than one, on purpose: `assert_aligned` is the check that
    # a shared resample index is meaningful across separately-produced dumps,
    # and it cannot be exercised by a single frame.
    frame_gaussian = score_frame(
        index, embeddings, {**gaussians, knn.name: knn}, name="gaussian+knn"
    )
    frame_probe = score_frame(
        index, embeddings, [perfect_a, perfect_b], name="probes"
    )
    assert_aligned([frame_gaussian, frame_probe], names=["gaussian+knn", "probes"])

    plan = build_plan(frame_gaussian, ood_groups=OOD_GROUPS)
    result = run_bootstrap(
        {"gaussian+knn": frame_gaussian, "probes": frame_probe},
        plan,
        seed=20260831,
        n_replicates=200,
    )
    return {
        "index": index,
        "embeddings": embeddings,
        "frames": (frame_gaussian, frame_probe),
        "plan": plan,
        "result": result,
    }


# --------------------------------------------------------------------------- #
# Seam 1: score_frame and the canonical order
# --------------------------------------------------------------------------- #


def test_the_two_dumps_cover_the_same_rows_in_the_same_order(pipeline):
    """If this ever fails, every paired difference in the thesis is meaningless."""
    a, b = pipeline["frames"]
    assert len(a) == len(b) == len(pipeline["index"])
    assert_aligned([a, b], names=["a", "b"])


def test_scores_follow_their_rows_through_the_canonical_sort(pipeline):
    """The permutation is applied to the embeddings, not just to the keys.

    `perfect_a` returns coordinate 0 verbatim, so its column must equal the
    coordinate-0 value of the row its key names -- looked up in the *unsorted*
    input. A sort that moved keys without moving embeddings would still produce
    a plausible-looking frame.
    """
    index, embeddings = pipeline["index"], pipeline["embeddings"]
    _, probes = pipeline["frames"]

    def key(row) -> tuple:
        # Nulls arrive as None on the object column and as NaN on the numeric
        # one, and they differ again between the input frame and the dump, so
        # both sides are normalised rather than compared as they come.
        corruption = None if pd.isna(row.corruption) else str(row.corruption)
        severity = None if pd.isna(row.severity) else int(row.severity)
        return (str(row.split), str(row.image_id), corruption, severity)

    lookup = {
        key(row): embeddings[i, 0]
        for i, row in enumerate(index.itertuples(index=False))
    }
    assert len(lookup) == len(index), "the fixture's keys are not unique"
    for row in probes.itertuples(index=False):
        assert row.perfect_a == lookup[key(row)]


# --------------------------------------------------------------------------- #
# Seam 2: build_plan and the cluster structure
# --------------------------------------------------------------------------- #


def test_the_plan_resamples_photographs_and_not_rows(pipeline):
    diagnostics = pipeline["plan"].diagnostics
    assert diagnostics["resampling_unit"] == "source_photograph"
    assert diagnostics["n_id_clusters"] == N_PHOTOGRAPHS
    assert diagnostics["n_csid_photographs"] == N_CORRUPTED
    assert diagnostics["n_csid_rows"] == N_CORRUPTED * ROWS_PER_CORRUPTED
    # Non-uniform under full-spectrum, uniform under standard. This is the
    # real shape in miniature, and it is what a row-level bootstrap would
    # flatten to [1, 1] on both legs.
    assert diagnostics["cluster_size_range"]["standard"] == [1, 1]
    assert diagnostics["cluster_size_range"]["full_spectrum"] == [
        1,
        1 + ROWS_PER_CORRUPTED,
    ]


# --------------------------------------------------------------------------- #
# Seam 3: run_bootstrap, and the two known answers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("protocol", ["standard", "full_spectrum"])
@pytest.mark.parametrize("group", ["near_ood", "far_ood"])
def test_a_perfectly_separating_scorer_gives_auroc_exactly_one(
    pipeline, protocol, group
):
    """Exactly 1.0, and 1.0 in every replicate, so the interval is degenerate.

    Not `approx`: coordinate 0 separates strictly by construction, and AUROC is
    a sum of integer win counts divided by their total, so the value is exact in
    floating point.
    """
    result = pipeline["result"]
    assert result.point[("perfect_a", protocol, group, "auroc")] == 1.0
    replicates = result.replicates("perfect_a", protocol, group, "auroc")
    assert (replicates == 1.0).all()


@pytest.mark.parametrize("protocol", ["standard", "full_spectrum"])
def test_a_perfectly_separating_scorer_gives_fpr95_exactly_zero(pipeline, protocol):
    """The other metric's known answer, at the same operating point."""
    result = pipeline["result"]
    assert result.point[("perfect_a", protocol, "near_ood", "fpr95")] == 0.0


def test_two_identical_scorers_give_a_difference_interval_containing_zero(pipeline):
    """The paired-difference machinery's null case.

    `perfect_a` and `perfect_b` compute the same function under different names,
    so within every replicate their metrics are identical and the difference is
    exactly zero. Anything else means the two scorers saw different resamples,
    which is precisely the failure the loop order exists to prevent: one index
    set per replicate, applied to every scorer.
    """
    result = pipeline["result"]
    for protocol in result.protocols:
        for group in result.ood_groups:
            for metric in result.metrics:
                interval = result.difference(
                    "perfect_a", "perfect_b", protocol, group, metric
                )
                assert interval.kind == "difference"
                assert interval.point == 0.0
                assert interval.low == 0.0 and interval.high == 0.0
                assert interval.excludes_zero is False
                assert (
                    result.replicates("perfect_a", protocol, group, metric)
                    == result.replicates("perfect_b", protocol, group, metric)
                ).all()


def test_every_scorer_in_both_frames_reached_the_result(pipeline):
    """Two input frames, one result. A dropped column would be silent."""
    names = set(pipeline["result"].scorers)
    assert {"perfect_a", "perfect_b", "knn_unnormalized"} <= names
    assert {"marginal_diagonal", "marginal_full", "rmd", "rmd_pp"} <= names
    assert len(names) == 8 + 1 + 2  # eight Gaussian cells, one kNN, two probes


def test_a_real_fitted_scorer_survives_the_whole_path(pipeline):
    """Not just the synthetic probes: a scorer fitted on train, end to end.

    The OOD rows sit ten standard deviations out in coordinate 0, so the four
    absolute-distance cells and kNN should separate them essentially perfectly.
    The point is that a fitted scorer produces a usable number through six
    modules, not that this fixture has any particular AUROC.
    """
    result = pipeline["result"]
    for scorer in (
        "marginal_diagonal",
        "marginal_full",
        "class_conditional_diagonal",
        "class_conditional_full",
        "knn_unnormalized",
    ):
        value = result.point[(scorer, "standard", "far_ood", "auroc")]
        assert 0.9 < value <= 1.0, (scorer, value)


def test_the_relative_cells_run_but_are_not_asserted_to_separate(pipeline):
    """RMD is a *relative* score and this fixture is the case it subtracts away.

    The shift is a pure translation along one coordinate, which raises the
    class-conditional and the marginal Mahalanobis terms together, so RMD's
    difference of the two carries little of it and its AUROC here is low. That
    is the scorer behaving as designed on a fixture built for the absolute
    cells, not a defect -- so what is asserted is that the value is a
    well-formed AUROC that reached the result, and nothing about its size.
    """
    result = pipeline["result"]
    for scorer in ("rmd", "rmd_pp"):
        value = result.point[(scorer, "standard", "far_ood", "auroc")]
        assert 0.0 <= value <= 1.0 and np.isfinite(value), (scorer, value)


# --------------------------------------------------------------------------- #
# Seam 4: the tables, and the single display-scale conversion
# --------------------------------------------------------------------------- #


def test_the_results_table_conforms_to_the_frozen_schema(pipeline):
    table = results_table(
        pipeline["result"],
        seed=20260831,
        repo_commit="0" * 40,
        openood_commit="1" * 40,
        timestamp="2026-08-31T00:00:00Z",
    )
    assert tuple(table.columns) == RESULTS_COLUMNS
    validate_results_frame(table)
    expected_rows = len(pipeline["result"].scorers) * 2 * 2  # protocols x groups
    assert len(table) == expected_rows


def test_the_perfect_scorer_reads_exactly_one_hundred_in_the_table(pipeline):
    """The display-scale conversion happens exactly once.

    A second conversion would give 10,000, which `validate_results_frame`
    catches; a *missing* one would give 1.0, which its zero-to-one heuristic
    only catches when every value is in that range. An exact `== 100.0` on a
    known-exact AUROC catches both directions.
    """
    table = results_table(
        pipeline["result"],
        seed=20260831,
        repo_commit="0" * 40,
        openood_commit="1" * 40,
        timestamp="2026-08-31T00:00:00Z",
    )
    rows = table[table["scorer"] == "perfect_a"]
    assert len(rows) == 4
    for column in ("auroc", "auroc_ci_low", "auroc_ci_high"):
        assert (rows[column] == 100.0).all(), rows[column].tolist()
    assert (rows["fpr95"] == 0.0).all()


def test_a_table_cell_is_exactly_the_scale_times_the_raw_interval(pipeline):
    """Pins the conversion factor itself, on a scorer whose value is not 1.0."""
    result = pipeline["result"]
    table = results_table(
        result,
        seed=20260831,
        repo_commit="0" * 40,
        openood_commit="1" * 40,
        timestamp="2026-08-31T00:00:00Z",
    )
    row = table[
        (table["scorer"] == "marginal_diagonal")
        & (table["protocol"] == "standard")
        & (table["split_role"] == "near_ood")
    ].iloc[0]
    interval = result.marginal("marginal_diagonal", "standard", "near_ood", "auroc")
    assert float(row["auroc"]) == AUROC_SCALE * interval.point
    assert float(row["auroc_ci_low"]) == AUROC_SCALE * interval.low
    assert float(row["auroc_ci_high"]) == AUROC_SCALE * interval.high


def test_the_differences_table_reaches_the_probes(pipeline):
    """`require_all=False` because this fixture has no PCA-residual scorers.

    The default is `True` and stays that way: a real run must produce a row for
    every required claim or raise.
    """
    table = differences_table(
        pipeline["result"],
        pairs=[("perfect_a", "perfect_b"), ("marginal_full", "marginal_diagonal")],
        require_all=False,
    )
    assert len(table) == 2 * 2 * 2 * 2  # pairs x protocols x groups x metrics
    null_rows = table[table["comparison"] == "perfect_a - perfect_b"]
    assert (null_rows["delta"] == 0.0).all()
    assert (~null_rows["excludes_zero"]).all()


def test_the_required_claims_raise_when_the_run_lacks_a_scorer(pipeline):
    """The guard that keeps a claim from being reported unsupported."""
    with pytest.raises(ValueError, match="paired differences were requested"):
        differences_table(pipeline["result"])


def test_the_degradation_table_pairs_both_protocols(pipeline):
    table = degradation_table(pipeline["result"], scorers=["perfect_a"])
    assert len(table) == 2 * 2  # groups x metrics
    # A perfect scorer degrades by exactly nothing: 100.0 on both legs.
    auroc_rows = table[table["metric"] == "auroc"]
    assert (auroc_rows["standard"] == 100.0).all()
    assert (auroc_rows["full_spectrum"] == 100.0).all()
    assert (auroc_rows["degradation"] == 0.0).all()


def test_the_run_reproduces_bit_for_bit_from_the_recorded_seed(pipeline):
    """The manifest records a seed and claims it reproduces the pairing.

    Checked here across the whole path rather than on the sampler alone, so it
    covers the scorers and the frames as well as `replicate_indices`.
    """
    frame_gaussian, frame_probe = pipeline["frames"]
    again = run_bootstrap(
        {"gaussian+knn": frame_gaussian, "probes": frame_probe},
        pipeline["plan"],
        seed=20260831,
        n_replicates=200,
    )
    original = pipeline["result"]
    for key, values in original.values.items():
        assert np.array_equal(values, again.values[key]), key
    assert original.level == again.level == CI_LEVEL
