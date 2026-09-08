# tests/test_run_phase2.py
#
# The driver's call-site properties, and the two answers that are known before
# the run.
#
# Fit-on-train, the manifest's provenance and the matched-size null are all
# properties of how the library is called, so no test inside the library can
# establish them. They are established here, against a synthetic embedding cache
# written to disk in the real layout, small enough to finish in seconds.
#
# The fixture and why it is built the way it is
# ---------------------------------------------
# Coordinate 0 separates by construction: the in-distribution rows take values on
# a grid inside [-1, 1] and the out-of-distribution rows on a grid inside
# [40, 41], far enough out that every out-of-distribution Mahalanobis distance
# strictly exceeds every in-distribution one. That makes a fitted scorer's AUROC
# exactly 1.0 as an arithmetic fact rather than with high probability, which is
# what lets the display-scale assertion read `== 100.0`. The strict separation is
# asserted directly, so a fixture that stopped being strict fails as itself
# rather than as a mysterious 0.999.
#
# The remaining coordinates carry seeded noise on a decaying scale, so the
# fitted covariances are non-singular and the eigenspectrum is anisotropic enough
# that the variance rule selects a dimension below the full rank. At full rank
# the residual is identically zero and the degeneracy correlation is undefined.
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_phase2 import (  # noqa: E402
    CONTROL_COLUMN,
    CONTROL_DRAWS,
    CONTROL_MATCHES,
    RandomSubspaceControl,
    control_seed,
    main,
    repo_commit_here,
)
from xai_ood.bootstrap import (  # noqa: E402
    build_plan,
    differences_table,
    results_table,
    run_bootstrap,
)
from xai_ood.csid_imglist import csid_path  # noqa: E402
from xai_ood.methods.gaussian import CONFIGURATIONS as GAUSSIAN_CONFIGURATIONS  # noqa: E402
from xai_ood.methods.gaussian import fit_scorer  # noqa: E402
from xai_ood.methods.knn import CONFIGURATIONS as KNN_CONFIGURATIONS  # noqa: E402
from xai_ood.methods.knn import fit_knn  # noqa: E402
from xai_ood.methods.pca_residual import (  # noqa: E402
    CONFIGURATIONS as PCA_RESIDUAL_CONFIGURATIONS,
)
from xai_ood.methods.pca_residual import fit_pca_residual  # noqa: E402
from xai_ood.schema import (  # noqa: E402
    CANONICAL_SORT_KEY,
    assert_aligned,
    assert_canonical_order,
    decode_hyperparams,
)
from xai_ood.visualization.style import AUROC_SCALE  # noqa: E402

N_PHOTOGRAPHS = 60
N_CORRUPTED = 15
CORRUPTIONS = ("gaussian_noise", "defocus_blur", "fog")
SEVERITIES = (1, 3, 5)
ROWS_PER_CORRUPTED = len(CORRUPTIONS) * len(SEVERITIES)
N_OOD_PER_DATASET = 50
N_FEATURES = 8
N_TRAIN = 400
N_CLASSES = 4
CLASS_NAMES = ("airplane", "automobile", "bird", "cat")
SEED = 20260908
REPO_COMMIT = "0" * 40
OPENOOD_COMMIT = "1" * 40

#: Per-coordinate standard deviations, shared by the fitting and the evaluation
#: rows so the two live in one space. Decaying, so the variance rule selects a
#: dimension strictly below N_FEATURES.
SCALE = np.array([4.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1])

#: Where the out-of-distribution grid sits along coordinate 0. Chosen so the
#: smallest out-of-distribution distance exceeds the largest in-distribution one
#: by a wide margin under every fitted covariance, which is what makes 1.0 exact.
OOD_LOW, OOD_HIGH = 40.0, 41.0

N_SCORERS = len(GAUSSIAN_CONFIGURATIONS) + len(KNN_CONFIGURATIONS) + len(
    PCA_RESIDUAL_CONFIGURATIONS
)
SEPARATING_SCORER = "marginal_diagonal"


# --------------------------------------------------------------------------- #
# The synthetic cache
# --------------------------------------------------------------------------- #


def _block(n: int, rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    x = rng.normal(size=(n, N_FEATURES)) * SCALE
    x[:, 0] = np.linspace(low, high, n)
    return x


def _write_split(
    root: Path, name: str, paths: list[str], embeddings: np.ndarray, labels: np.ndarray
) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    np.save(directory / "cls.npy", embeddings)
    np.save(directory / "patchmean.npy", embeddings)
    np.save(directory / "labels.npy", labels)
    (directory / "filelist.txt").write_text("\n".join(paths) + "\n")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "split_name": name,
                "row_count": len(paths),
                "cls_shape": list(embeddings.shape),
                "imglist_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "openood_repo_commit": OPENOOD_COMMIT,
            },
            indent=2,
        )
    )


def _id_test_paths() -> list[str]:
    return [
        f"cifar10/test/{CLASS_NAMES[i % N_CLASSES]}/{i:04d}.png"
        for i in range(N_PHOTOGRAPHS)
    ]


def _csid_frame(id_paths: list[str]) -> pd.DataFrame:
    """The index sidecar the converter writes beside the cs-ID imglist."""
    records = [
        {
            "split": "csid",
            "image_id": image_id,
            "corruption": corruption,
            "severity": severity,
            "label": index % N_CLASSES,
            "path": csid_path(image_id, corruption, severity),
        }
        for index, image_id in enumerate(id_paths[:N_CORRUPTED])
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    ]
    frame = pd.DataFrame.from_records(records)
    return frame.sort_values(list(CANONICAL_SORT_KEY)).reset_index(drop=True)


@pytest.fixture(scope="module")
def cache(tmp_path_factory) -> dict:
    """An embedding cache and a cs-ID sidecar, in the layout the driver reads."""
    root = tmp_path_factory.mktemp("cache")
    rng = np.random.default_rng(SEED)

    train = rng.normal(size=(N_TRAIN, N_FEATURES)) * SCALE
    train_labels = np.tile(np.arange(N_CLASSES), N_TRAIN // N_CLASSES)
    train[:, 1] += train_labels * 0.5
    _write_split(
        root,
        "cifar10_train",
        [f"cifar10/train/{i:05d}.png" for i in range(N_TRAIN)],
        train,
        train_labels,
    )

    id_paths = _id_test_paths()
    _write_split(
        root,
        "cifar10_test",
        id_paths,
        _block(N_PHOTOGRAPHS, rng, -1.0, 1.0),
        np.arange(N_PHOTOGRAPHS) % N_CLASSES,
    )

    csid = _csid_frame(id_paths)
    _write_split(
        root,
        "csid",
        list(csid["path"]),
        _block(len(csid), rng, -0.99, 0.99),
        csid["label"].to_numpy(),
    )
    sidecar = root / "test_cifar10c_index.csv"
    csid.to_csv(sidecar, index=False)

    for dataset in ("cifar100", "mnist"):
        _write_split(
            root,
            dataset,
            [f"{dataset}/{i:04d}.png" for i in range(N_OOD_PER_DATASET)],
            _block(N_OOD_PER_DATASET, rng, OOD_LOW, OOD_HIGH),
            np.zeros(N_OOD_PER_DATASET, dtype=np.int64),
        )

    return {"root": root, "sidecar": sidecar, "csid": csid, "id_paths": id_paths}


def _parquet_engine_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def drive(cache: dict, results_root: Path, extra_argv: list[str]) -> dict:
    """Run the driver end to end and return its frame, its manifest and its dump.

    Where no parquet engine is installed the writer is substituted with another
    lossless binary format under the same filename, so every remaining step still
    runs and the reload check still reloads bytes that left the process. What the
    substitution leaves uncovered is pandas' own parquet serialiser, which the dry
    run covers wherever an engine is present.
    """
    captured: dict = {}
    real_writer = pd.DataFrame.to_parquet

    def writer(self, path, **kwargs):
        captured["handed_to_writer"] = self.copy()
        if _parquet_engine_available():
            real_writer(self, path, **kwargs)
            captured["reload"] = pd.read_parquet
        else:
            self.to_pickle(path)
            captured["reload"] = pd.read_pickle

    argv = [
        "--cache-root", str(cache["root"]),
        "--results-root", str(results_root),
        "--csid-index", str(cache["sidecar"]),
        "--seed", str(SEED),
        "--repo-commit", REPO_COMMIT,
        "--ood-group", "near_ood=cifar100",
        "--ood-group", "far_ood=mnist",
        *extra_argv,
    ]
    with mock.patch.object(pd.DataFrame, "to_parquet", writer):
        assert main(argv) == 0

    scores_dir = next((results_root / "runs").iterdir()) / "scores"
    manifest = json.loads((scores_dir / "run_metadata.json").read_text())
    return {
        "frame": captured["handed_to_writer"],
        "reloaded": captured["reload"](scores_dir / "per_sample_scores.parquet"),
        "manifest": manifest,
        "scores_dir": scores_dir,
    }


@pytest.fixture(scope="module")
def driven(cache, tmp_path_factory) -> dict:
    return drive(cache, tmp_path_factory.mktemp("results"), ["--run-id", "acceptance"])


# --------------------------------------------------------------------------- #
# The dump contract
# --------------------------------------------------------------------------- #


def test_the_dump_carries_one_column_per_configuration_plus_the_control(driven):
    """Establishes: the driver wires every configuration and the control into one
    dump."""
    columns = [c for c in driven["frame"].columns if c not in CANONICAL_SORT_KEY]
    expected = [c.name for c in GAUSSIAN_CONFIGURATIONS]
    expected += [c.name for c in KNN_CONFIGURATIONS]
    expected += [c.name for c in PCA_RESIDUAL_CONFIGURATIONS]
    expected += [f"{CONTROL_COLUMN}_{draw:02d}" for draw in range(CONTROL_DRAWS)]
    assert set(columns) == set(expected)
    assert len(columns) == N_SCORERS + CONTROL_DRAWS == 33


def test_the_key_columns_are_null_valued_rather_than_absent(driven):
    """Establishes: the index the driver builds emits nulls, not sentinels or
    omissions.
    """
    frame = driven["frame"]
    assert list(frame.columns[: len(CANONICAL_SORT_KEY)]) == list(CANONICAL_SORT_KEY)
    clean = frame[frame["split"] != "csid"]
    assert clean["corruption"].isna().all()
    assert clean["severity"].isna().all()
    corrupted = frame[frame["split"] == "csid"]
    assert corrupted["corruption"].notna().all()
    assert corrupted["severity"].notna().all()


def test_the_dump_is_written_in_canonical_order(driven):
    """Establishes: the order survives the write, so a reloaded frame goes to
    ``build_plan`` unchanged. ``score_frame`` already guarantees the in-memory
    order and the library pins it; what happens to it on disk it cannot know.
    """
    assert_canonical_order(driven["reloaded"], name="reloaded")


def test_the_reloaded_dump_reproduces_every_computed_score(driven):
    """Establishes: the persisted dump is bitwise what was computed, all thirty-three
    columns."""
    written, reloaded = driven["frame"], driven["reloaded"]
    assert len(written) == len(reloaded)
    columns = [c for c in written.columns if c not in CANONICAL_SORT_KEY]
    assert len(columns) == N_SCORERS + CONTROL_DRAWS
    for column in columns:
        deviation = np.abs(
            written[column].to_numpy(dtype=np.float64)
            - reloaded[column].to_numpy(dtype=np.float64)
        ).max()
        assert deviation == 0.0, (column, deviation)


def test_the_scored_row_count_is_every_row_of_every_scored_split(driven, cache):
    """Establishes: no split and no row is dropped between the cache and the dump.
    """
    expected = (
        N_PHOTOGRAPHS + len(cache["csid"]) + 2 * N_OOD_PER_DATASET
    )
    assert len(driven["frame"]) == expected
    assert driven["manifest"]["scored"]["rows"] == expected


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def test_the_manifest_carries_the_provenance_a_rerun_needs(driven):
    """Establishes: the driver passes the seed and both commits through, and hashes
    the cache and the dump.
    """
    manifest = driven["manifest"]
    assert manifest["repo_commit"] == REPO_COMMIT
    assert manifest["openood_commit"] == OPENOOD_COMMIT
    assert manifest["seed"] == SEED
    assert len(manifest["embedding_cache"]["manifest_sha256"]) == 64
    assert manifest["score_dump"]["sha256"]
    assert len(manifest["score_dump"]["columns"]) == N_SCORERS + CONTROL_DRAWS


def test_the_openood_commit_defaults_to_what_the_extraction_recorded(driven):
    """Establishes: the commit is read off the cache rather than invented."""
    recorded = driven["manifest"]["embedding_cache"]["splits"]
    assert {entry["openood_repo_commit"] for entry in recorded.values()} == {
        OPENOOD_COMMIT
    }


def test_the_manifest_records_the_train_row_count_for_every_scorer(driven):
    """Establishes: fit-on-train, which no test in the library can reach.

    Both counts are measured beside the fit, so a scorer handed the evaluation
    rows records the evaluation count here rather than the count the caller
    intended. The evaluation total differs from the fitting total in this
    fixture, which is what makes the two distinguishable.
    """
    fit = driven["manifest"]["fit_reference"]
    assert fit["split"] == "cifar10_train"
    assert fit["rows"] == N_TRAIN
    assert len(fit["per_scorer"]) == N_SCORERS
    scored_rows = driven["manifest"]["scored"]["rows"]
    assert scored_rows != N_TRAIN, "the fixture cannot tell the two counts apart"
    for name, entry in fit["per_scorer"].items():
        assert entry["fit_rows"] == N_TRAIN, name
        assert entry["fit_rows"] != scored_rows, name


def test_the_reference_set_of_every_scorer_is_read_off_the_fitted_object(driven):
    """Establishes: the second count comes off the fitted object, and its one
    exception is intended."""
    per_scorer = driven["manifest"]["fit_reference"]["per_scorer"]
    class_mean = {
        c.name for c in PCA_RESIDUAL_CONFIGURATIONS if c.subspace == "class_mean"
    }
    for name, entry in per_scorer.items():
        expected = N_CLASSES if name in class_mean else N_TRAIN
        assert entry["reference_rows"] == expected, name


def test_the_manifest_carries_a_degeneracy_report_for_every_residual_scorer(driven):
    """Establishes: rho is computed on the rows actually scored, for all five, rather
    than just enough to satisfy the required key.
    """
    degeneracy = driven["manifest"]["degeneracy"]
    assert set(degeneracy) == {c.name for c in PCA_RESIDUAL_CONFIGURATIONS}
    for name, report in degeneracy.items():
        assert -1.0 <= report["rho_centred_norm"] <= 1.0, name
        assert report["verdict"] in {"rescaled_norm_score", "subspace_structure"}


def test_the_timing_separates_fit_from_score_and_states_its_caveat(driven):
    """Establishes: the call-site wrapper covers every column and the caveat reaches
    the manifest. Nothing upstream times anything.
    """
    timing = driven["manifest"]["timing"]
    assert set(timing["fit"]) == set(driven["manifest"]["fit_reference"]["per_scorer"])
    assert set(timing["score"]) == {
        *timing["fit"],
        *(f"{CONTROL_COLUMN}_{draw:02d}" for draw in range(CONTROL_DRAWS)),
    }
    for entry in timing["score"].values():
        assert entry["microseconds_per_sample"] > 0.0
        assert entry["rows"] == driven["manifest"]["scored"]["rows"]
    assert timing["machine"]["node"]
    assert "faiss" in timing["caveat"] and "Cholesky" in timing["caveat"]


# --------------------------------------------------------------------------- #
# The matched-size null
# --------------------------------------------------------------------------- #


def test_the_control_matches_the_dimension_the_variance_rule_selected(driven):
    """Establishes: the control's dimension is the one the fitted scorer chose, not a
    constant.
    """
    control = driven["manifest"]["control"]
    matched = driven["manifest"]["scorers"][CONTROL_MATCHES]
    assert control["d"] == matched["d"]
    assert control["d"] < N_FEATURES, "at full rank the residual is identically zero"
    assert control["basis_seed"] == SEED
    assert control["d_source"] == CONTROL_MATCHES
    assert control["draws"] == CONTROL_DRAWS == 20
    assert control["columns"] == [
        f"{CONTROL_COLUMN}_{draw:02d}" for draw in range(CONTROL_DRAWS)
    ]


@pytest.fixture(scope="module")
def embeddings_in_dump_order(driven, cache) -> np.ndarray:
    """The cached embeddings gathered into the order the dump's rows are in.

    Built from the cache and the dump's key columns alone, so a column that
    parted company with its row during the canonical sort disagrees with a score
    recomputed here.
    """
    by_path: dict[str, np.ndarray] = {}
    for split in ("cifar10_test", "cifar100", "mnist", "csid"):
        paths = (cache["root"] / split / "filelist.txt").read_text().splitlines()
        by_path.update(dict(zip(paths, np.load(cache["root"] / split / "cls.npy"))))

    csid = cache["csid"]
    keyed = {
        f"{r.image_id}|{r.corruption}|{int(r.severity)}": r.path
        for r in csid.itertuples(index=False)
    }
    ordered = []
    for row in driven["frame"].itertuples(index=False):
        if row.split == "csid":
            key = f"{row.image_id}|{row.corruption}|{int(row.severity)}"
            ordered.append(by_path[keyed[key]])
        else:
            ordered.append(by_path[row.image_id])
    return np.asarray(ordered, dtype=np.float64)


def test_the_control_column_rebuilds_bitwise_from_what_the_manifest_records(
    driven, cache, embeddings_in_dump_order
):
    """Establishes: the recorded seed and dimension do reproduce the column.

    Recomputed from the cache rather than reloaded from the dump, so this closes
    the path from embeddings to disk rather than only from disk back. A score
    silently narrowed anywhere along the way shows up here and nowhere else, and
    narrowing is what the choice of a binary dump format exists to avoid.
    """
    control = driven["manifest"]["control"]
    train = np.load(cache["root"] / "cifar10_train" / "cls.npy")
    for draw, column in enumerate(control["columns"]):
        rebuilt = RandomSubspaceControl(
            column,
            train.mean(axis=0),
            control["d"],
            np.random.default_rng(control_seed(control["basis_seed"], draw)),
        )
        expected = rebuilt.score(embeddings_in_dump_order)
        dumped = driven["frame"][column].to_numpy(dtype=np.float64)
        assert np.abs(expected - dumped).max() == 0.0, column


def test_every_fitted_column_still_names_the_row_it_was_computed_from(
    driven, cache, embeddings_in_dump_order
):
    """Establishes: the sort permuted the embeddings with the keys, for all thirteen.

    The failure this exists for is silent: a permutation applied to the keys and
    not to the embeddings produces a plausible frame and wrong paired
    differences. Every check downstream of the sort inherits whatever order the
    sort produced, so crossing it from outside is the only way to see it. One
    column checked and twelve assumed is the same gap one column wider.
    """
    train = np.load(cache["root"] / "cifar10_train" / "cls.npy")
    labels = np.load(cache["root"] / "cifar10_train" / "labels.npy")

    rebuilt = {c.name: fit_scorer(c, train, labels) for c in GAUSSIAN_CONFIGURATIONS}
    rebuilt.update({c.name: fit_knn(c, train) for c in KNN_CONFIGURATIONS})
    rebuilt.update(
        {c.name: fit_pca_residual(c, train, labels) for c in PCA_RESIDUAL_CONFIGURATIONS}
    )
    assert len(rebuilt) == N_SCORERS == 13

    for name, scorer in rebuilt.items():
        deviation = np.abs(
            scorer.score(embeddings_in_dump_order)
            - driven["frame"][name].to_numpy(dtype=np.float64)
        ).max()
        assert deviation == 0.0, (name, deviation)


def test_every_draw_is_a_different_subspace(driven):
    """Establishes: the twenty columns are twenty draws, not one draw copied.

    The failure this exists for is a seeding bug that gives every draw the same
    basis, which produces twenty identical columns, a range of width zero, and a
    control that looks unanimous because it was only ever asked once.
    """
    columns = driven["manifest"]["control"]["columns"]
    values = {c: driven["frame"][c].to_numpy(dtype=np.float64) for c in columns}
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            assert not np.array_equal(values[first], values[second]), (first, second)


def test_the_draws_have_spread_and_the_matched_scorer_is_read_against_it(driven):
    """Establishes: the block supports the reading the plan declares for it.

    The declared reporting is the matched scorer's value against the minimum,
    median and maximum over the draws, which needs the draws to differ from each
    other by more than floating-point noise. A block whose spread is at the noise
    floor would make every landing look like a tie.
    """
    columns = driven["manifest"]["control"]["columns"]
    stacked = np.stack(
        [driven["frame"][c].to_numpy(dtype=np.float64) for c in columns]
    )
    per_row_spread = stacked.max(axis=0) - stacked.min(axis=0)
    assert per_row_spread.min() > 0.0
    assert np.median(per_row_spread) > 1e-9
    assert driven["frame"][CONTROL_MATCHES].to_numpy(dtype=np.float64).shape == (
        stacked.shape[1],
    )


def test_the_control_seed_derivation_is_per_draw(driven):
    """Establishes: the manifest's recorded rule is the rule the code used.

    Recorded as a string, so it is prose until something asserts it. Two draws
    seeded from the same run seed under this rule must not collide.
    """
    assert driven["manifest"]["control"]["seed_derivation"] == (
        "numpy default_rng([basis_seed, draw_index])"
    )
    assert control_seed(SEED, 0) != control_seed(SEED, 1)
    first = np.random.default_rng(control_seed(SEED, 0)).normal(size=8)
    second = np.random.default_rng(control_seed(SEED, 1)).normal(size=8)
    assert not np.array_equal(first, second)


def test_the_control_basis_is_orthonormal():
    """Establishes: the residual is a projection residual.

    ``score`` subtracts ``B B^T`` applied to the centred vector, which is a
    projector only for an orthonormal ``B``. The bitwise rebuild above cannot see
    this, because it would construct the same wrong basis.
    """
    control = RandomSubspaceControl(
        "c", np.zeros(N_FEATURES), 3, np.random.default_rng(SEED)
    )
    assert np.allclose(control.basis.T @ control.basis, np.eye(3))


# --------------------------------------------------------------------------- #
# The shrinkage arms
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm,shrunk", [("scored_arm", False), ("unscored_arm", True)])
def test_both_shrinkage_arms_record_intensity_and_condition_number(driven, arm, shrunk):
    """Establishes: both arms were fitted, and each fragment carries its own flag.

    The two arms are asserted to *be* two arms. Fitting the same arm twice would
    fill both fragments with numbers that pass every other check here.
    """
    entries = driven["manifest"]["shrinkage_diagnostics"][arm]
    assert set(entries) == {c.name for c in GAUSSIAN_CONFIGURATIONS}
    for name, covariances in entries.items():
        assert covariances, name
        for fit in covariances.values():
            assert fit["shrinkage"] is shrunk, (arm, name)
            assert fit["condition_number"] > 0.0
            assert fit["shrinkage_intensity"] >= 0.0
            assert fit["n_samples"] == N_TRAIN


def test_the_two_arms_are_not_the_same_fit_under_two_names(driven):
    """Establishes: the two arms are two fits, not one fit labelled twice.

    Fitting one arm twice would fill both fragments and pass every other check."""
    scored = driven["manifest"]["shrinkage_diagnostics"]["scored_arm"]
    unscored = driven["manifest"]["shrinkage_diagnostics"]["unscored_arm"]
    differing = [
        name
        for name in scored
        for key in scored[name]
        if scored[name][key]["condition_number"] != unscored[name][key]["condition_number"]
    ]
    assert differing, "every covariance is identically conditioned in both arms"


def test_the_scored_arm_is_readable_off_the_per_scorer_hyperparams(driven):
    """Establishes: which arm a dump holds is recoverable with no convention invented
    here."""
    for config in GAUSSIAN_CONFIGURATIONS:
        hyperparams = decode_hyperparams(
            driven["manifest"]["scorers"][config.name]["hyperparams"]
        )
        assert hyperparams["shrinkage"] is False, config.name
        assert hyperparams["shrinkage_target"] is None, config.name


def test_the_shrunk_arm_is_distinguishable_in_a_dump_with_the_same_columns(
    cache, tmp_path_factory
):
    """Establishes: the only exercise of --shrinkage, end to end."""
    shrunk = drive(
        cache,
        tmp_path_factory.mktemp("results_shrunk"),
        ["--run-id", "shrunk", "--shrinkage"],
    )
    for config in GAUSSIAN_CONFIGURATIONS:
        hyperparams = decode_hyperparams(
            shrunk["manifest"]["scorers"][config.name]["hyperparams"]
        )
        assert hyperparams["shrinkage"] is True, config.name
        assert hyperparams["shrinkage_target"], config.name
    intensities = [
        fit["shrinkage_intensity"]
        for covariances in shrunk["manifest"]["shrinkage_diagnostics"][
            "scored_arm"
        ].values()
        for fit in covariances.values()
    ]
    assert max(intensities) > 0.0, "a shrunk arm that selected no intensity anywhere"


# --------------------------------------------------------------------------- #
# The dry run
# --------------------------------------------------------------------------- #


def test_a_dry_run_writes_its_own_directory_and_says_so_in_the_manifest(
    cache, tmp_path_factory
):
    """Establishes: a truncated run cannot be mistaken for a full one."""
    results = tmp_path_factory.mktemp("results_dry")
    dry = drive(cache, results, ["--run-id", "gate", "--dry-run", "--dry-run-rows", "100"])
    assert dry["scores_dir"].parent.name == "gate-dryrun"
    assert dry["manifest"]["dry_run"] is True
    assert dry["manifest"]["dry_run_rows"] == 100
    columns = [c for c in dry["frame"].columns if c not in CANONICAL_SORT_KEY]
    assert len(columns) == N_SCORERS + CONTROL_DRAWS
    corrupted = dry["frame"]["split"] == "csid"
    assert int(corrupted.sum()) == 100 < len(cache["csid"])
    assert dry["manifest"]["fit_reference"]["rows"] == 100
    assert_canonical_order(dry["frame"], name="dry")


def test_a_dry_run_keeps_every_corrupted_row_joined_to_a_clean_one(cache, tmp_path_factory):
    """Establishes: the row cap does not orphan corrupted rows from their photographs."""
    dry = drive(
        cache,
        tmp_path_factory.mktemp("results_dry_join"),
        ["--run-id", "join", "--dry-run", "--dry-run-rows", "100"],
    )
    frame = dry["frame"]
    clean = set(frame.loc[frame["split"] == "id_test", "image_id"])
    corrupted = set(frame.loc[frame["split"] == "csid", "image_id"])
    assert corrupted <= clean
    build_plan(frame, ood_groups={"near_ood": ("cifar100",), "far_ood": ("mnist",)})


# --------------------------------------------------------------------------- #
# The two answers that are known before the run
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bootstrapped(driven) -> dict:
    """The reloaded dump through the plan and the bootstrap, plus a duplicate column.

    The duplicate is a second frame rather than a second column of the first,
    because one resample index applied to two separately-carried arrays is what
    the paired difference rests on, and a single frame cannot exercise it.
    """
    frame = driven["reloaded"]
    duplicate = frame.loc[:, list(CANONICAL_SORT_KEY)].copy()
    duplicate[f"{SEPARATING_SCORER}_copy"] = frame[SEPARATING_SCORER].to_numpy()
    assert_aligned([frame, duplicate], names=["dump", "duplicate"])

    plan = build_plan(
        frame, ood_groups={"near_ood": ("cifar100",), "far_ood": ("mnist",)}
    )
    result = run_bootstrap(
        {"dump": frame, "duplicate": duplicate}, plan, seed=SEED, n_replicates=200
    )
    return {"frame": frame, "duplicate": duplicate, "plan": plan, "result": result}


def test_the_bootstrap_carries_every_column_the_dump_holds(bootstrapped):
    """Establishes: widening the control block to twenty draws does not lose columns
    downstream.

    ``_score_columns`` takes whatever non-key columns a frame carries, so thirty-three
    should pass as readily as fourteen did. Should is what the driver review found
    wanting elsewhere, so it is asserted here: every dumped column reaches the
    bootstrap result, and the control draws are neither dropped nor silently merged.
    """
    scored = {c for c in bootstrapped["frame"].columns if c not in CANONICAL_SORT_KEY}
    reached = {key[0] for key in bootstrapped["result"].point}
    assert scored <= reached, sorted(scored - reached)
    draws = {c for c in scored if c.startswith(CONTROL_COLUMN)}
    assert len(draws) == CONTROL_DRAWS
    assert draws <= reached


def test_the_fixture_separates_strictly_so_the_known_answer_is_arithmetic(driven):
    """Establishes: nothing about the driver. It is a fixture invariant, and it is
    what makes the two answers below arithmetic rather than probable."""
    frame = driven["frame"]
    ood = frame.loc[frame["split"].isin({"cifar100", "mnist"}), SEPARATING_SCORER]
    within = frame.loc[~frame["split"].isin({"cifar100", "mnist"}), SEPARATING_SCORER]
    assert ood.min() > within.max()


@pytest.mark.parametrize("protocol", ["standard", "full_spectrum"])
@pytest.mark.parametrize("group", ["near_ood", "far_ood"])
def test_a_strictly_separating_scorer_gives_auroc_exactly_one(
    bootstrapped, protocol, group
):
    """Establishes: the driver's persisted dump reaches build_plan and the bootstrap
    unchanged, in both protocols and both groups. Also the first required
    acceptance number."""
    result = bootstrapped["result"]
    assert result.point[(SEPARATING_SCORER, protocol, group, "auroc")] == 1.0
    assert (result.replicates(SEPARATING_SCORER, protocol, group, "auroc") == 1.0).all()


def test_the_separating_scorer_reads_exactly_one_hundred_in_the_table(bootstrapped):
    """Establishes: the second half of the first acceptance number. The display-scale
    conversion itself is pinned upstream; this is kept because the number has to be
    reported."""
    table = results_table(
        bootstrapped["result"],
        seed=SEED,
        repo_commit=REPO_COMMIT,
        openood_commit=OPENOOD_COMMIT,
        timestamp="2026-09-08T00:00:00Z",
    )
    rows = table[table["scorer"] == SEPARATING_SCORER]
    assert len(rows) == 4
    for column in ("auroc", "auroc_ci_low", "auroc_ci_high"):
        assert (rows[column] == 100.0).all(), rows[column].tolist()
    assert AUROC_SCALE == 100.0


def test_two_identical_columns_give_a_difference_interval_containing_zero(bootstrapped):
    """Establishes: the dump aligns with a separately carried frame under one resample
    index. Also the second required acceptance number.

    One column carried in two frames under two names must give exactly zero in
    every replicate, which it can only do if both saw the same resample index.
    """
    result = bootstrapped["result"]
    for protocol in result.protocols:
        for group in result.ood_groups:
            for metric in result.metrics:
                interval = result.difference(
                    SEPARATING_SCORER, f"{SEPARATING_SCORER}_copy", protocol, group, metric
                )
                assert interval.point == 0.0
                assert interval.low == 0.0 and interval.high == 0.0
                assert interval.excludes_zero is False


# --------------------------------------------------------------------------- #
# Invocation guards
# --------------------------------------------------------------------------- #


def test_the_commit_of_the_repository_holding_the_script_is_readable():
    """Establishes: the only exercise of the derived-commit branch, since every other
    test passes --repo-commit.
    """
    commit = repo_commit_here()
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)


def test_a_missing_corrupted_split_stops_the_run_unless_it_is_waived(
    cache, tmp_path_factory
):
    """Establishes: the driver refuses rather than silently scoring one protocol."""
    results = tmp_path_factory.mktemp("results_nocsid")
    argv = [
        "--cache-root", str(cache["root"]),
        "--results-root", str(results),
        "--run-id", "nocsid",
        "--seed", str(SEED),
        "--repo-commit", REPO_COMMIT,
        "--csid-cache-split", "absent",
    ]
    with pytest.raises(SystemExit, match="absent"):
        main(argv)


def test_an_index_naming_a_row_the_cache_lacks_is_refused(cache, tmp_path_factory):
    """Establishes: the path-keyed lookup raises rather than misaligning."""
    results = tmp_path_factory.mktemp("results_badindex")
    broken = results / "broken_index.csv"
    frame = cache["csid"].copy()
    frame.loc[0, "path"] = "cifar10c/nowhere/1/cat/0000.png"
    frame.to_csv(broken, index=False)
    argv = [
        "--cache-root", str(cache["root"]),
        "--results-root", str(results),
        "--csid-index", str(broken),
        "--run-id", "broken",
        "--seed", str(SEED),
        "--repo-commit", REPO_COMMIT,
        "--ood-group", "near_ood=cifar100",
        "--ood-group", "far_ood=mnist",
    ]
    with pytest.raises(ValueError, match="filelist.txt"):
        main(argv)
