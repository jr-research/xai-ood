"""Fit every OOD scorer on CIFAR-10 train, score every evaluation split, and
persist the per-sample scores and a run manifest.

Reads a DINOv2 embedding cache laid out one directory per split, each holding
``cls.npy``, ``patchmean.npy``, ``labels.npy``, ``filelist.txt`` and
``manifest.json``. Reads the cs-ID index sidecar written beside the cs-ID
imglist, which already carries the four canonical key columns and each row's
path, so the corrupted-copy grid is never rebuilt here.

Writes two files into ``<results-root>/runs/<run-id>/scores/``:
``per_sample_scores.parquet``, holding the four key columns and one column per
scorer in canonical row order, and ``run_metadata.json`` beside it. Parquet
rather than a text format because the scores are float64 and a text round trip
changes them.

Thirty-five score columns, not fifteen. The extra twenty are
``control_random_subspace_residual_00`` through ``_19``: the residual to a random
subspace of the same dimension the variance rule selected, drawn twenty times.
They answer a question the residual scorers cannot answer alone, which is whether
the discarded directions had to be the leading ones, or whether discarding any
that many leaves as informative a norm. Sweeping the dimension does not answer it
either, since every point on that sweep is variance-ordered.

Twenty draws rather than one, which is the whole reason this is a block of
columns and not a column. One draw says only whether the variance-ordered
subspace beats *that* subspace, and it carries no spread, so a residual landing
near it is uninterpretable: a real finding and an unlucky basis look identical.
Twenty draws give a range to read the observed value against. The columns need
the embeddings, so none of this can be recovered from the score dump afterwards,
which is why the count is fixed before the run rather than after it.

Shrinkage is a fit-time argument and not part of a configuration's name, so both
arms produce identically named columns and one invocation scores one arm. They
stay distinguishable without any convention invented here:
``GaussianConfig.encoded_hyperparams`` emits ``"shrinkage"`` and
``"shrinkage_target"``, and the manifest carries that encoding for every scorer.
Both arms are fitted regardless, and their condition numbers and selected
intensities are recorded, because that measurement is what settles whether
shrinkage is inert at this sample-to-dimension ratio and it is unavailable
afterwards without refitting.

``--dry-run`` runs the identical path over the first rows of each split and
writes to a separate directory, so the write reaches disk once at a size that
costs a minute before it is asked to do so at full scale.

    python3 run_phase2.py --cache-root DIR --results-root DIR \\
        --run-id NAME --seed INT --csid-index FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np
import pandas as pd

from xai_ood.bootstrap.plan import CSID_SPLIT, DEFAULT_OOD_GROUPS, ID_TEST_SPLIT
from xai_ood.methods.dump import score_frame
from xai_ood.methods.gaussian import CONFIGURATIONS as GAUSSIAN_CONFIGURATIONS
from xai_ood.methods.gaussian import fit_scorer
from xai_ood.methods.knn import CONFIGURATIONS as KNN_CONFIGURATIONS
from xai_ood.methods.knn import DEFAULT_K, REFERENCE_SPLIT, fit_knn
from xai_ood.methods.manifest import write_run_manifest
from xai_ood.methods.pca_residual import CONFIGURATIONS as PCA_RESIDUAL_CONFIGURATIONS
from xai_ood.methods.pca_residual import degeneracy_report, fit_pca_residual
from xai_ood.schema import CANONICAL_SORT_KEY, assert_canonical_order

#: Cache directory names for the two CIFAR-10 splits. The cache is keyed by the
#: source split name, while the canonical ``split`` value of the clean
#: evaluation rows is ``ID_TEST_SPLIT``, which is a different string.
TRAIN_CACHE_SPLIT: str = REFERENCE_SPLIT
ID_TEST_CACHE_SPLIT: str = "cifar10_test"

#: Cache directory name for the corrupted copies.
CSID_CACHE_SPLIT: str = CSID_SPLIT

#: The two embedding fields every split directory carries.
EMBEDDING_FIELDS: tuple[str, ...] = ("cls", "patchmean")

#: Stem of the matched-size null columns. Prefixed so a reader of the dump reads
#: them as a control on the residual scorers rather than as further scorers. Each
#: draw appends its zero-padded index, so the columns sort into draw order.
CONTROL_COLUMN: str = "control_random_subspace_residual"

#: Draws of the matched-size null. Twenty, not one: a single draw cannot say
#: whether it was typical, so a residual landing near it cannot be told from an
#: unlucky basis. Fixed before the run because the draws need the embeddings and
#: no arithmetic on the dumped scores recovers them. ``--control-draws`` can
#: lower it, but only on a timing measurement taken before any AUROC is computed,
#: and the value used is recorded in the manifest either way.
CONTROL_DRAWS: int = 20

#: The configuration whose selected dimension the control matches, and whose
#: comparison against the naive marginal-diagonal baseline the control qualifies.
CONTROL_MATCHES: str = "pca_residual_all_id"

#: Rows per split under ``--dry-run``. The cap applies to the fitting split as
#: well as to the scored ones, so it sits above the 768-dimensional embedding
#: width: below that the full covariances are singular and a dry run fails for a
#: reason a full run never would.
DRY_RUN_ROWS: int = 2_000

#: Rows the PCA-residual degeneracy control is measured on. The control asks
#: whether the class-mean residual is a rescaled embedding norm, which is a
#: question about the in-distribution geometry the subspace was fitted to
#: describe. Both the residual and the norm grow on an out-of-distribution row,
#: so including those raises rho for a reason unrelated to what the threshold
#: asks. Every set is computed and recorded; this one carries the verdict.
DEGENERACY_ROW_SET: str = "id_test_and_csid"

TIMING_CAVEAT: str = (
    "Timings measure this implementation as much as they measure the methods. "
    "Nearest-neighbour search is exact numpy rather than faiss, and the "
    "covariance factorisations use numpy's Cholesky rather than scipy's. Any "
    "comparison against the nearest-neighbour scorers is implementation-inclusive."
)


# --------------------------------------------------------------------------- #
# The embedding cache
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CachedSplit:
    """One split of the embedding cache: paths, embeddings, labels, manifest."""

    def __init__(
        self,
        name: str,
        paths: list[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        manifest: dict[str, Any],
        manifest_sha256: str,
    ) -> None:
        self.name = name
        self.paths = paths
        self.embeddings = embeddings
        self.labels = labels
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.position = {path: i for i, path in enumerate(paths)}
        if len(self.position) != len(paths):
            raise ValueError(
                f"{name}: filelist.txt repeats a path, so a row cannot be "
                f"identified by one. Every key downstream is that path."
            )

    def rows_for(self, wanted: Sequence[str]) -> np.ndarray:
        """Embedding rows for ``wanted``, in that order.

        Looked up by path rather than by position, so a filelist written in a
        different order from the index still aligns instead of silently pairing
        each score with a different image.
        """
        try:
            positions = [self.position[path] for path in wanted]
        except KeyError as exc:
            raise ValueError(
                f"{self.name}: {exc.args[0]!r} is named by the index but is not "
                f"in filelist.txt. The cache and the index describe different rows."
            ) from None
        return self.embeddings[positions]


def load_split(
    cache_root: Path, name: str, field: str, *, limit: int | None = None
) -> CachedSplit:
    """Read one split directory and check its four files agree on row count."""
    directory = cache_root / name
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    lines = (directory / "filelist.txt").read_text().splitlines()
    paths = [line for line in lines if line]
    embeddings = np.load(directory / f"{field}.npy")
    labels = np.load(directory / "labels.npy")

    counts = {
        "filelist.txt": len(paths),
        f"{field}.npy": int(embeddings.shape[0]),
        "labels.npy": int(labels.shape[0]),
        "manifest.row_count": int(manifest["row_count"]),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"{name}: row counts disagree: {counts}")
    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{name}: {int((~np.isfinite(embeddings)).sum())} non-finite value(s) "
            f"in {field}.npy"
        )

    if limit is not None and limit < len(paths):
        paths = paths[:limit]
        embeddings = embeddings[:limit]
        labels = labels[:limit]

    return CachedSplit(
        name,
        paths,
        np.asarray(embeddings, dtype=np.float64),
        np.asarray(labels),
        manifest,
        sha256_file(manifest_path),
    )


def cache_fragment(
    splits: dict[str, CachedSplit], root: Path, field: str
) -> dict[str, Any]:
    """Manifest fragment identifying the cache these scores came from."""
    per_split = {
        name: {
            "rows": len(split.paths),
            "manifest_sha256": split.manifest_sha256,
            "imglist_sha256": split.manifest.get("imglist_sha256"),
            "openood_repo_commit": split.manifest.get("openood_repo_commit"),
        }
        for name, split in splits.items()
    }
    combined = hashlib.sha256()
    for name in sorted(per_split):
        combined.update(f"{name}:{per_split[name]['manifest_sha256']}\n".encode())
    return {
        "root": str(root),
        "field": field,
        "manifest_sha256": combined.hexdigest(),
        "splits": per_split,
    }


def openood_commit_from_cache(splits: dict[str, CachedSplit]) -> str:
    """The OpenOOD commit the extraction recorded, if every split agrees on one."""
    recorded = {
        name: split.manifest.get("openood_repo_commit") for name, split in splits.items()
    }
    missing = sorted(name for name, value in recorded.items() if not value)
    if missing:
        raise ValueError(
            f"the extraction manifest(s) for {missing} record no OpenOOD commit. "
            f"Pass --openood-commit to state the one this run claims."
        )
    distinct = sorted(set(recorded.values()))
    if len(distinct) != 1:
        raise ValueError(
            f"the splits were extracted against different OpenOOD commits "
            f"{distinct}. Pass --openood-commit to state which one this run claims."
        )
    return distinct[0]


def repo_commit_here() -> str:
    """Commit of the repository holding this script."""
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"cannot read the commit of {root}: {result.stderr.strip()}. "
            f"Pass --repo-commit."
        )
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# The evaluation index
# --------------------------------------------------------------------------- #


def clean_keys(split_value: str, paths: Sequence[str]) -> pd.DataFrame:
    """Canonical key columns for rows that carry no corruption."""
    return pd.DataFrame(
        {
            "split": split_value,
            "image_id": list(paths),
            "corruption": None,
            "severity": None,
        }
    )


def csid_keys(
    index_path: Path, id_test_paths: Sequence[str], limit: int | None
) -> pd.DataFrame:
    """Canonical key columns and paths for the corrupted copies, from the sidecar.

    Under a row limit the sidecar is first restricted to photographs that
    survived the same limit on the clean split, because a corrupted row whose
    ``image_id`` names no clean row is a cluster the resampling plan refuses.
    """
    sidecar = pd.read_csv(index_path)
    for column in (*CANONICAL_SORT_KEY, "path"):
        if column not in sidecar.columns:
            raise ValueError(f"{index_path}: missing the {column!r} column")
    if limit is not None:
        survivors = sidecar[sidecar["image_id"].isin(set(id_test_paths))]
        if len(sidecar) and not len(survivors):
            raise ValueError(
                f"{index_path}: the row limit left no corrupted rows joined to a "
                f"clean one, so this run would score none of the covariate-shift "
                f"leg and still report success. The limit is applied to the clean "
                f"split and to this sidecar separately, and the two windows "
                f"coincide only when both are ordered by image_id. Widen the "
                f"limit, or check filelist.txt row order against the sidecar."
            )
        sidecar = survivors.iloc[:limit]
    return sidecar.reset_index(drop=True)


def evaluation_index(
    splits: dict[str, CachedSplit],
    *,
    id_test_cache_split: str,
    csid_cache_split: str | None,
    csid_index: Path | None,
    ood_splits: Sequence[str],
    limit: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """One key frame and one embedding matrix over every scored split.

    Returned unsorted, because ``score_frame`` imposes the canonical order and
    permutes the embeddings with it.
    """
    frames: list[pd.DataFrame] = []
    blocks: list[np.ndarray] = []

    def add(cache_name: str, keys: pd.DataFrame, paths: Sequence[str]) -> None:
        frames.append(keys.loc[:, list(CANONICAL_SORT_KEY)])
        blocks.append(splits[cache_name].rows_for(paths))

    id_test_paths = splits[id_test_cache_split].paths
    add(
        id_test_cache_split,
        clean_keys(ID_TEST_SPLIT, id_test_paths),
        id_test_paths,
    )

    if csid_cache_split is not None:
        sidecar = csid_keys(csid_index, id_test_paths, limit)
        add(csid_cache_split, sidecar, list(sidecar["path"]))

    for name in ood_splits:
        add(name, clean_keys(name, splits[name].paths), splits[name].paths)

    return pd.concat(frames, ignore_index=True), np.concatenate(blocks, axis=0)


# --------------------------------------------------------------------------- #
# Fitting, and the matched-size null
# --------------------------------------------------------------------------- #


def control_seed(seed: int, draw: int) -> list[int]:
    """Entropy for one draw, derived from the run seed and the draw index.

    A list rather than an integer so each draw gets its own stream from the one
    run seed: ``default_rng([seed, draw])`` is independent across ``draw`` and
    reproducible without replaying the draws before it. The manifest records this
    rule and the seed, which is what makes a column rebuildable from the manifest
    alone.
    """
    return [int(seed), int(draw)]


def control_column_name(draw: int) -> str:
    """The dump column for one draw, zero-padded so the columns sort in draw order."""
    return f"{CONTROL_COLUMN}_{draw:02d}"


def build_controls(
    matched: Any, seed: int, draws: int
) -> list["RandomSubspaceControl"]:
    """One control per draw, all at the matched scorer's centre and dimension.

    The draws differ only in their basis. Anything else that differed would make
    the spread across them a measurement of that difference instead of a
    measurement of how much the variance ordering was worth.
    """
    if draws < 1:
        raise ValueError(f"phase2: --control-draws must be at least 1, got {draws}")
    return [
        RandomSubspaceControl(
            control_column_name(draw),
            matched.mean,
            matched.d,
            np.random.default_rng(control_seed(seed, draw)),
        )
        for draw in range(draws)
    ]


class RandomSubspaceControl:
    """Residual to a random subspace of the dimension the variance rule selected.

    Same centre and same dimension as the scorer it matches, so the only
    difference between one of these columns and the scorer it matches is whether
    the discarded directions were chosen by variance or at random.
    """

    def __init__(
        self, name: str, mean: np.ndarray, d: int, rng: np.random.Generator
    ) -> None:
        self.name = name
        self.mean = np.asarray(mean, dtype=np.float64)
        self.d = int(d)
        basis, _ = np.linalg.qr(rng.normal(size=(self.mean.shape[0], self.d)))
        self.basis = basis

    def score(self, x: np.ndarray) -> np.ndarray:
        """Higher = more OOD, matching every other score column."""
        centred = np.asarray(x, dtype=np.float64) - self.mean
        residual = centred - (centred @ self.basis) @ self.basis.T
        return np.sqrt(np.einsum("ij,ij->i", residual, residual))


class TimedScorer:
    """Records the time a scorer spends scoring, without altering what it returns."""

    def __init__(self, scorer: Any) -> None:
        self.scorer = scorer
        self.name = scorer.name
        self.seconds = 0.0
        self.rows = 0

    def score(self, x: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        values = self.scorer.score(x)
        self.seconds += time.perf_counter() - start
        self.rows += int(np.asarray(x).shape[0])
        return values


def fit_gaussian_arm(
    embeddings: np.ndarray, labels: np.ndarray, *, shrinkage: bool
) -> tuple[dict[str, Any], dict[str, float], dict[str, dict[str, int]]]:
    """Fit the eight Gaussian configurations on one arm, timing and measuring each."""
    scorers: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    rows: dict[str, dict[str, int]] = {}
    for config in GAUSSIAN_CONFIGURATIONS:
        start = time.perf_counter()
        scorer = fit_scorer(config, embeddings, labels, shrinkage=shrinkage)
        seconds[config.name] = time.perf_counter() - start
        scorers[config.name] = scorer
        rows[config.name] = {
            "fit_rows": int(embeddings.shape[0]),
            "reference_rows": int(scorer.primary.diagnostics["n_samples"]),
        }
    return scorers, seconds, rows


def fit_scorers(
    embeddings: np.ndarray, labels: np.ndarray, *, shrinkage: bool
) -> tuple[dict[str, Any], dict[str, float], dict[str, dict[str, int]]]:
    """Fit all fifteen configurations; return them, their seconds and their row counts.

    Loops the per-configuration fit functions rather than the ``fit_all_*``
    wrappers, so each fit can be timed separately without touching the library.

    Both row counts are taken **here**, beside the call that fits: ``fit_rows``
    off the array being handed over and ``reference_rows`` off the fitted object.
    Re-deriving either where the manifest is assembled would record what the
    caller believes it fitted rather than what it fitted, which is the one thing
    these numbers exist to rule out.
    """
    scorers, seconds, rows = fit_gaussian_arm(embeddings, labels, shrinkage=shrinkage)
    for config in KNN_CONFIGURATIONS:
        start = time.perf_counter()
        scorer = fit_knn(config, embeddings)
        seconds[config.name] = time.perf_counter() - start
        scorers[config.name] = scorer
        rows[config.name] = {
            "fit_rows": int(embeddings.shape[0]),
            "reference_rows": int(scorer.reference.shape[0]),
        }
    for config in PCA_RESIDUAL_CONFIGURATIONS:
        start = time.perf_counter()
        scorer = fit_pca_residual(config, embeddings, labels)
        seconds[config.name] = time.perf_counter() - start
        scorers[config.name] = scorer
        # The class-mean subspace is fitted to the K class means rather than to
        # the rows they were averaged from, so its reference_rows is K while its
        # fit_rows is the split. Both are reported; neither is the other.
        rows[config.name] = {
            "fit_rows": int(embeddings.shape[0]),
            "reference_rows": int(scorer.subspace.n_fitting_rows),
        }
    return scorers, seconds, rows


def arm_diagnostics(scorers: dict[str, Any]) -> dict[str, Any]:
    """Condition number and selected intensity for every covariance in one arm."""
    out: dict[str, Any] = {}
    for config in GAUSSIAN_CONFIGURATIONS:
        scorer = scorers[config.name]
        fits = [scorer.primary] + ([scorer.background] if scorer.background else [])
        out[config.name] = {
            fit.name: {
                "condition_number": fit.diagnostics["condition_number"],
                "shrinkage": fit.diagnostics["shrinkage"],
                "shrinkage_intensity": fit.diagnostics["shrinkage_intensity"],
                "n_samples": fit.diagnostics["n_samples"],
                "samples_per_dimension": fit.diagnostics["samples_per_dimension"],
            }
            for fit in fits
        }
    return out


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def run(
    options: argparse.Namespace, log: TextIO = sys.stdout
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load, fit, score and assemble. Returns the frame, the manifest extra and the scorers.

    Resolves ``--repo-commit`` and ``--openood-commit`` onto ``options`` when
    they were not given, since the second is read off the cache this loads.
    """

    def say(message: str) -> None:
        print(message, file=log, flush=True)

    limit = options.dry_run_rows if options.dry_run else None
    ood_splits = [name for group in options.ood_groups.values() for name in group]
    wanted = [options.train_split, options.id_test_cache_split, *ood_splits]
    if options.csid_cache_split is not None:
        wanted.append(options.csid_cache_split)

    say(f"[load] cache {options.cache_root}, field {options.embedding_field}")
    splits = {
        name: load_split(options.cache_root, name, options.embedding_field, limit=limit)
        for name in wanted
    }
    for name in wanted:
        say(f"[load] {name:<18}{len(splits[name].paths):>9} rows")

    if options.repo_commit is None:
        options.repo_commit = repo_commit_here()
    if options.openood_commit is None:
        options.openood_commit = openood_commit_from_cache(splits)

    train = splits[options.train_split]
    index, embeddings = evaluation_index(
        splits,
        id_test_cache_split=options.id_test_cache_split,
        csid_cache_split=options.csid_cache_split,
        csid_index=options.csid_index,
        ood_splits=ood_splits,
        limit=limit,
    )
    say(f"[index] {len(index)} evaluation rows over {index['split'].nunique()} splits")

    say(
        f"[fit] {len(train.paths)} rows from {options.train_split}, "
        f"shrinkage={options.shrinkage}"
    )
    scorers, fit_seconds, fit_rows = fit_scorers(
        train.embeddings, train.labels, shrinkage=options.shrinkage
    )
    unscored_arm, _, _ = fit_gaussian_arm(
        train.embeddings, train.labels, shrinkage=not options.shrinkage
    )
    for name in scorers:
        say(
            f"[fit] {name:<34}{fit_seconds[name]:>8.2f} s   "
            f"{fit_rows[name]['fit_rows']} rows fitted, "
            f"{fit_rows[name]['reference_rows']} in the reference set"
        )

    matched = scorers[CONTROL_MATCHES]
    controls = build_controls(matched, options.seed, options.control_draws)
    say(
        f"[control] {len(controls)} draw(s) of {CONTROL_COLUMN} at "
        f"d = {controls[0].d}, matching {CONTROL_MATCHES}, "
        f"seeded [{options.seed}, draw]"
    )

    timed = [TimedScorer(s) for s in scorers.values()]
    timed += [TimedScorer(control) for control in controls]
    frame = assert_canonical_order(
        score_frame(index, embeddings, timed, name="phase2"), name="phase2"
    )
    for wrapper in timed:
        say(
            f"[score] {wrapper.name:<34}{wrapper.seconds:>8.2f} s   "
            f"{1e6 * wrapper.seconds / max(wrapper.rows, 1):>8.2f} us/sample"
        )

    nonfinite = {
        column: int((~np.isfinite(frame[column].to_numpy(dtype=np.float64))).sum())
        for column in frame.columns
        if column not in CANONICAL_SORT_KEY
    }
    nonfinite = {column: count for column, count in nonfinite.items() if count}
    if nonfinite:
        raise ValueError(
            f"phase2: non-finite score(s) in {len(nonfinite)} column(s): "
            f"{nonfinite}. The embeddings are checked on load and the scores were "
            f"not, so this stops the run rather than dumping a column that would "
            f"reach AUROC without raising."
        )
    say(f"[finite] all {len(frame.columns) - len(CANONICAL_SORT_KEY)} score columns finite")

    scored = [c for c in frame.columns if c not in CANONICAL_SORT_KEY]
    expected_columns = [w.name for w in timed]
    if scored != expected_columns:
        raise ValueError(
            f"phase2: the dump carries {len(scored)} score column(s) and "
            f"{len(expected_columns)} were scored. Missing "
            f"{sorted(set(expected_columns) - set(scored))}, unexpected "
            f"{sorted(set(scored) - set(expected_columns))}. The count is checked "
            f"rather than assumed because a control draw that silently failed to "
            f"reach the frame would leave every other check passing."
        )
    say(
        f"[columns] {len(scored)} score columns: {len(scorers)} scorers plus "
        f"{len(controls)} control draw(s)"
    )

    degeneracy_row_sets = {
        DEGENERACY_ROW_SET: index["split"].isin((ID_TEST_SPLIT, CSID_SPLIT)).to_numpy(),
        "id_test": (index["split"] == ID_TEST_SPLIT).to_numpy(),
        "all_evaluation": np.ones(len(index), dtype=bool),
    }
    degeneracy: dict[str, Any] = {}
    for config in PCA_RESIDUAL_CONFIGURATIONS:
        scorer = scorers[config.name]
        by_row_set = {
            name: degeneracy_report(scorer, embeddings[mask])
            for name, mask in degeneracy_row_sets.items()
        }
        by_row_set["id_train"] = degeneracy_report(scorer, train.embeddings)
        report = dict(by_row_set[DEGENERACY_ROW_SET])
        report["row_set"] = DEGENERACY_ROW_SET
        report["row_sets"] = by_row_set
        degeneracy[config.name] = report
        say(
            f"[degeneracy] {config.name:<34}"
            f"rho_centred={report['rho_centred_norm']:.4f}  {report['verdict']}"
            f"  on {DEGENERACY_ROW_SET}"
        )
        say(
            "             "
            + "  ".join(
                f"{name}={by_row_set[name]['rho_centred_norm']:.4f}"
                for name in by_row_set
            )
        )

    extra = {
        "run_id": options.run_id,
        "dry_run": bool(options.dry_run),
        "dry_run_rows": limit,
        "embedding_cache": cache_fragment(
            splits, options.cache_root, options.embedding_field
        ),
        "fit_reference": {
            "split": options.train_split,
            "rows": len(train.paths),
            "per_scorer": {
                name: {"fit_split": options.train_split, **fit_rows[name]}
                for name in scorers
            },
        },
        "scored": {
            "rows": len(frame),
            "splits": {str(k): int(v) for k, v in frame["split"].value_counts().items()},
            "ood_groups": {k: list(v) for k, v in options.ood_groups.items()},
        },
        "timing": {
            "machine": {
                "node": platform.node(),
                "platform": platform.platform(),
                "processor": platform.processor() or platform.machine(),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "fit": {
                name: {
                    "seconds": fit_seconds[name],
                    "microseconds_per_sample": 1e6 * fit_seconds[name] / len(train.paths),
                }
                for name in scorers
            },
            "score": {
                w.name: {
                    "seconds": w.seconds,
                    "rows": w.rows,
                    "microseconds_per_sample": 1e6 * w.seconds / max(w.rows, 1),
                }
                for w in timed
            },
            "caveat": TIMING_CAVEAT,
        },
        "degeneracy": degeneracy,
        "control": {
            "columns": [c.name for c in controls],
            "draws": len(controls),
            "draws_default": CONTROL_DRAWS,
            "basis_seed": options.seed,
            "seed_derivation": "numpy default_rng([basis_seed, draw_index])",
            "d": controls[0].d,
            "d_source": CONTROL_MATCHES,
            "n_residual_directions": int(controls[0].mean.shape[0] - controls[0].d),
            "construction": (
                "orthonormal basis from the QR factorisation of a seeded Gaussian "
                "matrix, residual taken about the same centre as the scorer it matches"
            ),
            "read_as": (
                "a range, not a test: the matched scorer's value is reported against "
                "the minimum, median and maximum over the draws, with no rank "
                "statistic and no p-value"
            ),
        },
        "shrinkage_diagnostics": {
            "scored_arm": arm_diagnostics(scorers),
            "unscored_arm": arm_diagnostics(unscored_arm),
        },
    }
    return frame, extra, scorers


def write_outputs(
    frame: pd.DataFrame,
    extra: dict[str, Any],
    scorers: dict[str, Any],
    options: argparse.Namespace,
    log: TextIO = sys.stdout,
) -> Path:
    """Write the score dump and the manifest beside it, and return their directory."""

    def say(message: str) -> None:
        print(message, file=log, flush=True)

    run_id = f"{options.run_id}-dryrun" if options.dry_run else options.run_id
    directory = options.results_root / "runs" / run_id / "scores"
    directory.mkdir(parents=True, exist_ok=True)
    scores_path = directory / "per_sample_scores.parquet"
    frame.to_parquet(scores_path, index=False)

    extra = dict(extra)
    extra["score_dump"] = {
        "path": str(scores_path),
        "rows": len(frame),
        "columns": [c for c in frame.columns if c not in CANONICAL_SORT_KEY],
        "sha256": sha256_file(scores_path),
    }
    manifest_path = directory / "run_metadata.json"
    write_run_manifest(
        manifest_path,
        scorers,
        seed=options.seed,
        repo_commit=options.repo_commit,
        openood_commit=options.openood_commit,
        extra=extra,
    )

    say(f"[write] {scores_path}")
    say(
        f"[write] {len(frame)} rows, {len(extra['score_dump']['columns'])} score "
        f"columns, sha256 {extra['score_dump']['sha256'][:12]}"
    )
    say(f"[write] {manifest_path}")
    say(
        "[next] in run_metadata.json, check that fit_reference.rows is the train "
        "count and not the test count, that scored.splits matches the cache row "
        "counts above, that every scorers.*.hyperparams records the shrinkage arm "
        "this dump holds, and that degeneracy.*.row_set names the set the verdict "
        "was taken on, with the other sets recorded beside it."
    )
    return directory


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #


def parse_ood_group(value: str) -> tuple[str, tuple[str, ...]]:
    """Parse one ``NAME=split,split`` argument."""
    name, _, members = value.partition("=")
    if not name or not members:
        raise argparse.ArgumentTypeError(f"expected NAME=split,split, got {value!r}")
    return name, tuple(m for m in members.split(",") if m)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the run options. Paths are arguments here, never literals."""
    parser = argparse.ArgumentParser(description="Score every evaluation split.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--control-draws",
        type=int,
        default=CONTROL_DRAWS,
        help=(
            "draws of the matched-size null (default %(default)s). Lower it only on "
            "a timing measurement taken before any AUROC is computed; the value used "
            "is recorded in the manifest either way"
        ),
    )
    parser.add_argument("--repo-commit")
    parser.add_argument("--openood-commit")
    parser.add_argument("--embedding-field", choices=EMBEDDING_FIELDS, default="cls")
    parser.add_argument("--train-split", default=TRAIN_CACHE_SPLIT)
    parser.add_argument("--id-test-cache-split", default=ID_TEST_CACHE_SPLIT)
    parser.add_argument("--csid-cache-split", default=CSID_CACHE_SPLIT)
    parser.add_argument("--csid-index", type=Path)
    parser.add_argument("--allow-missing-csid", action="store_true")
    parser.add_argument(
        "--ood-group", type=parse_ood_group, action="append", dest="ood_group_list"
    )
    parser.add_argument("--shrinkage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-rows", type=int, default=DRY_RUN_ROWS)

    options = parser.parse_args(argv)
    options.ood_groups = (
        dict(options.ood_group_list)
        if options.ood_group_list
        else {k: tuple(v) for k, v in DEFAULT_OOD_GROUPS.items()}
    )
    if options.dry_run and options.dry_run_rows < DEFAULT_K:
        raise SystemExit(
            f"--dry-run-rows {options.dry_run_rows} caps the fitting split too, "
            f"and the nearest-neighbour scorers need at least k = {DEFAULT_K} "
            f"reference rows. The covariance fits additionally need more rows "
            f"than the embeddings have dimensions."
        )
    if options.allow_missing_csid:
        options.csid_cache_split = None
    elif not (options.cache_root / options.csid_cache_split).is_dir():
        raise SystemExit(
            f"no {options.csid_cache_split!r} directory under {options.cache_root}. "
            f"The full-spectrum protocol has no in-distribution rows without it. "
            f"Pass --allow-missing-csid to score the standard protocol alone."
        )
    elif options.csid_index is None or not options.csid_index.is_file():
        raise SystemExit(
            "--csid-index must name the index sidecar written beside the cs-ID "
            "imglist. Its four key columns are what make a corrupted row join the "
            "photograph it came from, and no other file carries them."
        )
    return options


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    frame, extra, scorers = run(options)
    write_outputs(frame, extra, scorers, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
