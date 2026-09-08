"""Builder for the CIFAR-10-C cs-ID imglist.

**The one expensive-to-undo artefact here.** Everything else is cheap
to rebuild; the 2,000-image draw is not, because once embeddings have been
extracted against a particular set of ids, a different set means re-extracting.
So the draw is fixed here, with a recorded seed, before the data has ever been
seen.

What the declared procedure commits to, and what this module therefore enforces:

    One shared set of 2,000 image ids, drawn once from the 9,000 OpenOOD
    ID-test ids with a recorded seed, and reused in **all 75** (corruption,
    severity) cells. Not 75 independent draws.

That single sentence is what makes the severity curve a within-image comparison
-- the reason CIFAR-10-C is used rather than CINIC-10 -- and what makes the
cluster bootstrap well defined: 2,000 photographs own 76 rows each (1 clean +
15 corruptions x 5 severities), the other 7,000 own 1 row each, and the
bootstrap draws photographs and carries whatever rows each one owns.

The four properties are asserted on the write path, not described in a comment:
exactly 75 cells, exactly 2,000 rows per cell, the *same* 2,000 ids in every
cell (set equality across all 75 -- the assertion that catches a per-cell
redraw), and the draw reproducing from the recorded seed.

Row identity
------------
``image_id`` is **the ID-test imglist path the corrupted image came from**, e.g.
``cifar10/test/cat/3975.png``. Not the corrupted path, and not the filename
stem. Two reasons, both load-bearing:

* A photograph's clean row and its 75 corrupted rows must share an ``image_id``,
  or the cluster bootstrap cannot carry them together and the severity curve is
  not paired.
* The path is unique by construction. A bare stem would only be unique if
  OpenOOD numbers test images globally rather than per class, which is not
  something this module is in a position to check.

**The clean ID-test rows must use the same convention**, i.e. take
``image_id`` straight from the cached ``filelist.txt`` line. If the clean dump
uses stems and this uses paths, nothing raises -- the two just never join, and
every cs-ID photograph silently becomes a singleton cluster.

The one assumption this module invents
--------------------------------------
**Where the corrupted images live on disk.** ``DEFAULT_PATH_TEMPLATE`` is a
guess, flagged as one, and it is the only guess here.

It is deliberately isolated in a template string so that being wrong about it
costs one argument at the call site and nothing else. In particular,
if the download turns out to ship Hendrycks & Dietterich's original ``.npy``
arrays rather than per-image files, the ``path`` column is the only thing that
changes: the 2,000 drawn ids, the 75 cells, the row order and the manifest are
all unaffected, and rebuilding from the recorded seed reproduces the identical
draw. Verify the layout against the real tree before extracting (pass
``data_root=`` to ``write_csid_imglist`` and it will check every path).

Row order
---------
The emitted imglist is in canonical order, ``(split, image_id, corruption,
severity)`` per ``xai_ood.schema``, so the extraction cache's ``filelist.txt``
comes out canonically ordered too and the index sidecar aligns with the score
arrays row for row with no reindexing. That order is image-major, which is the
worst possible access pattern for an extractor reading from per-corruption
arrays. An extractor that would rather go cell-major may -- it just has to write
its own ``filelist.txt`` in the order it actually used and run the index frame
through ``xai_ood.schema.canonical_sort`` afterwards.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .schema import VALID_SEVERITIES, assert_canonical_order, canonical_sort

__all__ = [
    "CORRUPTIONS",
    "HELD_OUT_CORRUPTIONS",
    "SEVERITIES",
    "SUBSET_SEED",
    "N_ID_TEST",
    "N_SUBSET",
    "N_CELLS",
    "N_ROWS",
    "ROWS_PER_DRAWN_IMAGE",
    "CSID_SPLIT",
    "DEFAULT_PATH_TEMPLATE",
    "IMGLIST_COLUMNS",
    "full_spectrum_mixture_weight",
    "parse_imglist",
    "read_id_test_imglist",
    "draw_shared_subset",
    "csid_path",
    "build_csid_imglist",
    "assert_csid_grid",
    "assert_draw_reproduces",
    "imglist_text",
    "sha256_text",
    "sha256_file",
    "count_imglist_lines",
    "extraction_volume_report",
    "csid_manifest",
    "write_csid_imglist",
]


# --------------------------------------------------------------------------- #
# The grid. None of these is re-derivable; all were decided upstream.
# --------------------------------------------------------------------------- #

#: The **standard 15**, in Hendrycks & Dietterich's own grouping order (noise,
#: blur, weather, digital). CIFAR-10-C ships 19; the other four are distributed
#: as a held-out validation set for tuning robustness methods and are not part
#: of the standard benchmark. This thesis commits to the 15.
#: 15 x 5 x 10,000 = 750,000 is the standard CIFAR-10-C grid,
#: 19 x 5 x 10,000 = 950,000 is the whole distributed dataset. Quoting the
#: wrong one at the wrong scope is an easy mistake to make.
CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

#: The four this thesis does **not** use. Named rather than merely omitted, so
#: that "why isn't spatter in there" has an answer in the source instead of only
#: in a dataset note.
HELD_OUT_CORRUPTIONS: tuple[str, ...] = (
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
)

#: The five severities, from ``xai_ood.schema`` so there is one definition.
SEVERITIES: tuple[int, ...] = VALID_SEVERITIES

#: **The recorded seed.** 2026-08-26, the date the draw was fixed.
#:
#: The date is not decoration. It is evidence that the seed was chosen before
#: the CIFAR-10-C data was on disk and before any scorer had produced a number,
#: so it cannot have been selected against a result. A seed picked after seeing
#: scores is a garden of forking paths; this one could not have been.
SUBSET_SEED: int = 20260826

#: OpenOOD's CIFAR-10 test split. **9,000, not 10,000** -- 1,000 of the original
#: test set is reserved as ``val``. Confirmed against measured row counts, not
#: assumed: using 10,000 here changes every derived count below it.
N_ID_TEST: int = 9_000

#: Photographs drawn once and reused in every cell.
N_SUBSET: int = 2_000

#: 15 corruptions x 5 severities.
N_CELLS: int = 75

#: 75 cells x 2,000 rows.
N_ROWS: int = 150_000

#: Rows a *drawn* photograph owns under the full-spectrum protocol: 1 clean +
#: 75 corrupted. The remaining 7,000 ID-test photographs own 1 row each. Cluster
#: sizes are non-uniform by construction and the bootstrap needs no correction
#: for it, since it resamples ids and carries whatever rows each id owns.
ROWS_PER_DRAWN_IMAGE: int = 76

#: ``split_role`` for these rows. Enumerated in ``xai_ood.schema.SPLIT_ROLES``.
CSID_SPLIT: str = "csid"

#: **The invented assumption.** See the module docstring. Fields available:
#: ``{corruption}``, ``{severity}``, ``{class_name}``, ``{filename}``,
#: ``{stem}``, ``{id_path}``.
DEFAULT_PATH_TEMPLATE: str = "cifar10c/{corruption}/{severity}/{class_name}/{filename}"

#: Columns of the emitted index frame, in order. The first four are the canonical
#: sort key; ``label`` and ``path`` are what the extractor needs.
IMGLIST_COLUMNS: tuple[str, ...] = (
    "split",
    "image_id",
    "corruption",
    "severity",
    "label",
    "path",
)


def full_spectrum_mixture_weight(
    n_id_test: int = N_ID_TEST, n_csid: int = N_ROWS
) -> float:
    """The full-spectrum ID mixture weight, ``w = n_id / (n_id + n_csid)``.

    Computed rather than written down as 0.057, because the number that gets
    quoted should follow from the counts rather than sit beside them. At these
    sizes it is 9,000 / 159,000.
    """
    total = n_id_test + n_csid
    if total <= 0:
        raise ValueError("n_id_test + n_csid must be positive")
    return n_id_test / total


# --------------------------------------------------------------------------- #
# Reading OpenOOD's imglists
# --------------------------------------------------------------------------- #


def parse_imglist(text: str, *, name: str = "imglist") -> pd.DataFrame:
    """Parse OpenOOD imglist text into a ``(path, label)`` frame.

    The format is one ``<relative path> <label>`` pair per line, paths relative
    to ``data/images_classic``. Split from the right on whitespace, so a path
    containing a space would still parse; a line with no label at all would not,
    and raises rather than being silently dropped.
    """
    rows: list[tuple[str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            raise ValueError(
                f"{name}: line {lineno} is not '<path> <label>': {raw!r}"
            )
        path, label = parts
        try:
            rows.append((path, int(label)))
        except ValueError as exc:
            raise ValueError(
                f"{name}: line {lineno} has a non-integer label {label!r}"
            ) from exc

    frame = pd.DataFrame(rows, columns=["path", "label"])
    duplicated = frame["path"].duplicated()
    if duplicated.any():
        offenders = sorted(set(frame.loc[duplicated, "path"]))[:5]
        raise ValueError(
            f"{name}: {int(duplicated.sum())} duplicate path(s), first few "
            f"{offenders}. Paths are the image ids downstream, so a duplicate "
            f"would make one photograph two clusters in the bootstrap."
        )
    return frame


def read_id_test_imglist(
    path: str | Path, *, expected_rows: int | None = N_ID_TEST
) -> pd.DataFrame:
    """Read OpenOOD's ``test_cifar10.txt`` and check its size.

    ``expected_rows`` defaults to 9,000 and exists to catch one specific,
    plausible, and expensive mistake: pointing this at the original 10,000-image
    CIFAR-10 test set, or at a concatenation that includes ``val_cifar10.txt``.
    Either would put the 1,000 validation photographs into the cs-ID draw, which
    is a leak from the split reserved for hyperparameter choices into the split
    the covariate-shift result is measured on. It would not raise anywhere else
    and it would not be visible in any number the run reports.
    """
    path = Path(path)
    frame = parse_imglist(path.read_text(), name=str(path))
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"{path}: {len(frame)} rows, expected {expected_rows}. OpenOOD's "
            f"CIFAR-10 test split is 9,000 images -- 1,000 of the original "
            f"10,000 are reserved as val. A 10,000-row file is the full test "
            f"set and would leak val into the cs-ID draw; pass "
            f"expected_rows=None only if you are deliberately building against "
            f"a different split."
        )
    return frame


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #


def draw_shared_subset(
    image_ids: Sequence[str], *, seed: int = SUBSET_SEED, n: int = N_SUBSET
) -> list[str]:
    """Draw ``n`` ids without replacement, deterministically from ``seed``.

    Called **once**. The returned list is the subset for every cell; a caller
    that calls this inside a per-cell loop has built the artefact this whole
    module exists to prevent, and ``assert_csid_grid`` will say so.

    Returned in input order (i.e. imglist order), not draw order. The result is
    a set as far as the design is concerned, and a stable order makes two builds
    comparable by ``==`` rather than by ``set()``.

    Note the draw depends on the *order* of ``image_ids``, so reproducing it
    needs the same source imglist and not merely the same seed. That is why
    ``csid_manifest`` records the source file's SHA-256 alongside the seed.
    """
    ids = list(image_ids)
    if len(set(ids)) != len(ids):
        raise ValueError(
            "image_ids contains duplicates; the draw would be over a multiset "
            "and the resulting cells could not have 2,000 distinct ids."
        )
    if n > len(ids):
        raise ValueError(f"cannot draw {n} ids without replacement from {len(ids)}")
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(ids), size=n, replace=False))
    return [ids[int(i)] for i in positions]


# --------------------------------------------------------------------------- #
# Path construction
# --------------------------------------------------------------------------- #


def csid_path(
    id_path: str,
    corruption: str,
    severity: int,
    *,
    template: str = DEFAULT_PATH_TEMPLATE,
) -> str:
    """Where the corrupted copy of ``id_path`` lives, per ``template``.

    See the module docstring: the template is the one guess in this module, and
    it is a guess precisely so it can be replaced with one argument.
    """
    parts = PurePosixPath(id_path).parts
    if len(parts) < 2:
        raise ValueError(
            f"{id_path!r} has no parent directory, so {{class_name}} cannot be "
            f"resolved. OpenOOD imglist paths look like "
            f"'cifar10/test/cat/3975.png'."
        )
    filename = parts[-1]
    try:
        return template.format(
            corruption=corruption,
            severity=severity,
            class_name=parts[-2],
            filename=filename,
            stem=PurePosixPath(filename).stem,
            id_path=id_path,
        )
    except KeyError as exc:
        raise ValueError(
            f"path template {template!r} refers to unknown field {exc}. "
            f"Available: corruption, severity, class_name, filename, stem, id_path."
        ) from exc


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #


def build_csid_imglist(
    id_test: pd.DataFrame,
    *,
    seed: int = SUBSET_SEED,
    n_subset: int = N_SUBSET,
    corruptions: Sequence[str] = CORRUPTIONS,
    severities: Sequence[int] = SEVERITIES,
    template: str = DEFAULT_PATH_TEMPLATE,
    name: str = "csid imglist",
) -> pd.DataFrame:
    """Build the cs-ID index frame: one shared draw, replicated across every cell.

    ``id_test`` is a ``(path, label)`` frame from ``read_id_test_imglist``.
    The returned frame has ``IMGLIST_COLUMNS``, is in canonical order, and
    has been through ``assert_csid_grid``, ``assert_draw_reproduces``
    and ``assert_canonical_order`` before it is returned.

    Deterministic given ``(seed, id_test)``: calling this twice returns equal
    frames, which is the fourth of the four assertions and is what lets the
    draw be fixed once and reproduced on another machine later.
    """
    corruptions, severities = _validated_grid(corruptions, severities)
    for column in ("path", "label"):
        if column not in id_test.columns:
            raise ValueError(f"{name}: id_test is missing the {column!r} column")

    drawn = draw_shared_subset(list(id_test["path"]), seed=seed, n=n_subset)
    labels = dict(zip(id_test["path"], id_test["label"]))

    # One draw, then a product over the grid. The shape of this loop *is* the
    # commitment: `drawn` is computed above it, never inside it.
    records = [
        {
            "split": CSID_SPLIT,
            "image_id": image_id,
            "corruption": corruption,
            "severity": severity,
            "label": int(labels[image_id]),
            "path": csid_path(image_id, corruption, severity, template=template),
        }
        for corruption, severity in itertools.product(corruptions, severities)
        for image_id in drawn
    ]

    frame = pd.DataFrame.from_records(records, columns=list(IMGLIST_COLUMNS))
    frame = canonical_sort(frame, name=name)

    assert_csid_grid(
        frame,
        expected_ids=drawn,
        corruptions=corruptions,
        severities=severities,
        n_subset=n_subset,
        name=name,
    )
    assert_draw_reproduces(
        frame, id_test, seed=seed, n_subset=n_subset, name=name
    )
    return assert_canonical_order(frame, name=name)


def _validated_grid(
    corruptions: Sequence[str], severities: Sequence[int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    corruptions = tuple(corruptions)
    severities = tuple(int(s) for s in severities)
    if not corruptions or not severities:
        raise ValueError("the grid needs at least one corruption and one severity")
    if len(set(corruptions)) != len(corruptions):
        raise ValueError(f"duplicate corruption name(s) in {list(corruptions)}")
    if len(set(severities)) != len(severities):
        raise ValueError(f"duplicate severity value(s) in {list(severities)}")
    held_out = set(corruptions) & set(HELD_OUT_CORRUPTIONS)
    if held_out:
        raise ValueError(
            f"{sorted(held_out)} are CIFAR-10-C's held-out validation "
            f"corruptions, not part of the standard 15 this thesis uses. "
            f"Including them makes the grid 19-wide and the row counts stop "
            f"matching every number recorded for this design."
        )
    bad = [s for s in severities if s not in VALID_SEVERITIES]
    if bad:
        raise ValueError(
            f"severity value(s) {bad} outside {list(VALID_SEVERITIES)}. 0 is the "
            f"null sentinel in the canonical sort key, not a real severity."
        )
    return corruptions, severities


# --------------------------------------------------------------------------- #
# The four assertions
# --------------------------------------------------------------------------- #


def assert_csid_grid(
    frame: pd.DataFrame,
    *,
    expected_ids: Iterable[str] | None = None,
    corruptions: Sequence[str] = CORRUPTIONS,
    severities: Sequence[int] = SEVERITIES,
    n_subset: int = N_SUBSET,
    name: str = "csid imglist",
) -> pd.DataFrame:
    """Assert the grid is exactly the declared one. Returns ``frame``.

    Three of the four checks live here:

    1. **Exactly the expected cells**, 75 at the defaults -- as a set, not a
       count, so a duplicated cell plus a missing one cannot cancel out.
    2. **Exactly ``n_subset`` rows in every cell.**
    3. **The same ids in every cell**, by set equality against the first cell
       and, when ``expected_ids`` is given, against the draw itself. This is the
       one that catches a per-cell redraw: 75 independent draws would produce 75
       cells of the right size whose id sets differ, and every other check here
       would pass.
    """
    corruptions, severities = _validated_grid(corruptions, severities)
    expected_cells = set(itertools.product(corruptions, severities))

    present = set(map(tuple, frame[["corruption", "severity"]].drop_duplicates().to_numpy()))
    present = {(str(c), int(s)) for c, s in present}
    if present != expected_cells:
        missing = sorted(expected_cells - present)
        extra = sorted(present - expected_cells)
        raise ValueError(
            f"{name}: cell set mismatch. {len(present)} cells present, "
            f"{len(expected_cells)} expected.\n  missing: {missing[:10]}\n"
            f"  unexpected: {extra[:10]}"
        )

    by_cell = frame.groupby(["corruption", "severity"], sort=True)

    sizes = by_cell.size()
    wrong = sizes[sizes != n_subset]
    if len(wrong):
        raise ValueError(
            f"{name}: {len(wrong)} cell(s) do not have exactly {n_subset} rows. "
            f"First few:\n{wrong.head().to_string()}"
        )

    id_sets = {cell: frozenset(group["image_id"]) for cell, group in by_cell}
    reference_cell, reference = next(iter(sorted(id_sets.items())))
    if len(reference) != n_subset:
        raise ValueError(
            f"{name}: cell {reference_cell} has {n_subset} rows but only "
            f"{len(reference)} distinct image ids, so an id repeats within a cell."
        )
    mismatched = sorted(cell for cell, ids in id_sets.items() if ids != reference)
    if mismatched:
        example = id_sets[mismatched[0]]
        raise ValueError(
            f"{name}: {len(mismatched)} of {len(id_sets)} cells hold a different "
            f"set of image ids from {reference_cell}; first is {mismatched[0]}, "
            f"which shares {len(example & reference)} of {n_subset} ids with it.\n"
            f"The 2,000 photographs must be drawn ONCE and reused in every "
            f"cell. This is what a per-cell redraw looks like, and it makes the "
            f"severity curve "
            f"a between-image comparison rather than the within-image one "
            f"CIFAR-10-C was chosen for."
        )

    if expected_ids is not None:
        expected = frozenset(expected_ids)
        if reference != expected:
            raise ValueError(
                f"{name}: the ids in the frame are not the ids that were drawn "
                f"({len(reference & expected)} of {len(expected)} in common)."
            )

    expected_rows = len(expected_cells) * n_subset
    if len(frame) != expected_rows:
        raise ValueError(
            f"{name}: {len(frame)} rows, expected "
            f"{len(expected_cells)} cells x {n_subset} = {expected_rows}"
        )
    return frame


def assert_draw_reproduces(
    frame: pd.DataFrame,
    id_test: pd.DataFrame,
    *,
    seed: int = SUBSET_SEED,
    n_subset: int = N_SUBSET,
    name: str = "csid imglist",
) -> pd.DataFrame:
    """Assert the ids in ``frame`` are what ``seed`` draws from ``id_test``.

    The fourth check, and the one that makes the recorded seed worth
    recording. Re-runs the draw independently and compares; if the two disagree,
    the seed in the manifest does not describe the file on disk, and the same
    imglist could not be rebuilt from it later.
    """
    redrawn = frozenset(
        draw_shared_subset(list(id_test["path"]), seed=seed, n=n_subset)
    )
    actual = frozenset(frame["image_id"])
    if actual != redrawn:
        # Report the frame's own size alongside the overlap. Without it a frame
        # that is a strict *superset* of the draw reads as "n of n ids in
        # common" -- a message saying everything matched, at the one moment it
        # is read, which is when nothing did.
        raise ValueError(
            f"{name}: re-drawing with seed={seed} gives a different subset "
            f"(the frame holds {len(actual)} distinct ids, {len(actual & redrawn)} "
            f"of which are among the {n_subset} redrawn). The seed "
            f"recorded in the manifest does not reproduce this imglist, so it "
            f"cannot be rebuilt. Either the source imglist changed or the draw "
            f"did not come from this seed."
        )
    return frame


# --------------------------------------------------------------------------- #
# Digests, line counts, extraction volume
# --------------------------------------------------------------------------- #


def sha256_text(text: str) -> str:
    """SHA-256 of ``text`` as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_imglist_lines(path: str | Path) -> int:
    """Non-empty line count of an imglist. ``wc -l``, but recordable."""
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


def extraction_volume_report(
    csid_rows: int,
    imglist_counts: Mapping[str, int] | None = None,
    *,
    already_cached: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Total extraction volume, known before extraction rather than during it.

    The extraction-volume report needs the cs-ID size and the far-OOD
    imglist line counts recorded up front. ``imglist_counts`` maps a split name
    to its line count -- get them with ``count_imglist_lines`` on the real
    files. ``already_cached`` is anything an earlier extraction run already
    holds, so the report can separate what still has to run from what does not.
    """
    counts = dict(imglist_counts or {})
    cached = dict(already_cached or {})
    return {
        "csid_rows": int(csid_rows),
        "imglist_line_counts": counts,
        "imglist_line_count_total": int(sum(counts.values())),
        "already_cached": cached,
        "already_cached_total": int(sum(cached.values())),
        "remaining_to_extract": int(csid_rows)
        + int(sum(v for k, v in counts.items() if k not in cached)),
    }


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def imglist_text(frame: pd.DataFrame) -> str:
    """The frame as OpenOOD imglist text: ``<path> <label>`` per row.

    Row order is the frame's order, which ``build_csid_imglist`` guarantees
    is canonical. The extraction cache's ``filelist.txt`` will therefore be
    canonically ordered too, and the index sidecar aligns with it row for row.
    """
    lines = [f"{row.path} {int(row.label)}" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def csid_manifest(
    frame: pd.DataFrame,
    *,
    seed: int,
    n_subset: int,
    corruptions: Sequence[str],
    severities: Sequence[int],
    template: str,
    source: Mapping[str, Any],
    imglist_sha256: str,
    n_id_test: int = N_ID_TEST,
    data_root: str | Path | None = None,
    extraction_volume: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The manifest fragment for this imglist.

    ``seed`` is the point of the whole thing, but a seed alone does not
    reproduce the draw -- ``draw_shared_subset`` is a function of the seed
    *and* of the order of the source imglist. So ``source`` carries that file's
    path, row count and SHA-256, and the two together are what make the draw
    reproducible rather than merely deterministic.
    """
    cells = len(corruptions) * len(severities)
    fragment: dict[str, Any] = {
        "artefact": "cifar10c_csid_imglist",
        "seed": seed,
        "n_subset": n_subset,
        "n_cells": cells,
        "n_rows": len(frame),
        "rows_per_cell": n_subset,
        "corruptions": list(corruptions),
        "corruption_set": "standard_15",
        "held_out_corruptions_excluded": list(HELD_OUT_CORRUPTIONS),
        "severities": list(severities),
        "split_role": CSID_SPLIT,
        "image_id_convention": "id_test_imglist_path",
        "path_template": template,
        # Records whether the one guess in this module was checked against a
        # real tree or merely assumed. A manifest that does not distinguish
        # those two is exactly as useless as no manifest.
        "path_template_verified_against_data_root": data_root is not None,
        "data_root": None if data_root is None else str(data_root),
        "source_imglist": dict(source),
        "imglist_sha256": imglist_sha256,
        "n_id_test": n_id_test,
        "rows_per_drawn_image_full_spectrum": 1 + cells,
        "full_spectrum_id_rows": n_id_test + len(frame),
        "full_spectrum_mixture_weight": full_spectrum_mixture_weight(
            n_id_test, len(frame)
        ),
    }
    if extraction_volume is not None:
        fragment["extraction_volume"] = dict(extraction_volume)
    if extra:
        fragment.update(dict(extra))
    return fragment


def write_csid_imglist(
    out_dir: str | Path,
    id_test_imglist: str | Path,
    *,
    seed: int = SUBSET_SEED,
    n_subset: int = N_SUBSET,
    corruptions: Sequence[str] = CORRUPTIONS,
    severities: Sequence[int] = SEVERITIES,
    template: str = DEFAULT_PATH_TEMPLATE,
    expect_rows: int | None = N_ROWS,
    expected_id_test_rows: int | None = N_ID_TEST,
    data_root: str | Path | None = None,
    far_ood_imglists: Mapping[str, str | Path] | None = None,
    already_cached: Mapping[str, int] | None = None,
    stem: str = "test_cifar10c",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write the imglist, its index sidecar, and its manifest.

    Writes three files into ``out_dir``: ``{stem}.txt`` (OpenOOD format, for the
    extractor), ``{stem}_index.csv`` (the canonical key columns, for scoring),
    and ``{stem}_manifest.json``. All three describe the same rows in the same
    order.

    ``expect_rows`` defaults to 150,000 and ``expected_id_test_rows`` to 9,000.
    Together they are the gate on the write path: neither a grid that is not the
    declared one nor a source split that is not OpenOOD's 9,000-image test set
    can reach disk without the caller saying so out loud. Pass ``None`` to
    either for a deliberately different build.

    ``data_root`` optionally verifies ``DEFAULT_PATH_TEMPLATE`` against the
    real tree -- the one assumption in this module. **Callers should pass it.**
    Failing here costs one argument; not failing here costs a 150,000-image
    extraction run against paths that do not exist.

    Returns the manifest.
    """
    out_dir = Path(out_dir)
    id_test_imglist = Path(id_test_imglist)

    id_test = read_id_test_imglist(
        id_test_imglist, expected_rows=expected_id_test_rows
    )
    frame = build_csid_imglist(
        id_test,
        seed=seed,
        n_subset=n_subset,
        corruptions=corruptions,
        severities=severities,
        template=template,
    )

    if expect_rows is not None and len(frame) != expect_rows:
        raise ValueError(
            f"refusing to write {len(frame)} rows; the declared cs-ID grid is "
            f"{expect_rows} rows ({N_CELLS} cells x {N_SUBSET}). Every row count "
            f"downstream assumes it. Pass expect_rows=None if a different "
            f"build is genuinely intended."
        )

    if data_root is not None:
        _assert_paths_exist(frame, data_root)

    text = imglist_text(frame)
    out_dir.mkdir(parents=True, exist_ok=True)
    imglist_path = out_dir / f"{stem}.txt"
    index_path = out_dir / f"{stem}_index.csv"
    manifest_path = out_dir / f"{stem}_manifest.json"

    imglist_path.write_text(text)
    frame.to_csv(index_path, index=False)

    volume = extraction_volume_report(
        len(frame),
        {name: count_imglist_lines(p) for name, p in (far_ood_imglists or {}).items()},
        already_cached=already_cached,
    )
    manifest = csid_manifest(
        frame,
        seed=seed,
        n_subset=n_subset,
        corruptions=tuple(corruptions),
        severities=tuple(int(s) for s in severities),
        template=template,
        source={
            "path": str(id_test_imglist),
            "n_rows": len(id_test),
            "sha256": sha256_file(id_test_imglist),
        },
        imglist_sha256=sha256_text(text),
        # Measured off the file that was actually read, never the constant.
        # They coincide on the declared path, which is what makes the constant
        # dangerous rather than harmless: with expected_id_test_rows=None the
        # manifest would otherwise report 9,000 for a source that is not 9,000,
        # and both derived counts below it would follow the wrong number.
        n_id_test=len(id_test),
        data_root=data_root,
        extraction_volume=volume,
        extra={
            "files": {
                "imglist": str(imglist_path),
                "index": str(index_path),
            },
            **dict(extra or {}),
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _assert_paths_exist(frame: pd.DataFrame, data_root: str | Path) -> None:
    """Check every emitted path resolves under ``data_root``.

    Reports how many are missing and shows a few, rather than dying on the
    first: a wrong template misses all 150,000 and a genuinely absent corruption
    directory misses exactly 10,000, and the count is what tells those apart.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"data_root {root} is not a directory")
    missing = [p for p in frame["path"] if not (root / p).exists()]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(frame)} cs-ID paths do not exist under "
            f"{root}. First few: {missing[:5]}\n"
            f"If all of them are missing, the path template is wrong -- it is "
            f"the one guess in this module (see the module docstring), and it "
            f"is a single argument to change. If a round 10,000 are missing, "
            f"one (corruption, severity) directory is absent from the download."
        )
