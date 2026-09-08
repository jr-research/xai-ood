"""Convert the CIFAR-10-C corruption arrays into the per-image tree the cs-ID
imglist points at.

Reads the fifteen ``{corruption}.npy`` arrays, each ``(50000, 32, 32, 3)``
uint8, and writes one PNG per (drawn photograph, corruption, severity) under
``<data-root>/cifar10c/``, at the four-level path
``xai_ood.csid_imglist.DEFAULT_PATH_TEMPLATE`` names. The photographs are the
2,000 shared ids that module draws, the grid is its fifteen corruptions by five
severities, and the product is 150,000 files.

The per-class 1-indexed trap
----------------------------
OpenOOD numbers ``cifar10/test`` **per class and 1-indexed**. Every class
directory runs ``0001.png`` to ``1000.png``, and the next class restarts at
``0001.png``. CIFAR-10-C is indexed by position in the original 10,000-image
CIFAR-10 test set, 0 to 9999. The two numbers look alike, occupy the same
range, and agree almost nowhere: ``cifar10/test/airplane/0298.png`` is original
test index 2979.

Reading a filename as an index would hand every corrupted image the pixels of a
different clean photograph, and nothing would raise. The file count would be
right, the paths would be right, every image would be a real corrupted CIFAR-10
image, and the only casualty would be the correspondence that makes a
photograph's corrupted copies belong to it. So the correspondence is an input
rather than a derivation: a CSV built by exact pixel match, validated in both
directions on load, and never reconstructed from a filename here.

Row arithmetic
--------------
Each corruption array stacks five severity blocks of the whole test set::

    corrupted_row = (severity - 1) * 10000 + original_test_index

``labels.npy`` is what checks that layout rather than what assumes it. It holds
50,000 labels whose five 10,000-row blocks must be identical, and they are
compared here before a pixel is read.

Usage::

    python3 build_csid_images.py --data-root DIR --cifar10c-dir DIR \\
        --index-map FILE --id-imglist FILE --out-manifest FILE \\
        --out-contact-sheet FILE [--limit N] [--repo-commit SHA]
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence, TextIO

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from xai_ood.csid_imglist import (
    CORRUPTIONS,
    DEFAULT_PATH_TEMPLATE,
    N_CELLS,
    N_ID_TEST,
    N_SUBSET,
    ROWS_PER_DRAWN_IMAGE,
    SEVERITIES,
    SUBSET_SEED,
    assert_csid_grid,
    assert_draw_reproduces,
    csid_path,
    draw_shared_subset,
    read_id_test_imglist,
    sha256_file,
)

#: The original CIFAR-10 test set, all of which CIFAR-10-C corrupts. A valid
#: ``original_test_index`` lies in ``[0, N_ORIGINAL_TEST - 1]``.
N_ORIGINAL_TEST: int = 10_000

#: Rows of one severity block. Every severity carries the whole test set, which
#: is what makes the block stride the test set size rather than a free
#: parameter.
ROWS_PER_SEVERITY_BLOCK: int = N_ORIGINAL_TEST

#: Rows of a whole corruption array: one block per severity.
N_ARRAY_ROWS: int = len(SEVERITIES) * ROWS_PER_SEVERITY_BLOCK

#: Every CIFAR-10 image, and so every array row.
IMAGE_SHAPE: tuple[int, ...] = (32, 32, 3)

#: Shape every ``{corruption}.npy`` must have.
ARRAY_SHAPE: tuple[int, ...] = (N_ARRAY_ROWS, *IMAGE_SHAPE)

#: Files one drawn photograph receives: one per cell. ``ROWS_PER_DRAWN_IMAGE``
#: counts that photograph's clean row too, so it is one larger. Derived rather
#: than written down, so the grid width keeps a single definition.
FILES_PER_DRAWN_PHOTOGRAPH: int = ROWS_PER_DRAWN_IMAGE - 1

#: Labels file beside the corruption arrays.
LABELS_FILENAME: str = "labels.npy"

#: Header of the index map, in order.
INDEX_MAP_COLUMNS: tuple[str, ...] = ("imglist_path", "original_test_index")

#: Rows of the index map: OpenOOD's CIFAR-10 test split, which reserves 1,000
#: of the original 10,000 as a validation split and so resolves 9,000.
N_INDEX_MAP_ROWS: int = N_ID_TEST

#: First path segment ``DEFAULT_PATH_TEMPLATE`` emits, and the only directory
#: under the data root this script writes into.
OUTPUT_SUBDIR: str = "cifar10c"

#: The unusable flat CIFAR-10-C set OpenOOD distributes, renamed aside next to
#: the output tree. Writing into it would overwrite an archived artefact.
FLAT_ARCHIVE_NAME: str = "cifar10c_openood_flat"

#: Severity rows of the contact sheet. Severity 5 is the load-bearing one: at
#: severity 1 several corruptions are near-threshold and a wrong pairing can
#: still look plausible, while at severity 5 a pair is either obviously the
#: same scene under obvious damage or obviously not.
CONTACT_SHEET_SEVERITIES: tuple[int, ...] = (1, 5)

#: Magnification of each 32x32 tile on the contact sheet. Nearest-neighbour
#: rather than smooth, so the sheet shows the pixels that were written instead
#: of an interpolation of them.
CONTACT_SHEET_SCALE: int = 4

#: Contact sheet furniture, in pixels: room for the row captions on the left,
#: for the column captions above, and between tiles.
CONTACT_SHEET_MARGIN: int = 78
CONTACT_SHEET_CAPTION: int = 13
CONTACT_SHEET_GAP: int = 3


class ConverterError(RuntimeError):
    """A check refused an input or an output."""


class RoundTripFailure(ConverterError):
    """A written PNG did not read back as the array it was written from."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def read_index_map(
    path: str | Path, *, expected_rows: int = N_INDEX_MAP_ROWS
) -> dict[str, int]:
    """Read and validate the OpenOOD-path to original-test-index map.

    Validated rather than trusted: it is an input this script reads, not a
    result it produces, and a map that had drifted would repair nothing while
    breaking the correspondence invisibly.

    Uniqueness is checked in both directions before any dictionary is built,
    because only one of the two failures is visible afterwards. A repeated
    ``imglist_path`` is absorbed silently by a dict, which keeps the last of
    them; a repeated ``original_test_index`` means two clean photographs would
    receive the same corrupted pixels.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    if tuple(frame.columns) != INDEX_MAP_COLUMNS:
        raise ConverterError(
            f"{path}: header is {tuple(frame.columns)}, expected "
            f"{INDEX_MAP_COLUMNS}"
        )
    if len(frame) != expected_rows:
        raise ConverterError(
            f"{path}: {len(frame)} rows, expected {expected_rows}. OpenOOD's "
            f"CIFAR-10 test split is 9,000 images; 1,000 of the original "
            f"10,000 are reserved as a validation split and resolve to no "
            f"imglist path."
        )

    for column, consequence in (
        (
            INDEX_MAP_COLUMNS[0],
            "a mapping keyed by path keeps only the last of them and the "
            "earlier row vanishes without raising",
        ),
        (
            INDEX_MAP_COLUMNS[1],
            "two clean photographs would receive the same corrupted pixels",
        ),
    ):
        duplicated = frame[column].duplicated(keep=False)
        if duplicated.any():
            offenders = sorted(set(frame.loc[duplicated, column].tolist()))[:5]
            raise ConverterError(
                f"{path}: {int(frame[column].duplicated().sum())} duplicate "
                f"{column} value(s), first few {offenders}. Refused because "
                f"{consequence}."
            )

    indices = frame[INDEX_MAP_COLUMNS[1]]
    if not pd.api.types.is_integer_dtype(indices):
        raise ConverterError(
            f"{path}: {INDEX_MAP_COLUMNS[1]} has dtype {indices.dtype}, which "
            f"means at least one value is not an integer"
        )
    outside = frame[(indices < 0) | (indices >= N_ORIGINAL_TEST)]
    if len(outside):
        raise ConverterError(
            f"{path}: {len(outside)} index(es) outside [0, "
            f"{N_ORIGINAL_TEST - 1}], first few "
            f"{outside[INDEX_MAP_COLUMNS[1]].tolist()[:5]}"
        )

    return dict(zip(frame[INDEX_MAP_COLUMNS[0]], (int(i) for i in indices)))


def load_labels(cifar10c_dir: str | Path) -> np.ndarray:
    """Read ``labels.npy`` and check the severity-block layout it certifies.

    The row arithmetic rests on every severity block holding the whole test set
    in the same order. That is checkable rather than assumable: the five label
    blocks have to be identical, and if they are not, the offset selects the
    wrong photograph at every severity above 1.
    """
    path = Path(cifar10c_dir) / LABELS_FILENAME
    if not path.is_file():
        raise ConverterError(f"{path} does not exist")
    labels = np.load(path)
    if labels.shape != (N_ARRAY_ROWS,):
        raise ConverterError(
            f"{path}: shape {labels.shape}, expected ({N_ARRAY_ROWS},), which "
            f"is {len(SEVERITIES)} severity blocks of {ROWS_PER_SEVERITY_BLOCK}"
        )
    blocks = labels.reshape(len(SEVERITIES), ROWS_PER_SEVERITY_BLOCK)
    differing = [
        i
        for i in range(1, len(SEVERITIES))
        if not np.array_equal(blocks[0], blocks[i])
    ]
    if differing:
        raise ConverterError(
            f"{path}: severity block(s) {differing} differ from the first. The "
            f"blocks must be identical for "
            f"(severity - 1) * {ROWS_PER_SEVERITY_BLOCK} + index to select the "
            f"same photograph at every severity."
        )
    return labels


def load_corruption_array(
    cifar10c_dir: str | Path, corruption: str
) -> tuple[np.ndarray, Path]:
    """Memory-map one corruption array and check its shape and dtype.

    Memory-mapped rather than read: the fifteen arrays are roughly 2.3 GB
    together, and only one row of one of them is needed at a time.
    """
    path = Path(cifar10c_dir) / f"{corruption}.npy"
    if not path.is_file():
        raise ConverterError(f"{path} does not exist")
    array = np.load(path, mmap_mode="r")
    if array.dtype != np.uint8:
        raise ConverterError(f"{path}: dtype {array.dtype}, expected uint8")
    if array.shape != ARRAY_SHAPE:
        raise ConverterError(f"{path}: shape {array.shape}, expected {ARRAY_SHAPE}")
    return array, path


def corrupted_row(severity: int, original_test_index: int) -> int:
    """Row of a corruption array holding one photograph at one severity.

    The arithmetic lives here alone, so the offset cannot be written in two
    places and drift.
    """
    if severity not in SEVERITIES:
        raise ConverterError(f"severity {severity} outside {list(SEVERITIES)}")
    if not 0 <= original_test_index < N_ORIGINAL_TEST:
        raise ConverterError(
            f"original test index {original_test_index} outside "
            f"[0, {N_ORIGINAL_TEST - 1}]"
        )
    row = (severity - 1) * ROWS_PER_SEVERITY_BLOCK + original_test_index
    if not 0 <= row < N_ARRAY_ROWS:
        raise ConverterError(
            f"computed row {row} for severity {severity}, index "
            f"{original_test_index}, is outside the array. A negative row is a "
            f"valid numpy index that resolves from the far end, which would pair "
            f"every photograph correctly at the wrong severity and raise nothing, "
            f"so the arithmetic is checked on its result and not only on its "
            f"arguments."
        )
    return row


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #


def relative_output_path(image_id: str, corruption: str, severity: int) -> str:
    """Path under the data root for one corrupted copy, checked for escapes.

    Built by ``csid_path``, which owns the layout and is tested where it lives.
    What is added here is containment: the result has to stay inside the one
    output directory, since the embedding cache and the whole OpenOOD image
    tree hang off the same root.
    """
    relative = csid_path(
        image_id, corruption, severity, template=DEFAULT_PATH_TEMPLATE
    )
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or ".." in parts:
        raise ConverterError(f"path template produced an escaping path: {relative!r}")
    if not parts or parts[0] != OUTPUT_SUBDIR:
        raise ConverterError(
            f"path template produced {relative!r}, which is not under "
            f"{OUTPUT_SUBDIR}/. Writing outside it reaches the embedding cache "
            f"and the OpenOOD image tree."
        )
    if FLAT_ARCHIVE_NAME in parts:
        raise ConverterError(
            f"path template produced {relative!r}, inside {FLAT_ARCHIVE_NAME}, "
            f"which holds an archived set rather than converted output"
        )
    return relative


# --------------------------------------------------------------------------- #
# Writing, and the round trip
# --------------------------------------------------------------------------- #


def assert_round_trip(path: str | Path, array: np.ndarray) -> None:
    """Refuse a PNG that does not read back as ``array``.

    PNG is lossless, but that stops holding silently if the array is float, if
    a mode conversion to RGBA or to a palette happens, or if anything resizes.
    All three are ordinary mistakes and none of them raises on its own, so the
    identity is asserted rather than inherited from the format.
    """
    with Image.open(path) as reopened:
        mode = reopened.mode
        pixels = np.array(reopened)
    if mode != "RGB":
        raise RoundTripFailure(f"{path}: reopened as {mode}, expected RGB")
    if pixels.shape != array.shape:
        raise RoundTripFailure(
            f"{path}: reopened shape {pixels.shape}, wrote {array.shape}"
        )
    if not np.array_equal(pixels, array):
        differing = int(np.count_nonzero(pixels != array))
        raise RoundTripFailure(
            f"{path}: round trip is not identity, {differing} of {array.size} "
            f"values differ"
        )


def write_png(path: str | Path, array: np.ndarray, *, allow_skip: bool = True) -> bool:
    """Write one 32x32 RGB PNG and check it reads back identically.

    Returns whether a file was written. ``False`` means an existing file
    already held these exact pixels and was left alone.

    ``allow_skip`` makes a partial run resumable. An existing file is kept only
    when it passes the round trip, never merely because it exists: a truncated
    file from an interrupted run exists too, and skipping on existence would
    preserve exactly the damage this check is here to find.
    """
    path = Path(path)
    if array.dtype != np.uint8:
        raise ConverterError(f"{path}: dtype {array.dtype}, expected uint8")
    if array.shape != IMAGE_SHAPE:
        raise ConverterError(f"{path}: shape {array.shape}, expected {IMAGE_SHAPE}")

    if allow_skip and path.exists():
        try:
            assert_round_trip(path, array)
        except (RoundTripFailure, OSError, ValueError):
            pass
        else:
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="PNG")
    assert_round_trip(path, array)
    return True


# --------------------------------------------------------------------------- #
# The conversion
# --------------------------------------------------------------------------- #


def grid_cells(limit: int | None) -> list[tuple[str, int]]:
    """The (corruption, severity) cells to convert, in corruption-major order.

    ``limit`` truncates the grid for a partial run. It is validated at the call
    site to leave a complete rectangle, since the grid assertions compare a set
    of cells against a product and a ragged selection is no product.
    """
    cells = list(itertools.product(CORRUPTIONS, SEVERITIES))
    return cells if limit is None else cells[:limit]


def convert(
    *,
    data_root: Path,
    cifar10c_dir: Path,
    drawn: Sequence[str],
    index_map: dict[str, int],
    cells: Sequence[tuple[str, int]],
    log: TextIO = sys.stdout,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, dict[str, Any]]]:
    """Write every cell and return an inventory of what was written.

    The loop is corruption-major so each array is memory-mapped once. The
    inventory frame carries the columns the grid assertions read, plus the
    path, so what was written can be checked against the layout afterwards
    rather than only counted.
    """

    def say(message: str) -> None:
        print(message, file=log, flush=True)

    by_corruption: dict[str, list[int]] = {}
    for corruption, severity in cells:
        by_corruption.setdefault(corruption, []).append(severity)

    records: list[dict[str, Any]] = []
    counts = {"written": 0, "skipped": 0}
    sources: dict[str, dict[str, Any]] = {}

    for corruption, severities in by_corruption.items():
        array, array_path = load_corruption_array(cifar10c_dir, corruption)
        sources[corruption] = {
            "path": str(array_path),
            "bytes": array_path.stat().st_size,
            "sha256": sha256_file(array_path),
        }
        say(
            f"[read] {array_path} {array.shape} sha256 "
            f"{sources[corruption]['sha256'][:12]}"
        )

        for severity in severities:
            written = skipped = 0
            for image_id in drawn:
                row = corrupted_row(severity, index_map[image_id])
                relative = relative_output_path(image_id, corruption, severity)
                if write_png(data_root / relative, np.array(array[row])):
                    written += 1
                else:
                    skipped += 1
                records.append(
                    {
                        "image_id": image_id,
                        "corruption": corruption,
                        "severity": severity,
                        "path": relative,
                    }
                )
            counts["written"] += written
            counts["skipped"] += skipped
            say(
                f"[cell] {corruption} severity {severity}: {written} written, "
                f"{skipped} already correct"
            )

        del array

    frame = pd.DataFrame.from_records(
        records, columns=["image_id", "corruption", "severity", "path"]
    )
    return frame, counts, sources


def verify_written_paths(frame: pd.DataFrame, data_root: Path) -> None:
    """Re-derive every path from the layout and confirm the file is there.

    This guards the layout, not the correspondence. A run that wrote correctly
    named files holding the wrong images passes it cleanly, which is why the
    index map is validated on load and why the contact sheet exists.

    Only the existence check is load bearing today. The recomputed path is
    derived from the same three stored fields that produced the written path,
    so that arm cannot fail while both come from ``relative_output_path``. It
    is kept because it starts guarding something the moment ``path`` is ever
    populated from another source.

    Counts the misses instead of dying on the first, because the count is what
    separates a wrong template, which misses everything, from one absent
    corruption directory, which misses a round 10,000.
    """
    missing: list[str] = []
    mismatched: list[tuple[str, str]] = []
    for record in frame.itertuples(index=False):
        expected = relative_output_path(
            record.image_id, record.corruption, int(record.severity)
        )
        if expected != record.path:
            mismatched.append((record.path, expected))
        elif not (data_root / expected).is_file():
            missing.append(expected)

    if mismatched:
        raise ConverterError(
            f"{len(mismatched)} written path(s) disagree with the layout. "
            f"First few: {mismatched[:3]}"
        )
    if missing:
        raise ConverterError(
            f"{len(missing)} of {len(frame)} expected file(s) are not on disk. "
            f"First few: {missing[:5]}"
        )


def assert_counts(
    frame: pd.DataFrame, cells: Sequence[tuple[str, int]], n_subset: int
) -> None:
    """Check the counts against values fixed before the run, not reported after.

    A count printed after a run is a report. A count compared against a number
    named in advance is a check, and these are the numbers the design rests on:
    one file per drawn photograph per cell, no path written twice, and a full
    grid whose width matches the rows a drawn photograph owns.
    """
    expected_rows = len(cells) * n_subset
    if len(frame) != expected_rows:
        raise ConverterError(
            f"{len(frame)} files, expected {len(cells)} cells x {n_subset} = "
            f"{expected_rows}"
        )
    if frame["path"].duplicated().any():
        repeated = frame.loc[frame["path"].duplicated(), "path"].tolist()[:5]
        raise ConverterError(
            f"the same path was written more than once, first few {repeated}"
        )
    if len(cells) == N_CELLS and FILES_PER_DRAWN_PHOTOGRAPH != N_CELLS:
        raise ConverterError(
            f"a drawn photograph receives {FILES_PER_DRAWN_PHOTOGRAPH} files "
            f"but the grid has {N_CELLS} cells; the two constants have drifted"
        )


# --------------------------------------------------------------------------- #
# The contact sheet
# --------------------------------------------------------------------------- #


def contact_sheet(
    image_id: str,
    data_root: Path,
    out_path: Path,
    *,
    corruptions: Sequence[str] = CORRUPTIONS,
    severities: Sequence[int] = CONTACT_SHEET_SEVERITIES,
) -> Path:
    """One photograph, clean and under every corruption, at two severities.

    One photograph across the corruptions rather than several photographs under
    one corruption, because each corruption is a separate array on disk. A
    layout showing several photographs under one corruption exercises the
    indexing of exactly one array, so a per-array fault such as one file
    written in a different row order stays invisible in it and is obvious here.

    Severity 5 carries the weight for the reason the severity constant gives:
    near-threshold corruptions can make a wrong pairing look plausible, and
    heavy damage cannot.
    """
    tile = IMAGE_SHAPE[0] * CONTACT_SHEET_SCALE
    step = tile + CONTACT_SHEET_GAP
    columns = ["clean", *corruptions]
    width = CONTACT_SHEET_MARGIN + len(columns) * step
    height = CONTACT_SHEET_CAPTION + len(severities) * (step + CONTACT_SHEET_CAPTION)

    sheet = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for column, name in enumerate(columns):
        draw.text((CONTACT_SHEET_MARGIN + column * step, 2), name, fill=(0, 0, 0))

    for row, severity in enumerate(severities):
        top = CONTACT_SHEET_CAPTION + row * (step + CONTACT_SHEET_CAPTION)
        draw.text((2, top + tile // 2), f"severity {severity}", fill=(0, 0, 0))
        for column, name in enumerate(columns):
            relative = (
                image_id
                if name == "clean"
                else relative_output_path(image_id, name, severity)
            )
            source = data_root / relative
            if not source.is_file():
                raise ConverterError(
                    f"contact sheet needs {source}, which is absent"
                )
            with Image.open(source) as opened:
                tile_image = opened.convert("RGB").resize(
                    (tile, tile), resample=Image.NEAREST
                )
            sheet.paste(tile_image, (CONTACT_SHEET_MARGIN + column * step, top))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")
    return out_path


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


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
        raise ConverterError(
            f"cannot read the commit of {root}: {result.stderr.strip()}. "
            f"Pass --repo-commit."
        )
    return result.stdout.strip()


def build_manifest(
    *,
    options: argparse.Namespace,
    cells: Sequence[tuple[str, int]],
    drawn: Sequence[str],
    counts: dict[str, int],
    sources: dict[str, dict[str, Any]],
    labels_sha256: str,
    contact_sheet_path: Path | None,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Record what cannot be recovered once the archive is gone.

    The array digests are provenance, not verification. A matching digest says
    the same CIFAR-10-C distribution was on disk; it says nothing about whether
    the indexing was right, which is what the index map and the contact sheet
    are for. Each digest is recorded beside that array's size, so a truncated
    download and a different download stay distinguishable.
    """
    return {
        "artefact": "cifar10c_csid_images",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": options.repo_commit,
        "data_root": str(options.data_root),
        "output_subdir": OUTPUT_SUBDIR,
        "cifar10c_dir": str(options.cifar10c_dir),
        "path_template": DEFAULT_PATH_TEMPLATE,
        "corruptions": sorted({c for c, _ in cells}),
        "severities": sorted({s for _, s in cells}),
        "n_cells": len(cells),
        "full_grid": len(cells) == N_CELLS,
        "seed": SUBSET_SEED,
        "n_drawn": len(drawn),
        "rows_per_severity_block": ROWS_PER_SEVERITY_BLOCK,
        "files_per_drawn_photograph": FILES_PER_DRAWN_PHOTOGRAPH,
        "index_map": {
            "path": str(options.index_map),
            "sha256": sha256_file(options.index_map),
        },
        "id_imglist": {
            "path": str(options.id_imglist),
            "sha256": sha256_file(options.id_imglist),
        },
        "labels": {
            "path": str(Path(options.cifar10c_dir) / LABELS_FILENAME),
            "sha256": labels_sha256,
        },
        "corruption_arrays": sources,
        "files": {
            "expected": len(cells) * len(drawn),
            "total": len(frame),
            "written": counts["written"],
            "skipped": counts["skipped"],
            "round_trip_failures": 0,
        },
        "contact_sheet": (
            None if contact_sheet_path is None else str(contact_sheet_path)
        ),
    }


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the run options. Paths are arguments here, never literals."""
    parser = argparse.ArgumentParser(
        description="Convert CIFAR-10-C arrays into the cs-ID image tree."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cifar10c-dir", type=Path, required=True)
    parser.add_argument("--index-map", type=Path, required=True)
    parser.add_argument("--id-imglist", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-contact-sheet", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repo-commit")

    options = parser.parse_args(argv)

    for name in ("data_root", "cifar10c_dir"):
        directory = getattr(options, name)
        if not directory.is_dir():
            raise SystemExit(
                f"--{name.replace('_', '-')} {directory} is not a directory"
            )
    if FLAT_ARCHIVE_NAME in options.data_root.resolve().parts:
        raise SystemExit(
            f"--data-root resolves inside {FLAT_ARCHIVE_NAME}, which holds an "
            f"archived flat set rather than the converted tree"
        )
    for name in ("index_map", "id_imglist"):
        handle = getattr(options, name)
        if not handle.is_file():
            raise SystemExit(f"--{name.replace('_', '-')} {handle} is not a file")

    if options.limit is not None:
        if options.limit < 1 or options.limit > N_CELLS:
            raise SystemExit(f"--limit must be between 1 and {N_CELLS}")
        if options.limit > len(SEVERITIES) and options.limit % len(SEVERITIES):
            raise SystemExit(
                f"--limit {options.limit} stops part way through a corruption, "
                f"leaving a set of cells that is not a rectangle, and the grid "
                f"assertions compare against a product. Use a value up to "
                f"{len(SEVERITIES)} for one corruption, or a multiple of "
                f"{len(SEVERITIES)}."
            )
    return options


def run(options: argparse.Namespace, log: TextIO = sys.stdout) -> dict[str, Any]:
    """Convert, check, write the sheet and the manifest, and return it."""

    def say(message: str) -> None:
        print(message, file=log, flush=True)

    if options.repo_commit is None:
        options.repo_commit = repo_commit_here()

    index_map = read_index_map(options.index_map)
    say(
        f"[read] {options.index_map}: {len(index_map)} paths, unique in both "
        f"directions"
    )

    labels = load_labels(options.cifar10c_dir)
    labels_sha256 = sha256_file(Path(options.cifar10c_dir) / LABELS_FILENAME)
    say(
        f"[read] {LABELS_FILENAME}: {labels.shape[0]} labels, "
        f"{len(SEVERITIES)} identical severity blocks"
    )

    id_test = read_id_test_imglist(options.id_imglist)
    drawn = draw_shared_subset(list(id_test["path"]), seed=SUBSET_SEED, n=N_SUBSET)
    unmapped = [image_id for image_id in drawn if image_id not in index_map]
    if unmapped:
        raise ConverterError(
            f"{len(unmapped)} drawn id(s) are absent from the index map, first "
            f"few {unmapped[:5]}. Every drawn photograph needs an original "
            f"test index or it cannot be located in a corruption array."
        )
    say(f"[read] {options.id_imglist}: {len(id_test)} rows, {len(drawn)} drawn")

    cells = grid_cells(options.limit)
    say(
        f"[plan] {len(cells)} cell(s) x {len(drawn)} photographs = "
        f"{len(cells) * len(drawn)} files"
    )

    frame, counts, sources = convert(
        data_root=options.data_root,
        cifar10c_dir=options.cifar10c_dir,
        drawn=drawn,
        index_map=index_map,
        cells=cells,
        log=log,
    )

    assert_counts(frame, cells, len(drawn))
    assert_csid_grid(
        frame,
        expected_ids=drawn,
        corruptions=sorted({c for c, _ in cells}),
        severities=sorted({s for _, s in cells}),
        n_subset=len(drawn),
        name="converted images",
    )
    assert_draw_reproduces(frame, id_test, seed=SUBSET_SEED, n_subset=len(drawn))
    verify_written_paths(frame, options.data_root)
    say(
        f"[check] {len(frame)} files, {len(cells)} cells of {len(drawn)}, one "
        f"shared photograph set, every expected file present on disk"
    )

    sheet: Path | None = None
    covered_corruptions = {c for c, _ in cells}
    covered_severities = {s for _, s in cells}
    if covered_corruptions == set(CORRUPTIONS) and set(
        CONTACT_SHEET_SEVERITIES
    ) <= covered_severities:
        sheet = contact_sheet(drawn[0], options.data_root, options.out_contact_sheet)
        say(f"[write] {sheet}")
        say(
            f"[next] look at {sheet}. Every tile in a row is the same "
            f"photograph under a different corruption. At severity 5 a correct "
            f"pairing is obviously one scene under obvious damage; if any tile "
            f"shows a different subject, stop and do not extract."
        )
    else:
        say(
            f"[skip] no contact sheet: it needs every corruption at severities "
            f"{list(CONTACT_SHEET_SEVERITIES)}, and this run covered "
            f"{len(cells)} of {N_CELLS} cells"
        )

    manifest = build_manifest(
        options=options,
        cells=cells,
        drawn=drawn,
        counts=counts,
        sources=sources,
        labels_sha256=labels_sha256,
        contact_sheet_path=sheet,
        frame=frame,
    )
    options.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    options.out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    say(f"[write] {options.out_manifest}")
    say(
        f"[done] {counts['written']} written, {counts['skipped']} already "
        f"correct, 0 round trip failures"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    try:
        run(options)
    except ConverterError as error:
        print(f"[fail] {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
