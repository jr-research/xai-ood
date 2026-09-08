# tests/test_build_csid_images.py
#
# Three checks, deliberately. Everything else about the converter is checked at
# runtime against the real arrays, where the evidence is better than a
# synthetic fixture can give: the counts, the layout, the path existence and
# the contact sheet all need data this fixture does not have.
#
# What is left are the three that a run cannot establish about itself. The
# first is the row arithmetic, which is the one place a wrong answer stays
# plausible all the way to the results. The other two are refusals, and a
# refusal that has never fired is a refusal nobody has tested: they would
# otherwise first execute at the moment they matter.
#
# Every path-shaped property belongs to the layout builder in the library and
# is tested where that lives, so none of it is repeated here.
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_csid_images import (  # noqa: E402
    ARRAY_SHAPE,
    IMAGE_SHAPE,
    INDEX_MAP_COLUMNS,
    LABELS_FILENAME,
    N_INDEX_MAP_ROWS,
    ROWS_PER_SEVERITY_BLOCK,
    ConverterError,
    RoundTripFailure,
    assert_round_trip,
    convert,
    corrupted_row,
    load_labels,
    read_index_map,
    write_png,
)
from xai_ood.csid_imglist import SEVERITIES  # noqa: E402


def _pattern(seed: int) -> np.ndarray:
    """A distinct 32x32x3 uint8 block per seed, so equality locates a row."""
    return np.random.default_rng(seed).integers(
        0, 256, size=IMAGE_SHAPE, dtype=np.uint8
    )


def _fake_corruption_array(path: Path, rows: dict[int, np.ndarray]) -> None:
    """Write a full-shape corruption array holding only the named rows.

    Allocated through ``open_memmap``, which sizes the file without writing
    it, so the array has the real 50,000-row shape the loader checks while only
    the touched pages reach disk.
    """
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=ARRAY_SHAPE
    )
    for row, block in rows.items():
        array[row] = block
    array.flush()
    del array


def _fake_labels(path: Path) -> None:
    """Labels with five identical severity blocks, which is the real layout."""
    block = np.arange(ROWS_PER_SEVERITY_BLOCK, dtype=np.int64) % 10
    np.save(path, np.tile(block, len(SEVERITIES)))


def _index_map_frame() -> pd.DataFrame:
    """A well formed index map: unique paths, unique indices, all in range."""
    return pd.DataFrame(
        [(f"cifar10/test/airplane/{i + 1:04d}.png", i) for i in range(N_INDEX_MAP_ROWS)],
        columns=list(INDEX_MAP_COLUMNS),
    )


def test_the_row_is_the_severity_block_offset_plus_the_mapped_original_index(
    tmp_path,
):
    """The corrupted row comes from the map, never from the filename.

    The two candidate rows are both filled in the fixture, with different
    pixels, so the check distinguishes the mapped index from the number in the
    filename rather than merely finding something at the row it chose.
    """
    cifar10c = tmp_path / "CIFAR-10-C"
    cifar10c.mkdir()
    data_root = tmp_path / "images_classic"
    data_root.mkdir()

    _fake_labels(cifar10c / LABELS_FILENAME)
    assert load_labels(cifar10c).shape == (len(SEVERITIES) * ROWS_PER_SEVERITY_BLOCK,)

    image_id = "cifar10/test/airplane/0298.png"
    original_index = 2979
    corruption, severity = "fog", 3

    mapped_row = (severity - 1) * ROWS_PER_SEVERITY_BLOCK + original_index
    assert corrupted_row(severity, original_index) == mapped_row

    # The number in the filename counts airplanes from one, so the row it would
    # select holds an unrelated photograph at the same severity.
    filename_row = (severity - 1) * ROWS_PER_SEVERITY_BLOCK + 298
    array_path = cifar10c / f"{corruption}.npy"
    _fake_corruption_array(
        array_path,
        {mapped_row: _pattern(mapped_row), filename_row: _pattern(filename_row)},
    )

    frame, counts, sources = convert(
        data_root=data_root,
        cifar10c_dir=cifar10c,
        drawn=[image_id],
        index_map={image_id: original_index},
        cells=[(corruption, severity)],
        log=io.StringIO(),
    )

    assert counts == {"written": 1, "skipped": 0}
    assert list(frame["path"]) == [f"cifar10c/{corruption}/{severity}/airplane/0298.png"]

    with Image.open(data_root / frame["path"].iloc[0]) as written:
        pixels = np.array(written)
    array = np.load(array_path, mmap_mode="r")
    assert np.array_equal(pixels, array[mapped_row])
    assert not np.array_equal(pixels, array[filename_row])

    assert sources[corruption]["bytes"] == array_path.stat().st_size


def test_a_repeated_index_map_row_is_refused_in_both_directions(tmp_path):
    """Neither direction of a duplicate survives the load.

    Both are tested because only one of them is visible later. A repeated path
    is what a dict would absorb without complaint, keeping the last of them; a
    repeated index is what would give two clean photographs the same corrupted
    pixels.
    """
    good = _index_map_frame()
    path = tmp_path / "cifar10c_index_map.csv"
    good.to_csv(path, index=False)
    assert len(read_index_map(path)) == N_INDEX_MAP_ROWS

    repeated_path = good.copy()
    repeated_path.loc[1, INDEX_MAP_COLUMNS[0]] = repeated_path.loc[
        0, INDEX_MAP_COLUMNS[0]
    ]
    repeated_path.to_csv(path, index=False)
    with pytest.raises(ConverterError, match=INDEX_MAP_COLUMNS[0]):
        read_index_map(path)

    repeated_index = good.copy()
    repeated_index.loc[1, INDEX_MAP_COLUMNS[1]] = repeated_index.loc[
        0, INDEX_MAP_COLUMNS[1]
    ]
    repeated_index.to_csv(path, index=False)
    with pytest.raises(ConverterError, match=INDEX_MAP_COLUMNS[1]):
        read_index_map(path)


def test_an_altered_png_fails_the_round_trip_and_is_rewritten_on_a_second_pass(
    tmp_path,
):
    """The round trip refuses an altered file, and resuming does not keep it.

    The second half is the one that makes resuming safe. Skipping a file
    because it exists would preserve a truncated or altered file from an
    interrupted run, which is the failure the check is here to find rather than
    inherit.
    """
    array = _pattern(7)
    path = tmp_path / "cifar10c" / "fog" / "5" / "airplane" / "0298.png"

    assert write_png(path, array) is True
    assert write_png(path, array) is False

    with Image.open(path) as opened:
        altered = np.array(opened)
    altered[0, 0, 0] = (int(altered[0, 0, 0]) + 1) % 256
    Image.fromarray(altered, mode="RGB").save(path, format="PNG")

    with pytest.raises(RoundTripFailure, match="round trip is not identity"):
        assert_round_trip(path, array)

    assert write_png(path, array) is True
    assert_round_trip(path, array)
