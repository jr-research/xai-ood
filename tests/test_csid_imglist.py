# tests/test_csid_imglist.py
#
# The CIFAR-10-C cs-ID imglist builder -- the one expensive-to-undo artefact
# here, because once embeddings exist for a particular 2,000 photographs, a
# different 2,000 means re-extracting.
#
# Four properties are the point, not decoration:
#
#   1. exactly 75 cells
#   2. exactly 2,000 rows per cell
#   3. the SAME 2,000 ids in every cell
#   4. the draw reproduces from the recorded seed
#
# (3) is the one with teeth. A per-cell redraw satisfies (1), (2) and (4) and
# produces a file that looks entirely correct; the only thing that distinguishes
# it is set equality across cells. So the central test here is not "the builder
# works" but "a builder that redrew per cell would be caught" -- which is
# checked by constructing that exact wrong artefact by hand and asserting the
# guard rejects it. An assertion never shown to fire is not an assertion.
#
# The fixtures are small and hand-sized (three corruptions, two severities, six
# photographs) so every expected count below is countable on paper. The declared
# 15/5/2,000 numbers are checked separately against the real constants, and once
# end to end at full size.
#
# numpy / pandas / pytest only. No GPU, no embedding cache, no CIFAR-10-C.
import json

import numpy as np
import pandas as pd
import pytest

from xai_ood.csid_imglist import (
    CORRUPTIONS,
    CSID_SPLIT,
    DEFAULT_PATH_TEMPLATE,
    HELD_OUT_CORRUPTIONS,
    IMGLIST_COLUMNS,
    N_CELLS,
    N_ID_TEST,
    N_ROWS,
    N_SUBSET,
    ROWS_PER_DRAWN_IMAGE,
    SEVERITIES,
    SUBSET_SEED,
    assert_csid_grid,
    assert_draw_reproduces,
    build_csid_imglist,
    count_imglist_lines,
    csid_path,
    draw_shared_subset,
    extraction_volume_report,
    full_spectrum_mixture_weight,
    imglist_text,
    parse_imglist,
    read_id_test_imglist,
    sha256_file,
    sha256_text,
    write_csid_imglist,
)
from xai_ood.schema import CANONICAL_SORT_KEY, VALID_SEVERITIES, assert_canonical_order

CLASSES = ("airplane", "cat", "dog", "ship")

SMALL_CORRUPTIONS = ("gaussian_noise", "fog", "pixelate")
SMALL_SEVERITIES = (1, 3)


def id_test_frame(n=6):
    """A miniature ``test_cifar10.txt``: ``cifar10/test/<class>/<i>.png <label>``."""
    rows = [
        (f"cifar10/test/{CLASSES[i % len(CLASSES)]}/{i}.png", i % len(CLASSES))
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=["path", "label"])


def small_build(n=6, n_subset=4, seed=SUBSET_SEED, **kwargs):
    kwargs.setdefault("corruptions", SMALL_CORRUPTIONS)
    kwargs.setdefault("severities", SMALL_SEVERITIES)
    return build_csid_imglist(id_test_frame(n), seed=seed, n_subset=n_subset, **kwargs)


def write_id_test(tmp_path, n=6, name="test_cifar10.txt"):
    path = tmp_path / name
    path.write_text(imglist_text(id_test_frame(n)))
    return path


# --------------------------------------------------------------------------- #
# The declared constants
# --------------------------------------------------------------------------- #


def test_the_committed_numbers_are_mutually_consistent():
    """15 x 5 = 75, 75 x 2,000 = 150,000, 1 + 75 = 76.

    All five constants are decided upstream and none is derived from another in
    the source, so an edit to one could silently disagree with the rest. This is
    the arithmetic that ties them together.
    """
    assert len(CORRUPTIONS) == 15
    assert len(SEVERITIES) == 5
    assert len(CORRUPTIONS) * len(SEVERITIES) == N_CELLS == 75
    assert N_CELLS * N_SUBSET == N_ROWS == 150_000
    assert 1 + N_CELLS == ROWS_PER_DRAWN_IMAGE == 76


def test_the_standard_fifteen_and_the_held_out_four_are_disjoint_and_cover_nineteen():
    assert len(set(CORRUPTIONS)) == 15
    assert len(set(HELD_OUT_CORRUPTIONS)) == 4
    assert not set(CORRUPTIONS) & set(HELD_OUT_CORRUPTIONS)
    assert len(set(CORRUPTIONS) | set(HELD_OUT_CORRUPTIONS)) == 19


def test_severities_come_from_the_schema_so_there_is_one_definition():
    assert SEVERITIES == VALID_SEVERITIES
    assert 0 not in SEVERITIES  # 0 is the null sentinel in the sort key


def test_id_test_is_nine_thousand_not_ten():
    """OpenOOD reserves 1,000 of the 10,000 for val."""
    assert N_ID_TEST == 9_000


def test_full_spectrum_mixture_weight_is_nine_over_one_fifty_nine():
    w = full_spectrum_mixture_weight()
    assert w == pytest.approx(9_000 / 159_000)
    assert round(w, 3) == 0.057
    # And specifically not 0.0625, which is what using 10,000 would give.
    assert w != pytest.approx(0.0625)


# --------------------------------------------------------------------------- #
# Assertion 3, the one with teeth: the same ids in every cell
# --------------------------------------------------------------------------- #


def test_every_cell_holds_the_identical_id_set():
    frame = small_build()
    sets = {
        cell: frozenset(group["image_id"])
        for cell, group in frame.groupby(["corruption", "severity"])
    }
    assert len(sets) == len(SMALL_CORRUPTIONS) * len(SMALL_SEVERITIES)
    assert len(set(sets.values())) == 1, "cells hold different id sets"


def per_cell_redraw(n=6, n_subset=4, seed=SUBSET_SEED):
    """The wrong artefact, built deliberately: an independent draw per cell.

    This is what "a fixed random subset of 2,000 images per cell" reads as if
    you do not also say the same 2,000 recur. It has the right cell count, the
    right rows per cell, and even reproduces from its own seed -- so it is only
    distinguishable by set equality across cells.
    """
    id_test = id_test_frame(n)
    records = []
    for i, (corruption, severity) in enumerate(
        [(c, s) for c in SMALL_CORRUPTIONS for s in SMALL_SEVERITIES]
    ):
        drawn = draw_shared_subset(list(id_test["path"]), seed=seed + i, n=n_subset)
        for image_id in drawn:
            records.append(
                {
                    "split": CSID_SPLIT,
                    "image_id": image_id,
                    "corruption": corruption,
                    "severity": severity,
                    "label": 0,
                    "path": csid_path(image_id, corruption, severity),
                }
            )
    return pd.DataFrame.from_records(records, columns=list(IMGLIST_COLUMNS))


def test_the_per_cell_redraw_fixture_really_does_differ_between_cells():
    """Guards the guard: if the fixture happened to draw the same ids every
    time, the test below would pass vacuously against a broken assertion."""
    frame = per_cell_redraw()
    sets = {
        cell: frozenset(group["image_id"])
        for cell, group in frame.groupby(["corruption", "severity"])
    }
    assert len(set(sets.values())) > 1


def test_a_per_cell_redraw_is_rejected():
    """The failure this whole module exists to prevent."""
    frame = per_cell_redraw()
    with pytest.raises(ValueError, match="different set of image ids"):
        assert_csid_grid(
            frame,
            corruptions=SMALL_CORRUPTIONS,
            severities=SMALL_SEVERITIES,
            n_subset=4,
        )


def test_a_per_cell_redraw_passes_every_other_check_it_should():
    """Why set equality is not redundant with the cell and size assertions.

    The wrong artefact has exactly the right number of cells and exactly the
    right number of rows in each. Those two checks cannot tell it apart from the
    correct one; only the third can.
    """
    frame = per_cell_redraw()
    cells = frame.groupby(["corruption", "severity"]).size()
    assert len(cells) == len(SMALL_CORRUPTIONS) * len(SMALL_SEVERITIES)
    assert set(cells) == {4}
    assert len(frame) == len(cells) * 4


def test_one_swapped_id_in_one_cell_is_caught():
    """Not just a wholesale redraw -- a single wrong row in one of 75 cells."""
    frame = small_build(n=6, n_subset=4)
    present = set(frame["image_id"])
    outsider = next(p for p in id_test_frame(6)["path"] if p not in present)
    corrupted = frame.copy()
    corrupted.loc[corrupted.index[0], "image_id"] = outsider
    with pytest.raises(ValueError, match="different set of image ids"):
        assert_csid_grid(
            corrupted,
            corruptions=SMALL_CORRUPTIONS,
            severities=SMALL_SEVERITIES,
            n_subset=4,
        )


def test_a_duplicated_id_inside_one_cell_is_caught():
    """Right row count, wrong distinct count: an id repeated within a cell."""
    frame = small_build(n=6, n_subset=4)
    first_cell = (frame["corruption"] == "fog") & (frame["severity"] == 1)
    idx = list(frame.index[first_cell])
    corrupted = frame.copy()
    corrupted.loc[idx[1], "image_id"] = corrupted.loc[idx[0], "image_id"]
    with pytest.raises(ValueError, match="distinct image ids|different set"):
        assert_csid_grid(
            corrupted,
            corruptions=SMALL_CORRUPTIONS,
            severities=SMALL_SEVERITIES,
            n_subset=4,
        )


# --------------------------------------------------------------------------- #
# Assertions 1 and 2: cell count, rows per cell
# --------------------------------------------------------------------------- #


def test_a_missing_cell_is_caught():
    frame = small_build()
    dropped = frame[~((frame["corruption"] == "fog") & (frame["severity"] == 3))]
    with pytest.raises(ValueError, match="cell set mismatch"):
        assert_csid_grid(
            dropped,
            corruptions=SMALL_CORRUPTIONS,
            severities=SMALL_SEVERITIES,
            n_subset=4,
        )


def test_a_missing_row_inside_one_cell_is_caught():
    frame = small_build()
    dropped = frame.drop(frame.index[0])
    with pytest.raises(ValueError, match="do not have exactly 4 rows"):
        assert_csid_grid(
            dropped,
            corruptions=SMALL_CORRUPTIONS,
            severities=SMALL_SEVERITIES,
            n_subset=4,
        )


def test_an_unexpected_cell_is_caught():
    """A severity outside the declared grid, e.g. a 19-corruption build."""
    frame = small_build()
    extra = frame[frame["severity"] == 1].copy()
    extra["severity"] = 5
    with pytest.raises(ValueError, match="cell set mismatch"):
        assert_csid_grid(
            pd.concat([frame, extra], ignore_index=True),
            corruptions=SMALL_CORRUPTIONS,
            severities=SMALL_SEVERITIES,
            n_subset=4,
        )


def test_the_held_out_four_are_refused():
    with pytest.raises(ValueError, match="held-out validation"):
        build_csid_imglist(
            id_test_frame(6),
            n_subset=4,
            corruptions=("fog", "spatter"),
            severities=(1,),
        )


def test_severity_zero_is_refused():
    """0 is the null sentinel in the canonical key, not a severity."""
    with pytest.raises(ValueError, match="outside"):
        build_csid_imglist(
            id_test_frame(6), n_subset=4, corruptions=("fog",), severities=(0, 1)
        )


# --------------------------------------------------------------------------- #
# Assertion 4: the draw reproduces from the recorded seed
# --------------------------------------------------------------------------- #


def test_building_twice_gives_an_identical_frame():
    """Build twice, compare. The literal form of the fourth assertion."""
    assert small_build().equals(small_build())


def test_the_same_seed_draws_the_same_ids_and_a_different_seed_does_not():
    ids = list(id_test_frame(50)["path"])
    a = draw_shared_subset(ids, seed=SUBSET_SEED, n=10)
    b = draw_shared_subset(ids, seed=SUBSET_SEED, n=10)
    c = draw_shared_subset(ids, seed=SUBSET_SEED + 1, n=10)
    assert a == b
    assert a != c


def test_the_draw_reproduces_against_the_source_imglist():
    frame = small_build()
    assert_draw_reproduces(frame, id_test_frame(6), seed=SUBSET_SEED, n_subset=4)


def test_a_frame_built_from_another_seed_fails_the_reproduction_check():
    frame = small_build(n=50, n_subset=10, seed=SUBSET_SEED + 1)
    with pytest.raises(ValueError, match="does not reproduce"):
        assert_draw_reproduces(
            frame, id_test_frame(50), seed=SUBSET_SEED, n_subset=10
        )


def test_the_reproduction_failure_message_does_not_read_like_a_pass():
    """The diagnostic must report the frame's own id count, not just the overlap.

    This message is only ever read when something has already gone wrong, so it
    is the one string in the module that cannot afford to be reassuring. The
    superset case is the one that bites: a frame holding *every* source id
    contains all of the redrawn ones, so the intersection is the full ``n_subset``
    and a message quoting only ``len(actual & redrawn)`` of ``n_subset`` reads
    "4 of 4 ids in common" -- i.e. it tells the reader everything matched, at the
    exact moment nothing did. That is the shape of the per-cell-redraw mutant.
    """
    id_test = id_test_frame(6)
    # A frame holding all 6 ids where the draw is 4: a strict superset, so the
    # intersection is 4 of 4 and only the total distinguishes it from a pass.
    frame = pd.DataFrame({"image_id": list(id_test["path"])})

    with pytest.raises(ValueError) as excinfo:
        assert_draw_reproduces(frame, id_test, seed=SUBSET_SEED, n_subset=4)

    message = str(excinfo.value)
    assert "6 distinct" in message, (
        f"the frame holds 6 distinct ids and the message never says so: {message}"
    )


def test_the_draw_is_without_replacement_and_the_right_size():
    ids = list(id_test_frame(50)["path"])
    drawn = draw_shared_subset(ids, seed=SUBSET_SEED, n=10)
    assert len(drawn) == 10
    assert len(set(drawn)) == 10
    assert set(drawn) <= set(ids)


def test_the_draw_is_returned_in_imglist_order():
    """Stable order, so two builds compare with == rather than needing set()."""
    ids = list(id_test_frame(50)["path"])
    drawn = draw_shared_subset(ids, seed=SUBSET_SEED, n=10)
    assert drawn == [i for i in ids if i in set(drawn)]


def test_drawing_more_than_available_raises():
    with pytest.raises(ValueError, match="without replacement"):
        draw_shared_subset(list(id_test_frame(6)["path"]), n=7)


def test_a_duplicated_source_id_raises_rather_than_drawing_a_multiset():
    with pytest.raises(ValueError, match="duplicates"):
        draw_shared_subset(["a.png", "a.png", "b.png"], n=2)


def test_the_recorded_seed_is_the_date_it_was_fixed():
    """Documents intent: chosen before any data existed, so it cannot have been
    selected against a result."""
    assert SUBSET_SEED == 20260826


# --------------------------------------------------------------------------- #
# Row identity, canonical order, and the seam with schema.py
# --------------------------------------------------------------------------- #


def test_the_frame_is_in_canonical_order_with_a_total_key():
    frame = small_build()
    assert_canonical_order(frame)
    for column in CANONICAL_SORT_KEY:
        assert column in frame.columns
        assert not frame[column].isna().any(), (
            f"{column} is null on some row; cs-ID rows carry explicit corruption "
            f"and severity so the key is total"
        )


def test_the_split_is_the_enumerated_csid_role():
    from xai_ood.schema import SPLIT_ROLES

    assert CSID_SPLIT in SPLIT_ROLES
    assert set(small_build()["split"]) == {CSID_SPLIT}


def test_image_id_is_the_source_id_test_path_not_the_corrupted_one():
    """The convention the clean ID dump has to match.

    If the clean rows used stems and these used paths, nothing would raise --
    the two would simply never join, and every cs-ID photograph would silently
    become a singleton cluster in the bootstrap.
    """
    frame = small_build()
    assert set(frame["image_id"]) <= set(id_test_frame(6)["path"])
    assert not any(id_.startswith("cifar10c/") for id_ in frame["image_id"])


def test_a_photograph_owns_one_row_per_cell():
    """The 76-rows-per-drawn-photograph structure, at fixture scale."""
    frame = small_build()
    counts = frame.groupby("image_id").size()
    assert set(counts) == {len(SMALL_CORRUPTIONS) * len(SMALL_SEVERITIES)}


def test_labels_are_inherited_from_the_source_image():
    """Corruption changes pixels, not content. The class label must carry over."""
    frame = small_build()
    expected = dict(zip(id_test_frame(6)["path"], id_test_frame(6)["label"]))
    for row in frame.itertuples(index=False):
        assert row.label == expected[row.image_id]


def test_a_photographs_rows_are_contiguous_in_canonical_order():
    """Why the key is (split, image_id, corruption, severity) and not cell-major:
    the cluster bootstrap carries a photograph's rows together."""
    frame = small_build()
    positions = frame.reset_index(drop=True).groupby("image_id").apply(
        lambda g: (g.index.max() - g.index.min() + 1) == len(g)
    )
    assert positions.all()


# --------------------------------------------------------------------------- #
# Path construction: the one invented assumption, isolated
# --------------------------------------------------------------------------- #


def test_the_default_template_expands_by_hand():
    assert (
        csid_path("cifar10/test/cat/3975.png", "fog", 3)
        == "cifar10c/fog/3/cat/3975.png"
    )


def test_a_replacement_template_changes_only_the_path_column():
    """The point of isolating the assumption: being wrong about the layout costs
    one argument, and the draw, the cells and the row order are untouched."""
    default = small_build()
    other = small_build(template="cifar10c/{corruption}_{severity}/{stem}.npy")
    assert other.loc[0, "path"].endswith(".npy")
    for column in [c for c in IMGLIST_COLUMNS if c != "path"]:
        pd.testing.assert_series_equal(default[column], other[column])


def test_every_template_field_is_available():
    path = csid_path(
        "cifar10/test/cat/3975.png",
        "fog",
        3,
        template="{corruption}|{severity}|{class_name}|{filename}|{stem}|{id_path}",
    )
    assert path == "fog|3|cat|3975.png|3975|cifar10/test/cat/3975.png"


def test_an_unknown_template_field_raises_with_the_available_ones():
    with pytest.raises(ValueError, match="unknown field"):
        csid_path("cifar10/test/cat/3975.png", "fog", 3, template="{corrupton}/{stem}")


def test_a_path_with_no_class_directory_raises():
    with pytest.raises(ValueError, match="no parent directory"):
        csid_path("3975.png", "fog", 3)


def test_the_default_template_is_flagged_as_the_assumption_it_is():
    """The template is a guess. If it ever stops being one, this test is the
    place that says so -- and the module docstring is where to say it."""
    from xai_ood import csid_imglist

    assert "{corruption}" in DEFAULT_PATH_TEMPLATE
    assert "assumption" in csid_imglist.__doc__.lower()


# --------------------------------------------------------------------------- #
# Reading OpenOOD imglists
# --------------------------------------------------------------------------- #


def test_parse_imglist_reads_path_and_label():
    frame = parse_imglist("cifar10/test/cat/3975.png 3\ncifar10/test/dog/12.png 5\n")
    assert list(frame["path"]) == ["cifar10/test/cat/3975.png", "cifar10/test/dog/12.png"]
    assert list(frame["label"]) == [3, 5]


def test_parse_imglist_skips_blank_lines_and_keeps_negative_labels():
    frame = parse_imglist("a/b.png -1\n\n  \nc/d.png 0\n")
    assert len(frame) == 2
    assert list(frame["label"]) == [-1, 0]


def test_parse_imglist_rejects_a_line_without_a_label():
    with pytest.raises(ValueError, match="not '<path> <label>'"):
        parse_imglist("cifar10/test/cat/3975.png\n")


def test_parse_imglist_rejects_a_non_integer_label():
    with pytest.raises(ValueError, match="non-integer label"):
        parse_imglist("cifar10/test/cat/3975.png cat\n")


def test_parse_imglist_rejects_duplicate_paths():
    with pytest.raises(ValueError, match="duplicate path"):
        parse_imglist("a/b.png 1\na/b.png 1\n")


def test_reading_a_ten_thousand_row_test_split_raises(tmp_path):
    """The val-leak guard. A 10,000-row file is the full CIFAR-10 test set and
    would put the 1,000 reserved val photographs into the cs-ID draw."""
    path = write_id_test(tmp_path, n=12)
    with pytest.raises(ValueError, match="reserved as val"):
        read_id_test_imglist(path, expected_rows=9_000)


def test_reading_accepts_the_expected_size(tmp_path):
    path = write_id_test(tmp_path, n=12)
    assert len(read_id_test_imglist(path, expected_rows=12)) == 12


def test_the_builder_rejects_a_frame_without_labels():
    with pytest.raises(ValueError, match="missing the 'label' column"):
        build_csid_imglist(
            id_test_frame(6)[["path"]],
            n_subset=4,
            corruptions=("fog",),
            severities=(1,),
        )


# --------------------------------------------------------------------------- #
# Writing, and the manifest
# --------------------------------------------------------------------------- #


def test_imglist_text_is_openood_format_in_frame_order():
    frame = small_build()
    lines = imglist_text(frame).splitlines()
    assert len(lines) == len(frame)
    assert lines[0] == f"{frame.loc[0, 'path']} {frame.loc[0, 'label']}"
    assert all(len(line.rsplit(None, 1)) == 2 for line in lines)


def test_imglist_text_round_trips_through_the_parser():
    frame = small_build()
    reparsed = parse_imglist(imglist_text(frame))
    assert list(reparsed["path"]) == list(frame["path"])
    assert list(reparsed["label"]) == list(frame["label"])


def test_write_produces_three_aligned_files(tmp_path):
    source = write_id_test(tmp_path, n=6)
    manifest = write_csid_imglist(
        tmp_path / "out",
        source,
        n_subset=4,
        corruptions=SMALL_CORRUPTIONS,
        severities=SMALL_SEVERITIES,
        expect_rows=None,
        expected_id_test_rows=6,
    )
    imglist = tmp_path / "out" / "test_cifar10c.txt"
    index = tmp_path / "out" / "test_cifar10c_index.csv"
    written_manifest = tmp_path / "out" / "test_cifar10c_manifest.json"
    assert imglist.exists() and index.exists() and written_manifest.exists()

    frame = pd.read_csv(index)
    assert list(frame.columns) == list(IMGLIST_COLUMNS)
    assert count_imglist_lines(imglist) == len(frame) == manifest["n_rows"]
    # Row for row, in the same order: this is what makes the extraction cache's
    # filelist.txt align with the score arrays without any reindexing.
    assert list(parse_imglist(imglist.read_text())["path"]) == list(frame["path"])
    assert json.loads(written_manifest.read_text()) == manifest


def test_the_manifest_records_the_seed_and_the_source_digest(tmp_path):
    """A seed alone does not reproduce the draw -- it is a function of the seed
    and of the source imglist's row order. Both have to be recorded."""
    source = write_id_test(tmp_path, n=6)
    manifest = write_csid_imglist(
        tmp_path / "out",
        source,
        n_subset=4,
        corruptions=SMALL_CORRUPTIONS,
        severities=SMALL_SEVERITIES,
        expect_rows=None,
        expected_id_test_rows=6,
    )
    assert manifest["seed"] == SUBSET_SEED
    assert manifest["source_imglist"]["sha256"] == sha256_file(source)
    assert manifest["source_imglist"]["n_rows"] == 6
    assert manifest["imglist_sha256"] == sha256_text(
        (tmp_path / "out" / "test_cifar10c.txt").read_text()
    )


def test_the_manifest_records_the_grid_and_the_derived_counts(tmp_path):
    source = write_id_test(tmp_path, n=6)
    manifest = write_csid_imglist(
        tmp_path / "out",
        source,
        n_subset=4,
        corruptions=SMALL_CORRUPTIONS,
        severities=SMALL_SEVERITIES,
        expect_rows=None,
        expected_id_test_rows=6,
    )
    assert manifest["n_cells"] == 6
    assert manifest["rows_per_cell"] == 4
    assert manifest["n_rows"] == 24
    assert manifest["corruptions"] == list(SMALL_CORRUPTIONS)
    assert manifest["severities"] == list(SMALL_SEVERITIES)
    assert manifest["held_out_corruptions_excluded"] == list(HELD_OUT_CORRUPTIONS)
    assert manifest["image_id_convention"] == "id_test_imglist_path"
    assert manifest["path_template"] == DEFAULT_PATH_TEMPLATE
    assert manifest["path_template_verified_against_data_root"] is False
    assert manifest["rows_per_drawn_image_full_spectrum"] == 7  # 1 + 6 cells


def test_the_manifest_n_id_test_is_measured_not_the_constant(tmp_path):
    """``n_id_test`` must describe the file that was read, not ``N_ID_TEST``.

    ``expected_id_test_rows=None`` is a documented, deliberate escape hatch, but
    taking it used to produce a manifest that silently misreported the run: the
    constant 9,000 went in regardless of what was on disk, and the two derived
    quantities beside it -- the full-spectrum mixture weight and the ID row
    count -- were computed from the constant rather than from the source.

    9,000 is right for the declared protocol, which is exactly what makes the
    constant dangerous: it looks correct until the day it is not. Everything
    else in this module is careful to measure this number rather than assert it
    (see ``read_id_test_imglist``'s val-leak error), and the manifest is the one
    artefact that has to be defensible months later.
    """
    source = write_id_test(tmp_path, n=6)
    manifest = write_csid_imglist(
        tmp_path / "out",
        source,
        n_subset=4,
        corruptions=SMALL_CORRUPTIONS,
        severities=SMALL_SEVERITIES,
        expect_rows=None,
        expected_id_test_rows=None,  # the escape hatch, deliberately taken
    )
    assert manifest["source_imglist"]["n_rows"] == 6
    assert manifest["n_id_test"] == 6, "the constant leaked into the manifest"
    # Both derived from the measured count, not from N_ID_TEST.
    assert manifest["full_spectrum_id_rows"] == 6 + 24
    assert manifest["full_spectrum_mixture_weight"] == pytest.approx(6 / 30)


def test_the_write_gate_refuses_a_non_committed_grid(tmp_path):
    """`expect_rows` defaults to 150,000, so a small build cannot reach disk by
    accident -- only by the caller saying so."""
    source = write_id_test(tmp_path, n=6)
    with pytest.raises(ValueError, match="refusing to write"):
        write_csid_imglist(
            tmp_path / "out",
            source,
            n_subset=4,
            corruptions=SMALL_CORRUPTIONS,
            severities=SMALL_SEVERITIES,
            expected_id_test_rows=6,
        )
    assert not (tmp_path / "out" / "test_cifar10c.txt").exists()


def test_path_verification_catches_a_wrong_template(tmp_path):
    """What a run gets instead of a 150,000-image extraction against paths
    that do not exist."""
    source = write_id_test(tmp_path, n=6)
    root = tmp_path / "images_classic"
    root.mkdir()
    with pytest.raises(ValueError, match="do not exist under"):
        write_csid_imglist(
            tmp_path / "out",
            source,
            n_subset=4,
            corruptions=SMALL_CORRUPTIONS,
            severities=SMALL_SEVERITIES,
            expect_rows=None,
            expected_id_test_rows=6,
            data_root=root,
        )


def test_path_verification_passes_when_the_layout_matches(tmp_path):
    source = write_id_test(tmp_path, n=6)
    root = tmp_path / "images_classic"
    frame = small_build(n=6, n_subset=4)
    for rel in frame["path"]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    manifest = write_csid_imglist(
        tmp_path / "out",
        source,
        n_subset=4,
        corruptions=SMALL_CORRUPTIONS,
        severities=SMALL_SEVERITIES,
        expect_rows=None,
        expected_id_test_rows=6,
        data_root=root,
    )
    assert manifest["path_template_verified_against_data_root"] is True
    assert manifest["data_root"] == str(root)


# --------------------------------------------------------------------------- #
# Extraction volume
# --------------------------------------------------------------------------- #


def test_line_counting_ignores_blank_lines(tmp_path):
    path = tmp_path / "test_mnist.txt"
    path.write_text("a/b.png -1\n\nc/d.png -1\n")
    assert count_imglist_lines(path) == 2


def test_extraction_volume_separates_what_still_has_to_run():
    """Total extraction volume known before extraction rather than discovered
    during it. With the far-OOD splits already cached, the only thing still to
    extract is the cs-ID grid."""
    far_ood = {"mnist": 70_000, "svhn": 26_032, "texture": 5_640, "places365": 35_195}
    report = extraction_volume_report(N_ROWS, far_ood, already_cached=far_ood)
    assert report["imglist_line_count_total"] == 136_867
    assert report["remaining_to_extract"] == N_ROWS

    # And with nothing cached, the total is everything.
    fresh = extraction_volume_report(N_ROWS, far_ood)
    assert fresh["remaining_to_extract"] == N_ROWS + 136_867


def test_digests_are_stable_and_content_sensitive(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello\n")
    assert sha256_file(path) == sha256_text("hello\n")
    assert sha256_text("hello\n") != sha256_text("hello")


# --------------------------------------------------------------------------- #
# Once, at full size
# --------------------------------------------------------------------------- #


def test_the_committed_build_is_seventy_five_cells_of_two_thousand():
    """The real thing: 9,000 ids in, 150,000 rows out, 75 cells, 2,000 each, one
    shared subset. Everything above runs on hand-sized fixtures; this is the
    only test that exercises the numbers this design actually quotes."""
    id_test = id_test_frame(N_ID_TEST)
    frame = build_csid_imglist(id_test)

    assert len(frame) == N_ROWS
    cells = frame.groupby(["corruption", "severity"]).size()
    assert len(cells) == N_CELLS
    assert set(cells) == {N_SUBSET}

    ids = set(frame["image_id"])
    assert len(ids) == N_SUBSET
    assert ids <= set(id_test["path"])
    assert set(frame.groupby("image_id").size()) == {N_CELLS}

    # Reused in every cell, not redrawn per cell.
    assert (
        len({frozenset(g["image_id"]) for _, g in frame.groupby(["corruption", "severity"])})
        == 1
    )
    assert_canonical_order(frame)


def test_the_committed_build_reproduces_from_the_recorded_seed():
    """Build twice at full size and compare."""
    id_test = id_test_frame(N_ID_TEST)
    a = build_csid_imglist(id_test)
    b = build_csid_imglist(id_test)
    assert a.equals(b)
    assert list(a["image_id"]) == list(b["image_id"])
