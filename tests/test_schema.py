"""Tests for the frozen results schema and the canonical per-sample row order.

The canonical-order machinery is the precondition for shared bootstrap
resamples. Its failure mode is silent misalignment, so these tests exist to make
it loud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_ood.schema import (
    CANONICAL_SORT_KEY,
    NULL_SEVERITY,
    RESULTS_COLUMNS,
    VALID_SEVERITIES,
    canonical_order_key,
    assert_aligned,
    assert_canonical_order,
    canonical_sort,
    decode_hyperparams,
    empty_results_frame,
    encode_hyperparams,
    validate_results_frame,
)

CORRUPTIONS = ["gaussian_noise", "shot_noise", "defocus_blur"]


def make_per_sample_frame(n_images: int = 6, n_cs: int = 3, shuffle_seed: int | None = None) -> pd.DataFrame:
    """A per-sample frame with clean ID rows plus a cs-ID grid over a subset."""
    rows = []
    image_ids = [f"cifar10_test/{i:05d}.png" for i in range(n_images)]
    for image_id in image_ids:
        rows.append(
            {"split": "cifar10_test", "image_id": image_id, "corruption": None, "severity": None}
        )
    for image_id in image_ids[:n_cs]:
        for corruption in CORRUPTIONS:
            for severity in (1, 3, 5):
                rows.append(
                    {
                        "split": "cifar10c_csid",
                        "image_id": image_id,
                        "corruption": corruption,
                        "severity": severity,
                    }
                )
    df = pd.DataFrame(rows)
    df["score"] = np.arange(len(df), dtype=float)
    if shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Results table
# --------------------------------------------------------------------------- #


def test_empty_results_frame_matches_frozen_columns():
    df = empty_results_frame()
    assert tuple(df.columns) == RESULTS_COLUMNS
    validate_results_frame(df)


def test_validate_rejects_reordered_or_missing_columns():
    df = empty_results_frame()
    with pytest.raises(ValueError, match="frozen schema"):
        validate_results_frame(df[list(RESULTS_COLUMNS)[::-1]])
    with pytest.raises(ValueError, match="frozen schema"):
        validate_results_frame(df.drop(columns=["fpr95"]))


def _one_row(**overrides):
    row = {
        "dataset": "cifar10",
        "split_role": "far_ood",
        "scorer": "class_conditional_full",
        "variant": "unnormalized",
        "hyperparams": encode_hyperparams({"shrinkage": False}),
        "protocol": "standard",
        "auroc": 91.2,
        "auroc_ci_low": 90.1,
        "auroc_ci_high": 92.4,
        "fpr95": 38.7,
        "seed": 0,
        "repo_commit": "deadbee",
        "openood_commit": "8d44375e",
        "timestamp": "2026-08-25T12:00:00Z",
    }
    row.update(overrides)
    return pd.DataFrame([row])[list(RESULTS_COLUMNS)]


def test_validate_accepts_a_well_formed_row():
    validate_results_frame(_one_row())


def test_validate_rejects_unknown_protocol_and_split_role():
    with pytest.raises(ValueError, match="unknown protocol"):
        validate_results_frame(_one_row(protocol="fsood"))
    with pytest.raises(ValueError, match="unknown split_role"):
        validate_results_frame(_one_row(split_role="ood"))


def test_validate_catches_the_zero_to_one_scale_mistake():
    """AUROC is on 0-100 project-wide. A table written on 0-1 must not pass."""
    with pytest.raises(ValueError, match="scale mistake"):
        validate_results_frame(_one_row(auroc=0.912, auroc_ci_low=0.901, auroc_ci_high=0.924, fpr95=0.387))


def test_validate_catches_inverted_confidence_interval():
    with pytest.raises(ValueError, match="auroc_ci_low exceeds"):
        validate_results_frame(_one_row(auroc_ci_low=92.4, auroc_ci_high=90.1))


def test_hyperparams_roundtrip_and_key_order_independence():
    a = encode_hyperparams({"k": 50, "normalize": True})
    b = encode_hyperparams({"normalize": True, "k": 50})
    assert a == b, "hyperparams encoding must not depend on dict insertion order"
    assert decode_hyperparams(a) == {"k": 50, "normalize": True}
    assert decode_hyperparams(None) == {}


# --------------------------------------------------------------------------- #
# Canonical order
# --------------------------------------------------------------------------- #


def test_canonical_sort_is_idempotent_and_passes_the_assertion():
    shuffled = make_per_sample_frame(shuffle_seed=7)
    once = canonical_sort(shuffled)
    twice = canonical_sort(once)
    pd.testing.assert_frame_equal(once, twice)
    assert_canonical_order(once)


def test_assertion_rejects_an_unsorted_frame():
    shuffled = make_per_sample_frame(shuffle_seed=7)
    with pytest.raises(ValueError, match="not in canonical order"):
        assert_canonical_order(shuffled)


def test_clean_row_precedes_its_corrupted_rows():
    """The null sentinel must sort *before* real corruptions, not after.

    pandas sorts NaN last by default, which would scatter each image's clean row
    to the end of its block. The comparator overrides that deliberately.

    Note this assertion is *not* what pins the sentinel: on
    `make_per_sample_frame` the clean rows live in split `cifar10_test` and the
    corrupted rows in `cifar10c_csid`, and `"cifar10_test" < "cifar10c_csid"`
    lexically (`_` is 0x5F, `c` is 0x63). Since `split` is the first sort key,
    every clean row precedes every corrupted row whatever `NULL_CORRUPTION` is.
    The test below is the one that decides it.
    """
    df = canonical_sort(make_per_sample_frame(shuffle_seed=1))
    first_image = df["image_id"].iloc[0]
    block = df[df["image_id"] == first_image]
    assert block["corruption"].isna().iloc[0], "clean row must lead its image's block"
    assert block["corruption"].notna().iloc[1:].all()


def test_the_corruption_sentinel_is_what_orders_a_shared_split():
    """The sentinel, with the split column taken out of the argument.

    Clean and corrupted rows for one image, all in the *same* split, so
    `split` and `image_id` tie and `corruption` is the first key that separates
    them. This is the realistic shape if cs-ID rows are ever tagged by split
    role rather than by dataset, and it is the only arrangement in which the
    sentinel's value is observable.

    Mutation killed: `NULL_CORRUPTION = ""` -> `"zzzz"`, which sorts the clean
    row last within its image's block. That mutation otherwise passes the whole
    suite.
    """
    rows = [{"split": "cifar10c_csid", "image_id": "img_a", "corruption": None, "severity": None}]
    rows += [
        {"split": "cifar10c_csid", "image_id": "img_a", "corruption": c, "severity": s}
        for c in CORRUPTIONS
        for s in (1, 3, 5)
    ]
    df = canonical_sort(pd.DataFrame(rows).sample(frac=1.0, random_state=11).reset_index(drop=True))

    assert df["corruption"].isna().iloc[0], (
        "the clean row must lead even when the split column does not separate it"
    )
    assert df["corruption"].notna().iloc[1:].all()


def test_severity_is_compared_as_an_integer_not_a_string():
    """What the old lexical-ordering test claimed to check, restated as it can be.

    The previous version sorted severities and asserted `[1, 3, 5]`. That test
    could not fail: `VALID_SEVERITIES` is `(1..5)` with `0` as the sentinel and
    `canonical_order_key` *raises* on anything outside that set, and over
    `{0..5}` the lexical and numeric orderings are identical. No implementation
    could have broken it on ordering alone.

    The real property is the dtype of the comparator's severity column: an
    integer, so that if the severity vocabulary ever grows past 9 the comparison
    stays numeric. That is checkable, and it is what the docstring's "5 must not
    sort before 10-style strings" was actually about.

    Mutation killed: `.astype("int64")` -> `.astype("string")` in
    `canonical_order_key`.
    """
    key = canonical_order_key(make_per_sample_frame())
    assert key["severity"].dtype == "int64"

    # And the sentinel must stay outside the real vocabulary, or a genuine
    # severity would be indistinguishable from a null one.
    # Mutation killed: NULL_SEVERITY = 0 -> 3.
    #
    # Deliberately *not* claimed: that NULL_SEVERITY's numeric value affects
    # ordering. It does not, and `0 -> 9` still passes the whole suite. An image
    # has exactly one clean row, so the sentinel never breaks a tie --
    # `corruption` has already separated the rows by the time severity is
    # compared. The value is free apart from this non-collision requirement, and
    # 0 is a convention rather than a constraint.
    assert NULL_SEVERITY not in VALID_SEVERITIES


def test_missing_sort_key_column_is_rejected():
    """corruption/severity must be present and null-valued, never omitted."""
    df = canonical_sort(make_per_sample_frame())
    with pytest.raises(ValueError, match="missing sort-key column"):
        assert_canonical_order(df.drop(columns=["corruption"]))


def test_half_null_corruption_severity_is_rejected():
    df = canonical_sort(make_per_sample_frame())
    broken = df.copy()
    broken.loc[broken.index[-1], "severity"] = None
    with pytest.raises(ValueError, match="exactly one of"):
        assert_canonical_order(broken)


def test_out_of_range_severity_is_rejected():
    df = canonical_sort(make_per_sample_frame())
    broken = df.copy()
    broken.loc[broken.index[-1], "severity"] = 6
    with pytest.raises(ValueError, match="outside"):
        assert_canonical_order(broken)


def test_duplicate_keys_are_rejected():
    """A duplicated key makes the order non-total, so a shared resample index
    would silently refer to different rows in different scorers' arrays."""
    df = canonical_sort(make_per_sample_frame())
    doubled = canonical_sort(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="share a sort key"):
        assert_canonical_order(doubled)


def test_assert_aligned_accepts_two_scorers_over_the_same_rows():
    a = canonical_sort(make_per_sample_frame(shuffle_seed=3))
    b = canonical_sort(make_per_sample_frame(shuffle_seed=4))
    b["score"] = b["score"] * -1
    assert_aligned([a, b], names=["mds", "knn"])


def test_assert_aligned_rejects_frames_covering_different_rows():
    """Both frames are individually sorted; only the cross-check catches this."""
    a = canonical_sort(make_per_sample_frame(n_images=6, n_cs=3))
    b = canonical_sort(make_per_sample_frame(n_images=6, n_cs=2))
    assert_canonical_order(a)
    assert_canonical_order(b)
    with pytest.raises(ValueError, match="rows, but"):
        assert_aligned([a, b], names=["mds", "knn"])


def test_assert_aligned_rejects_same_length_different_ids():
    a = canonical_sort(make_per_sample_frame(n_images=6, n_cs=3))
    b = a.copy()
    b["image_id"] = b["image_id"].str.replace("cifar10_test", "cifar10_train", regex=False)
    b = canonical_sort(b)
    with pytest.raises(ValueError, match="sort keys differ"):
        assert_aligned([a, b], names=["mds", "knn"])


def test_assertion_compares_positions_not_index_labels():
    """A duplicated pandas index must not be able to hide a mis-ordered frame.

    `pd.concat([a, b])` without `ignore_index=True` produces duplicate labels
    every time. The order check used to compare the sorted `Index` against the
    frame's own `Index` elementwise, which compares *labels*: with labels
    `[0, 1, 0, 1]` a genuinely mis-ordered frame has a label sequence the sort
    leaves unchanged, so the check passed and the dump shipped mis-ordered.

    `score_frame` was never exposed -- `canonical_sort` resets the index before
    the assertion -- but the module docstring instructs every score-dump function
    to call `assert_canonical_order` directly, and a later dump that
    built its frame by concatenation would have been told it was fine.

    Mutation killed: comparing `expected == key.index` rather than positions.
    """
    ordered = pd.DataFrame(
        {
            "split": ["id_test"] * 4,
            "image_id": ["img_a", "img_d", "img_c", "img_b"],
            "corruption": [None] * 4,
            "severity": [None] * 4,
        },
        index=[0, 1, 0, 1],
    )
    with pytest.raises(ValueError, match="not in canonical order"):
        assert_canonical_order(ordered)


def test_canonical_sort_rejects_a_duplicated_pandas_index():
    """`.loc` with duplicate labels returns more rows than it was asked for.

    Four rows in, eight rows out, and the failure surfaced downstream as
    "8 row(s) share a sort key" on a 4-row input -- the wrong cause and the wrong
    count, at exactly the moment of cs-ID and bootstrap wiring it is most likely to
    happen. Name the real cause instead.

    Mutation killed: deleting the `has_duplicates` guard restores the silent
    row-duplication through `df.loc[order]`.
    """
    a = make_per_sample_frame(n_images=2, n_cs=0)
    b = make_per_sample_frame(n_images=2, n_cs=0)
    concatenated = pd.concat([a, b])  # no ignore_index=True
    assert concatenated.index.has_duplicates
    with pytest.raises(ValueError, match="duplicate labels"):
        canonical_sort(concatenated)


def test_a_null_split_or_image_id_is_rejected_rather_than_filled():
    """The first two key components are never null; do not coerce them.

    `corruption` and `severity` get a careful half-null check because they are
    legitimately null on non-cs-ID rows. `split` and `image_id` are not: a null
    there means the imglist builder is broken. Filling it with "" produced a
    legitimate-looking key `("id_test", "", None, None)` that sorted first and
    passed every check, and a single such row is invisible -- only a second one
    would trip the duplicate-key guard.

    Mutation killed: restoring `.fillna("")` on either column.
    """
    base = {
        "split": ["cifar10_test", "cifar10_test"],
        "image_id": ["img_a", "img_b"],
        "corruption": [None, None],
        "severity": [None, None],
    }
    # The null goes in row 0, where the sentinel "" sorts anyway, so the frame
    # stays canonically ordered and the *only* thing under test is whether the
    # null is rejected. Before the fix these frames passed silently.
    for col in ("split", "image_id"):
        broken = pd.DataFrame({**base, col: [None, base[col][1]]})
        with pytest.raises(ValueError, match=f"null {col}"):
            assert_canonical_order(broken)
