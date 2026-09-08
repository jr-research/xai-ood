"""Single source of truth for the results-table schema and the canonical
per-sample row order.

Two separate things live here, and they are separate on purpose:

1. **The results table** -- one row per (scorer configuration x protocol x OOD group)
   evaluation. The column list is fixed in advance, deliberately.
   Retrofitting a column across accumulated runs is miserable, so the column list
   is frozen here and every writer imports it rather than spelling it out.

2. **The canonical per-sample row order** -- the sort key
   ``(split, image_id, corruption, severity)`` that every per-sample score dump
   must be in. This is what makes shared bootstrap resamples possible at all:
   one index set per replicate is applied to every scorer's score array, which is
   only meaningful if row *i* refers to the same image in every array.

   ``corruption`` and ``severity`` are **null-valued, not absent**, for rows that
   are not covariate-shifted ID. Keeping the columns present with nulls is what
   makes the key total: every row has all four components, so any two rows are
   comparable. The comparator maps the nulls to sentinels that sort before any
   real value, so clean ID rows for an image precede its corrupted rows.

Scoring convention reminder (enforced by ``tests/test_sign_convention_and_naive_baseline.py``):
every scorer under ``xai_ood.methods`` returns **higher = more OOD**. One adapter
flips OpenOOD's confidence at the seam; nothing else in the codebase negates.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "RESULTS_COLUMNS",
    "RESULTS_DTYPES",
    "PROTOCOLS",
    "SPLIT_ROLES",
    "CANONICAL_SORT_KEY",
    "NULL_CORRUPTION",
    "NULL_SEVERITY",
    "VALID_SEVERITIES",
    "empty_results_frame",
    "encode_hyperparams",
    "decode_hyperparams",
    "validate_results_frame",
    "canonical_order_key",
    "canonical_sort",
    "assert_canonical_order",
    "assert_aligned",
]


# --------------------------------------------------------------------------- #
# 1. Results table
# --------------------------------------------------------------------------- #

#: Frozen column list, in order. Do not extend
#: without a deliberate migration of every table already written.
RESULTS_COLUMNS: tuple[str, ...] = (
    "dataset",
    "split_role",
    "scorer",
    "variant",
    "hyperparams",
    "protocol",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "fpr95",
    "seed",
    "repo_commit",
    "openood_commit",
    "timestamp",
)

#: pandas dtypes for the frozen columns. Nullable extension dtypes throughout, so
#: "not yet computed" is representable without an in-band sentinel like -1.
RESULTS_DTYPES: dict[str, str] = {
    "dataset": "string",
    "split_role": "string",
    "scorer": "string",
    "variant": "string",
    "hyperparams": "string",  # canonical JSON, see encode_hyperparams
    "protocol": "string",
    "auroc": "Float64",
    "auroc_ci_low": "Float64",
    "auroc_ci_high": "Float64",
    "fpr95": "Float64",
    "seed": "Int64",
    "repo_commit": "string",
    "openood_commit": "string",
    "timestamp": "string",  # ISO 8601, UTC, second resolution
}

#: The two protocols. ``standard`` excludes cs-ID entirely; ``full_spectrum``
#: counts cs-ID as ID. Per-scorer degradation between them is the covariate-shift
#: result. AUROC(ID vs cs-ID), if reported at all, is a diagnostic with a target
#: near 0.5 and stays out of the main comparison table.
PROTOCOLS: tuple[str, ...] = ("standard", "full_spectrum")

#: Role a split plays in an evaluation. ASSUMPTION: this vocabulary is not fixed
#: externally, only the column name is. It is enumerated here so
#: a typo fails loudly rather than silently forking a results row.
SPLIT_ROLES: tuple[str, ...] = (
    "id_train",
    "id_val",
    "id_test",
    "csid",
    "near_ood",
    "far_ood",
)


def encode_hyperparams(params: Mapping[str, Any] | None) -> str:
    """Serialise a hyperparameter dict to canonical JSON.

    Sorted keys and no whitespace, so two runs with the same settings produce
    byte-identical strings and can be grouped or deduplicated by string equality.
    """
    return json.dumps(dict(params or {}), sort_keys=True, separators=(",", ":"))


def decode_hyperparams(blob: str | None) -> dict[str, Any]:
    """Inverse of ``encode_hyperparams``."""
    if blob is None or blob == "" or (isinstance(blob, float) and pd.isna(blob)):
        return {}
    return json.loads(blob)


def empty_results_frame() -> pd.DataFrame:
    """An empty results table with the frozen columns and dtypes."""
    return pd.DataFrame(
        {col: pd.Series(dtype=RESULTS_DTYPES[col]) for col in RESULTS_COLUMNS}
    )


def validate_results_frame(df: pd.DataFrame, *, name: str = "results") -> None:
    """Raise if ``df`` does not conform to the frozen results schema.

    Checks column set and order, enumerated values for ``protocol`` and
    ``split_role``, AUROC/FPR ranges on the 0-100 scale, and CI ordering.
    """
    actual = tuple(df.columns)
    if actual != RESULTS_COLUMNS:
        missing = [c for c in RESULTS_COLUMNS if c not in actual]
        extra = [c for c in actual if c not in RESULTS_COLUMNS]
        raise ValueError(
            f"{name}: columns do not match the frozen schema.\n"
            f"  expected: {list(RESULTS_COLUMNS)}\n"
            f"  actual:   {list(actual)}\n"
            f"  missing:  {missing}\n"
            f"  extra:    {extra}"
        )
    if df.empty:
        return

    bad_protocol = set(df["protocol"].dropna()) - set(PROTOCOLS)
    if bad_protocol:
        raise ValueError(f"{name}: unknown protocol(s) {sorted(bad_protocol)}; expected {list(PROTOCOLS)}")

    bad_role = set(df["split_role"].dropna()) - set(SPLIT_ROLES)
    if bad_role:
        raise ValueError(f"{name}: unknown split_role(s) {sorted(bad_role)}; expected {list(SPLIT_ROLES)}")

    # AUROC is on the 0-100 scale project-wide (see xai_ood.visualization.style).
    # A value in [0, 1] is almost certainly a scale mistake, not a real AUROC of
    # 0.98 percent, so catch it here rather than in a figure axis six weeks later.
    for col in ("auroc", "auroc_ci_low", "auroc_ci_high", "fpr95"):
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        if (vals < 0).any() or (vals > 100).any():
            raise ValueError(f"{name}: {col} outside [0, 100]; AUROC/FPR are on the 0-100 scale")
        if ((vals > 0) & (vals <= 1)).all():
            raise ValueError(
                f"{name}: every non-zero {col} lies in (0, 1]. This is the 0-1-vs-0-100 "
                f"scale mistake; multiply by 100 or fix the writer."
            )

    ordered = df.dropna(subset=["auroc_ci_low", "auroc_ci_high"])
    if (ordered["auroc_ci_low"] > ordered["auroc_ci_high"]).any():
        raise ValueError(f"{name}: auroc_ci_low exceeds auroc_ci_high in at least one row")


# --------------------------------------------------------------------------- #
# 2. Canonical per-sample row order
# --------------------------------------------------------------------------- #

#: The sort key, in precedence order. Written once, here.
CANONICAL_SORT_KEY: tuple[str, ...] = ("split", "image_id", "corruption", "severity")

#: Sentinels used *only inside the comparator*, never written into a dump. They
#: sort before every real corruption name and severity, so an image's clean row
#: precedes its 75 corrupted rows.
NULL_CORRUPTION: str = ""
NULL_SEVERITY: int = 0

#: CIFAR-10-C severities. 0 is the null sentinel and is not a real severity.
VALID_SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)


def canonical_order_key(df: pd.DataFrame, *, name: str = "frame") -> pd.DataFrame:
    """Return the four sort-key columns, null-filled with sentinels.

    This is the comparator, defined exactly once. Both ``canonical_sort`` and
    ``assert_canonical_order`` go through it, so a sorter and an asserter can
    never drift apart.
    """
    missing = [c for c in CANONICAL_SORT_KEY if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing sort-key column(s) {missing}. "
            f"corruption and severity must be present and null-valued for "
            f"non-cs-ID rows, not omitted, or the key is not total."
        )

    # The first two key components are never null. corruption and severity are
    # legitimately null on non-cs-ID rows and get the half-null check below;
    # these two are not, and filling them would manufacture a legitimate-looking
    # key that sorts first and passes every downstream check.
    for col in ("split", "image_id"):
        if df[col].isna().any():
            raise ValueError(
                f"{name}: {int(df[col].isna().sum())} row(s) have a null {col}; "
                f"the first two key components are never null. This is an "
                f"imglist-builder bug, not something to fill in."
            )

    key = pd.DataFrame(index=df.index)
    key["split"] = df["split"].astype("string")
    key["image_id"] = df["image_id"].astype("string")
    key["corruption"] = df["corruption"].astype("string").fillna(NULL_CORRUPTION)
    key["severity"] = (
        pd.to_numeric(df["severity"], errors="coerce")
        .astype("Float64")
        .fillna(NULL_SEVERITY)
        .astype("int64")
    )

    bad_sev = set(key["severity"].unique()) - {NULL_SEVERITY, *VALID_SEVERITIES}
    if bad_sev:
        raise ValueError(f"{name}: severity values {sorted(bad_sev)} outside {{null}} u {list(VALID_SEVERITIES)}")

    # corruption and severity are null together or not at all. A row with one but
    # not the other is a builder bug, and would sort somewhere arbitrary.
    half_null = (key["corruption"] == NULL_CORRUPTION) ^ (key["severity"] == NULL_SEVERITY)
    if half_null.any():
        n = int(half_null.sum())
        raise ValueError(
            f"{name}: {n} row(s) have exactly one of (corruption, severity) null. "
            f"They must be null together (non-cs-ID) or both populated (cs-ID)."
        )
    return key


def canonical_sort(df: pd.DataFrame, *, name: str = "frame") -> pd.DataFrame:
    """Return ``df`` sorted into canonical order, with a fresh RangeIndex.

    Stable sort, so a caller that has already imposed a secondary order on ties
    keeps it. There should be no ties: the key is unique per row, and
    ``assert_canonical_order`` checks that.
    """
    if df.index.has_duplicates:
        raise ValueError(
            f"{name}: the frame's pandas index has duplicate labels "
            f"(pd.concat without ignore_index=True?). Reset it first. "
            f"df.loc with a duplicate-label list returns more rows than it was "
            f"asked for, so the failure would surface downstream as a spurious "
            f"duplicate-sort-key error with the wrong row count."
        )
    key = canonical_order_key(df, name=name)
    order = key.sort_values(list(CANONICAL_SORT_KEY), kind="stable").index
    return df.loc[order].reset_index(drop=True)


def assert_canonical_order(df: pd.DataFrame, *, name: str = "frame") -> pd.DataFrame:
    """Assert ``df`` is in canonical order with unique keys; return it unchanged.

    Call this at the end of **every** score-dump function. It is cheap, and the
    failure it prevents is silent: a misaligned pair of score arrays produces
    paired differences that are wrong without ever raising.

    Returns the frame so it can be used inline: ``return assert_canonical_order(df)``.
    """
    key = canonical_order_key(df, name=name)

    dup = key.duplicated(subset=list(CANONICAL_SORT_KEY), keep=False)
    if dup.any():
        offenders = key.loc[dup].head(5)
        raise ValueError(
            f"{name}: {int(dup.sum())} row(s) share a sort key, so the order is not "
            f"total and resamples cannot be shared. First few:\n{offenders.to_string()}"
        )

    # Compare *positions*, not index labels. Comparing the two Index objects
    # elementwise compares labels, and a frame with duplicate labels (which
    # pd.concat without ignore_index=True produces every time) can be genuinely
    # mis-ordered while its label sequence survives the sort unchanged.
    positions = key.reset_index(drop=True)
    expected = list(positions.sort_values(list(CANONICAL_SORT_KEY), kind="stable").index)
    if expected != list(range(len(positions))):
        first = next(i for i, pos in enumerate(expected) if i != pos)
        raise ValueError(
            f"{name}: rows are not in canonical order "
            f"{CANONICAL_SORT_KEY}; first violation at positional row {first}. "
            f"Run xai_ood.schema.canonical_sort before dumping."
        )
    return df


def assert_aligned(
    frames: Iterable[pd.DataFrame],
    *,
    names: Sequence[str] | None = None,
) -> None:
    """Assert several per-sample frames are canonically ordered *and identical* in key.

    This is the precondition for applying one bootstrap index set to every
    scorer's score array. Checking order per-frame is not enough: two frames can
    each be sorted correctly and still cover different row sets.
    """
    frames = list(frames)
    names = list(names) if names is not None else [f"frame[{i}]" for i in range(len(frames))]
    if not frames:
        return
    keys = [canonical_order_key(assert_canonical_order(f, name=n), name=n) for f, n in zip(frames, names)]
    reference = keys[0]
    for other, other_name in zip(keys[1:], names[1:]):
        if len(other) != len(reference):
            raise ValueError(
                f"{other_name}: {len(other)} rows, but {names[0]} has {len(reference)}"
            )
        if not reference.reset_index(drop=True).equals(other.reset_index(drop=True)):
            raise ValueError(
                f"{other_name}: sort keys differ from {names[0]} despite both being "
                f"canonically ordered. A shared resample index would silently misalign."
            )
