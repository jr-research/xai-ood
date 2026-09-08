"""What gets resampled: the clusters and the plan.

This module is the bottom of the package's three-layer stack::

    plan.py   <-  run.py   <-  tables.py

and imports nothing from the other two. It is deliberately the part that can be
read on its own to check that the resampling unit is the photograph, which is
the whole reason the module exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..csid_imglist import CSID_SPLIT
from ..schema import CANONICAL_SORT_KEY, PROTOCOLS, assert_canonical_order

__all__ = [
    "ID_TEST_SPLIT",
    "CSID_SPLIT",
    "PROTOCOL_ID_SPLITS",
    "DEFAULT_OOD_GROUPS",
    "ClusterIndex",
    "BootstrapPlan",
    "build_cluster_index",
    "build_plan",
]


# --------------------------------------------------------------------------- #
# Fixed parameters. None of these is a knob.
# --------------------------------------------------------------------------- #

#: ``split`` value of the clean ID-test rows. In ``xai_ood.schema.SPLIT_ROLES``;
#: a test asserts that rather than trusting it.
ID_TEST_SPLIT: str = "id_test"

#: Which ``split`` values count as in-distribution under each protocol. The
#: standard protocol excludes cs-ID entirely; full-spectrum counts it as ID.
#: This mapping *is* the protocol definition -- there is no other place where a
#: protocol means something.
PROTOCOL_ID_SPLITS: dict[str, tuple[str, ...]] = {
    "standard": (ID_TEST_SPLIT,),
    "full_spectrum": (ID_TEST_SPLIT, CSID_SPLIT),
}

#: The OOD groups, and the datasets each is composed of. Group composition is
#: fixed across replicates precisely because each dataset is resampled within
#: itself, so this mapping is load-bearing rather than presentational. Names
#: are ``split_role`` values from the frozen schema.
DEFAULT_OOD_GROUPS: dict[str, tuple[str, ...]] = {
    "near_ood": ("cifar100", "tin"),
    "far_ood": ("mnist", "svhn", "texture", "places365"),
}


# --------------------------------------------------------------------------- #
# 1. Clusters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClusterIndex:
    """Row positions grouped by resampling unit, in compressed form.

    ``positions`` holds every row this universe covers, grouped so that cluster
    *i* owns ``positions[starts[i] : starts[i] + counts[i]]``. Drawing a cluster
    means taking that whole slice, which is the cluster bootstrap in one
    sentence and the reason cluster sizes never need a correction.

    A row-level bootstrap is the degenerate case of this structure where every
    cluster holds exactly one row. That is not a code path here -- it is what
    the *data* would look like if the clustering key were wrong, which is
    exactly how the bug arises in practice (see ``build_plan``), and it is
    how ``tests/test_bootstrap.py`` builds the mutant it kills.
    """

    name: str
    labels: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    positions: np.ndarray

    @property
    def n_clusters(self) -> int:
        return int(self.labels.size)

    @property
    def n_rows(self) -> int:
        return int(self.positions.size)

    def gather(self, drawn: np.ndarray) -> np.ndarray:
        """Row positions owned by the drawn clusters, concatenated.

        ``drawn`` holds cluster *indices* (with repeats -- that is the "with
        replacement" part), so a cluster drawn twice contributes its rows twice.
        Vectorised: the alternative, a Python loop over 9,000 clusters per
        replicate, is nine million slice operations at 1,000 replicates and
        forty-five million at 5,000.
        """
        drawn = np.asarray(drawn, dtype=np.int64)
        counts = self.counts[drawn]
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        out_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        # For each output slot, the offset of its source row: the cluster's
        # start minus where that cluster begins in the output, broadcast over
        # the cluster's length, plus a running arange.
        base = np.repeat(self.starts[drawn] - out_starts, counts)
        return self.positions[base + np.arange(total, dtype=np.int64)]


def build_cluster_index(
    row_labels: Sequence[Any] | np.ndarray | pd.Series,
    row_positions: np.ndarray,
    *,
    universe: Sequence[Any] | np.ndarray,
    name: str,
    allow_empty_clusters: bool = False,
) -> ClusterIndex:
    """Group ``row_positions`` by ``row_labels`` against a fixed ``universe``.

    ``universe`` fixes both the set of clusters and their order, which is what
    lets one ``drawn`` array of cluster indices be applied to several
    ClusterIndexes over the same units -- the standard and full-spectrum ID
    indexes, which must be drawn together so the degradation between protocols
    is paired within a replicate.

    A label outside ``universe`` raises. That is the load-bearing check: it is
    the only thing standing between a cs-ID ``image_id`` convention mismatch and
    a silently row-level bootstrap.
    """
    labels = pd.Index(np.asarray(universe))
    if labels.has_duplicates:
        raise ValueError(f"{name}: the cluster universe has duplicate labels")
    positions = np.asarray(row_positions, dtype=np.int64)
    codes = labels.get_indexer(pd.Index(np.asarray(row_labels)))
    if len(codes) != len(positions):
        raise ValueError(
            f"{name}: {len(codes)} labels but {len(positions)} row positions"
        )

    unknown = codes < 0
    if unknown.any():
        offenders = sorted({str(v) for v in np.asarray(row_labels)[unknown]})[:5]
        raise ValueError(
            f"{name}: {int(unknown.sum())} row(s) carry an image_id that is not "
            f"in the cluster universe; first few {offenders}.\n"
            f"This is the failure the cs-ID imglist builder warns about: if the cs-ID "
            f"rows and the clean ID-test rows disagree about what an image_id "
            f"is (a path on one side, a filename stem on the other), the two "
            f"never join, every cs-ID photograph becomes its own cluster, and "
            f"the bootstrap silently degenerates to the row-level procedure "
            f"this module exists to avoid -- shrinking the full-spectrum "
            f"interval by up to about sqrt(76). Nothing else would raise."
        )

    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(labels)).astype(np.int64)
    if not allow_empty_clusters and (counts == 0).any():
        empty = [str(labels[i]) for i in np.flatnonzero(counts == 0)[:5]]
        raise ValueError(
            f"{name}: {int((counts == 0).sum())} cluster(s) own no rows; first "
            f"few {empty}. A photograph in the universe with no rows would be "
            f"drawn and contribute nothing, which is not the same procedure."
        )
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    return ClusterIndex(
        name=name,
        labels=labels.to_numpy(),
        starts=starts,
        counts=counts,
        positions=positions[order],
    )

# --------------------------------------------------------------------------- #
# 2. The plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BootstrapPlan:
    """Everything about *what gets resampled*, decided once and reused.

    Separated from ``run_bootstrap`` on purpose: the plan is the part that
    encodes the procedure, it is cheap to inspect, and it is what has to be read
    to check that the unit is the photograph. The run is then a loop.
    """

    id_labels: np.ndarray
    id_by_protocol: Mapping[str, ClusterIndex]
    ood_by_dataset: Mapping[str, ClusterIndex]
    ood_groups: Mapping[str, tuple[str, ...]]
    n_rows: int
    key_digest: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Every protocol's ID index must be over the *same* cluster universe,
        # in the same order. One ``drawn`` array of cluster indices is applied
        # to all of them, so a protocol with a different universe would be
        # indexed by numbers meaning something else. The failure is silent: the
        # indices are in range either way, so it produces a plausible wrong
        # variance rather than an error.
        for protocol, index in self.id_by_protocol.items():
            if index.n_clusters != self.id_labels.size:
                raise ValueError(
                    f"protocol {protocol!r} has {index.n_clusters} clusters but "
                    f"the plan's universe has {self.id_labels.size}. Both "
                    f"protocols must be indexed over one universe of "
                    f"photographs, or the shared draw means different things "
                    f"on the two legs."
                )
            if not np.array_equal(index.labels, self.id_labels):
                raise ValueError(
                    f"protocol {protocol!r} is indexed over a different cluster "
                    f"universe from the plan's, in a different order. The same "
                    f"drawn cluster index would refer to two different "
                    f"photographs on the two legs."
                )

    @property
    def n_id_clusters(self) -> int:
        return int(self.id_labels.size)

    @property
    def protocols(self) -> tuple[str, ...]:
        return tuple(self.id_by_protocol)

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self.ood_groups)

    def group_positions(self) -> dict[str, np.ndarray]:
        """Every row of each OOD group, unresampled. Used for point estimates."""
        return {
            group: np.concatenate(
                [self.ood_by_dataset[d].positions for d in datasets]
            )
            for group, datasets in self.ood_groups.items()
        }


def _key_digest(index: pd.DataFrame) -> int:
    """Content digest of the canonical key columns.

    Cheap insurance that the plan and the frames handed to ``run_bootstrap``
    describe the same rows. ``pd.util.hash_pandas_object`` is deterministic
    across processes, unlike builtin ``hash`` on strings, which matters because
    a plan may be built in one process and used in another.
    """
    keys = index.loc[:, list(CANONICAL_SORT_KEY)].astype("string").fillna("")
    return int(pd.util.hash_pandas_object(keys, index=False).sum())


def build_plan(
    index: pd.DataFrame,
    *,
    ood_groups: Mapping[str, Sequence[str]] | None = None,
    id_split: str = ID_TEST_SPLIT,
    csid_split: str = CSID_SPLIT,
    protocols: Sequence[str] = PROTOCOLS,
    allow_missing_csid: bool = False,
    allow_unused_splits: bool = False,
    name: str = "plan",
) -> BootstrapPlan:
    """Build the resampling plan from a canonically-ordered per-sample index.

    ``index`` is the key half of any scorer's score frame: the four columns
    ``(split, image_id, corruption, severity)``, in canonical order, covering
    every row of the evaluation -- ID test, cs-ID, and every OOD dataset.

    **The ID cluster universe is the ID-test photographs**, and only those. The
    cs-ID rows are attached to them by ``image_id``; they never create clusters
    of their own. That asymmetry is the procedure itself, and it is also the
    guard: a cs-ID row whose ``image_id`` does not name an ID-test
    photograph raises, rather than quietly becoming a singleton cluster.

    Each OOD dataset gets its own universe, resampled within itself.
    OOD rows have no duplication so image and row coincide there, but they are
    still expressed as clusters rather than as raw rows -- one mechanism, so a
    hypothetical duplicated OOD set would be handled correctly instead of
    needing a second code path.
    """
    missing = [c for c in CANONICAL_SORT_KEY if c not in index.columns]
    if missing:
        raise ValueError(f"{name}: index is missing key column(s) {missing}")
    groups = {k: tuple(v) for k, v in (ood_groups or DEFAULT_OOD_GROUPS).items()}
    if not groups:
        raise ValueError(f"{name}: no OOD groups; there is nothing to score against")

    protocols = tuple(protocols)
    unknown_protocols = [p for p in protocols if p not in PROTOCOL_ID_SPLITS]
    if unknown_protocols:
        raise ValueError(
            f"{name}: unknown protocol(s) {unknown_protocols}; "
            f"expected {sorted(PROTOCOL_ID_SPLITS)}"
        )

    split = index["split"].astype("string").to_numpy()
    image_id = index["image_id"].astype("string").to_numpy()
    positions = np.arange(len(index), dtype=np.int64)

    id_mask = split == id_split
    csid_mask = split == csid_split
    if not id_mask.any():
        raise ValueError(
            f"{name}: no rows with split == {id_split!r}. The ID-test "
            f"photographs are the cluster universe; without them there is "
            f"nothing to resample."
        )

    universe = image_id[id_mask]
    if len(set(universe)) != len(universe):
        raise ValueError(
            f"{name}: the {id_split!r} rows repeat an image_id, so a photograph "
            f"would be two clusters. Each photograph has exactly one clean row."
        )

    # Order, last of the structural preconditions and the one with no symptom.
    # ``positions`` above is positional, so a plan built from an index in a
    # different order than the score frames points every cluster at the wrong
    # rows. ``key_digest`` cannot catch it: it sums per-row hashes, and a sum is
    # permutation-invariant. Placed after the repeated-image_id guard so that
    # guard keeps its own message, which names the actual mistake.
    assert_canonical_order(index, name=name)

    n_csid = int(csid_mask.sum())
    if n_csid == 0 and "full_spectrum" in protocols and not allow_missing_csid:
        raise ValueError(
            f"{name}: no rows with split == {csid_split!r}, so the full-spectrum "
            f"protocol would be identical to the standard one and every "
            f"degradation would come out as exactly zero -- a result that looks "
            f"like a finding and is a missing input. Pass "
            f"allow_missing_csid=True for a deliberately standard-only fixture."
        )

    id_by_protocol: dict[str, ClusterIndex] = {}
    for protocol in protocols:
        wanted = PROTOCOL_ID_SPLITS[protocol]
        mask = np.isin(split, list(wanted))
        id_by_protocol[protocol] = build_cluster_index(
            image_id[mask],
            positions[mask],
            universe=universe,
            name=f"{name}:{protocol}",
        )

    ood_by_dataset: dict[str, ClusterIndex] = {}
    for group, datasets in groups.items():
        if not datasets:
            raise ValueError(f"{name}: OOD group {group!r} names no datasets")
        for dataset in datasets:
            if dataset in ood_by_dataset:
                raise ValueError(
                    f"{name}: dataset {dataset!r} appears in more than one OOD "
                    f"group; its rows would be resampled twice and the groups "
                    f"would not be independent."
                )
            mask = split == dataset
            if not mask.any():
                raise ValueError(
                    f"{name}: OOD group {group!r} names dataset {dataset!r}, "
                    f"which has no rows. Present splits: {sorted(set(split))}"
                )
            ood_by_dataset[dataset] = build_cluster_index(
                image_id[mask],
                positions[mask],
                universe=pd.unique(image_id[mask]),
                name=f"{name}:{dataset}",
            )

    accounted = set([id_split, csid_split]) | set(ood_by_dataset)
    unused = sorted(set(split) - accounted)
    if unused and not allow_unused_splits:
        raise ValueError(
            f"{name}: split(s) {unused} appear in the index but belong to no "
            f"protocol and no OOD group, so their rows would be silently "
            f"dropped. Forgetting one far-OOD dataset changes the far-OOD "
            f"average without changing anything visible. Add them to "
            f"ood_groups, or pass allow_unused_splits=True."
        )

    diagnostics = {
        "resampling_unit": "source_photograph",
        "n_id_clusters": int(len(universe)),
        "n_id_test_rows": int(id_mask.sum()),
        "n_csid_rows": n_csid,
        "n_csid_photographs": int(len(set(image_id[csid_mask]))) if n_csid else 0,
        "rows_per_protocol": {
            p: ci.n_rows for p, ci in id_by_protocol.items()
        },
        "cluster_size_range": {
            p: [int(ci.counts.min()), int(ci.counts.max())]
            for p, ci in id_by_protocol.items()
        },
        "ood_rows": {d: ci.n_rows for d, ci in ood_by_dataset.items()},
        "ood_groups": {g: list(d) for g, d in groups.items()},
        "unused_splits": unused,
    }
    return BootstrapPlan(
        id_labels=np.asarray(universe),
        id_by_protocol=id_by_protocol,
        ood_by_dataset=ood_by_dataset,
        ood_groups=groups,
        n_rows=int(len(index)),
        key_digest=_key_digest(index),
        diagnostics=diagnostics,
    )
