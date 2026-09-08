"""The cluster bootstrap, and every interval the thesis quotes.

The bootstrap-confidence-interval procedure, in seven parts. The whole module
in one sentence:

    cs-ID is corrupted copies of the same test photographs, so the bootstrap
    resamples PHOTOGRAPHS and carries all of a photograph's rows together;
    resamples are shared across every scorer; and every comparison is reported
    as the interval of the within-replicate difference rather than as CI
    overlap.

Where each part lives in this file
----------------------------------
1. **Resampling unit is the source photograph.** ``ClusterIndex`` and
   ``build_plan``. The ID universe is the 9,000 ID-test photographs; a drawn
   id carries its clean row and, under full-spectrum, every corrupted copy of it
   in the cs-ID subsample. Cluster sizes are non-uniform by construction (2,000
   photographs own 76 rows, 7,000 own 1) and need no correction, because the
   procedure draws ids and gathers whatever rows each id owns.
2. **The cs-ID subsample is one shared index set.** Built and asserted in
   ``xai_ood.csid_imglist``; consumed here. ``build_plan`` refuses a
   frame whose cs-ID ``image_id`` values do not join the ID-test ones, which is
   the cs-ID builder's named silent failure (paths on one side, stems on the other:
   nothing raises, every cs-ID photograph quietly becomes its own cluster, and
   the interval collapses by the exact factor this module exists to prevent).
3. **OOD resampled within dataset.** ``BootstrapPlan.ood_by_dataset``, one
   universe per dataset, each resampled to its own size, so near-OOD and far-OOD
   group composition is fixed across replicates.
4. **One index set per replicate, applied to every scorer.** ``run_bootstrap``
   draws once per replicate and loops scorers *inside* that draw. Every input
   frame goes through ``xai_ood.schema.assert_aligned`` first -- not
   ``assert_canonical_order``, which is per-frame and cannot see that two
   correctly-sorted frames cover different row sets.
5. **Within-replicate paired differences, never CI overlap.**
   ``BootstrapResult.difference``. ``Interval`` carries a ``kind``, and
   ``excludes_zero`` **raises** on a marginal interval, so the overlap heuristic
   is not merely discouraged in a docstring.
6. **Degradation paired across protocols from the same drawn ids.** Both
   protocols are evaluated inside one replicate from one ``drawn`` array;
   ``BootstrapResult.degradation``.
7. **95% percentile intervals, seed in the manifest, no BCa, and the
   FPR@95 threshold recomputed inside each replicate.** ``B_DEFAULT`` is
   1,000; the count an analysis uses is a ``run_bootstrap`` argument and is
   recorded in the manifest. Raising it costs CPU only, never a rescoring run.
   ``B_DEFAULT``,
   ``CI_LEVEL``, ``bootstrap_manifest``, and
   ``xai_ood.metrics.threshold_at_tpr`` called per replicate rather than
   once on the full sample.

The one property this module exists for
---------------------------------------
**A correct interval on the degradation must not shrink when duplicated rows are
added.** Moving from the standard to the full-spectrum protocol adds no new
photographs, only re-corrupted copies of photographs already present. A
row-level bootstrap treats those copies as independent draws and shrinks the
full-spectrum interval by up to roughly ``sqrt(76)`` on exactly the leg where
the effects are expected to be smallest.

``tests/test_bootstrap.py`` pins that, and pins it against the mutant rather
than in the abstract: with cs-ID scores that are exact copies of their clean
row, the cluster bootstrap's full-spectrum AUROC replicates are *equal* to its
standard ones and the interval width ratio is 1.0, while the same code driven by
a row-level cluster assignment collapses to about ``1/sqrt(76)``. A guard that
does not fail on the bug it names is not a guard.

Scale
-----
Everything computed here is a raw fraction in ``[0, 1]``, because that is what
``xai_ood.metrics`` returns. **The 0-100 conversion happens at exactly one
point in this module** -- ``_display``, called only by the three table
emitters -- and never again downstream. See the comment on that function.

Module layout
-------------
::

    bootstrap/plan.py    parts 1-3: clusters and the plan
    bootstrap/run.py     parts 4-7: intervals, result, the loop
    bootstrap/tables.py  the three tables and the manifest fragment

The dependency order is ``plan <- run <- tables`` with no cycles, and this
``__init__`` re-exports the whole public surface, so ``from xai_ood.bootstrap
import build_plan, run_bootstrap, results_table`` resolves regardless of which
module a name lives in. The one private name referenced elsewhere by path is
``_display``, at ``xai_ood.bootstrap.tables._display``.
"""

from __future__ import annotations

from .plan import (
    CSID_SPLIT,
    DEFAULT_OOD_GROUPS,
    ID_TEST_SPLIT,
    PROTOCOL_ID_SPLITS,
    BootstrapPlan,
    ClusterIndex,
    build_cluster_index,
    build_plan,
)
from .run import (
    B_DEFAULT,
    CI_LEVEL,
    QUANTILE_METHOD,
    BootstrapResult,
    Interval,
    percentile_interval,
    replicate_indices,
    run_bootstrap,
)
from .tables import (
    DEGRADATION_COLUMNS,
    DIFFERENCE_COLUMNS,
    FPR95_FULL_SPECTRUM_CAVEAT,
    REQUIRED_PAIRWISE_CLAIMS,
    bootstrap_manifest,
    degradation_table,
    differences_table,
    results_table,
)

__all__ = [
    "B_DEFAULT",
    "CI_LEVEL",
    "QUANTILE_METHOD",
    "ID_TEST_SPLIT",
    "CSID_SPLIT",
    "PROTOCOL_ID_SPLITS",
    "DEFAULT_OOD_GROUPS",
    "REQUIRED_PAIRWISE_CLAIMS",
    "FPR95_FULL_SPECTRUM_CAVEAT",
    "DIFFERENCE_COLUMNS",
    "DEGRADATION_COLUMNS",
    "ClusterIndex",
    "BootstrapPlan",
    "Interval",
    "BootstrapResult",
    "build_cluster_index",
    "build_plan",
    "percentile_interval",
    "replicate_indices",
    "run_bootstrap",
    "results_table",
    "differences_table",
    "degradation_table",
    "bootstrap_manifest",
]
