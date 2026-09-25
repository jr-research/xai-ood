"""Build the results tables from persisted bootstraps, and run the 3b count.

Reads the two pickles ``run_bootstrap_pair.py`` wrote. Emits three things and
stops: every number here is descriptive and carries no significance claim,
because the descriptive branch was elected on 2026-09-08 before any Phase 2
result existed.

**The pooled table and the dataset-averaged table are different estimands and
both are reported.** The dataset-averaged figure is the
headline, because it is what OpenOOD computes and what Phase 1's reproduction
table already used, so a pooled headline would quietly change what a "published
against reproduced" row means between phases. The pooled figure is reported
beside it, labelled, and never promoted.

**The dataset-averaged interval is a within-replicate construction.** The
per-dataset AUROCs are averaged *inside* each replicate and the percentile
interval is taken over that distribution. Averaging the per-dataset interval
endpoints is a different quantity and is not permitted: measured on a fixture
carrying the real far-OOD row proportions it came out 15.5% too wide.

Two library behaviours this works around rather than trips over:

* ``results_table`` validates that each OOD group is a ``split_role`` in the
  frozen schema, so it **raises** on the singleton result whose groups are
  dataset names. The singleton result is therefore used only through
  ``replicates()``, which is what it is needed for.
* ``results_table(scorers=None)`` would emit a row per matched-size-null draw.
  None of them gets an interval, so the fitted scorers are
  passed explicitly.

    python3 build_result_tables.py --run-dir results/runs/<id> --timestamp YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from xai_ood.bootstrap.plan import DEFAULT_OOD_GROUPS
from xai_ood.bootstrap.run import percentile_interval
from xai_ood.bootstrap.tables import results_table

CONTROL_PREFIX: str = "control_random_subspace_residual"

#: The scorers named by claims C1 to C6. The endpoint count is defined over the
#: intervals attached to those claims, not over every scorer in the dump.
CLAIM_SCORERS: tuple[str, ...] = (
    "marginal_diagonal",
    "marginal_full",
    "class_conditional_diagonal",
    "class_conditional_full",
    "rmd",
    "rmd_pp",
    "knn_normalized",
    "knn_unnormalized",
    "pca_residual_all_id",
)

#: The pre-declared endpoint threshold, fixed before any run so the count could
#: not be reframed once it was known.
CEILING: float = 0.999


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--protocol", default="standard")
    options = parser.parse_args(argv)

    manifest = json.loads(
        (options.run_dir / "scores" / "run_metadata.json").read_text()
    )
    results = {
        name: pickle.load((options.run_dir / "bootstrap" / f"bootstrap_{name}.pkl").open("rb"))
        for name in ("grouped", "singleton")
    }
    scorers = [
        c for c in manifest["score_dump"]["columns"] if not c.startswith(CONTROL_PREFIX)
    ]
    for name, result in results.items():
        print(f"{name:<10} B={result.n_replicates} seed={result.seed} "
              f"level={result.level} protocols={result.protocols} "
              f"groups={result.ood_groups}")

    tables = options.run_dir.parent.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    stem = options.run_dir.name

    pooled = results_table(
        results["grouped"],
        seed=manifest["seed"],
        repo_commit=manifest["repo_commit"],
        openood_commit=manifest["openood_commit"],
        timestamp=options.timestamp,
        scorers=scorers,
    )
    pooled.to_csv(tables / f"{stem}-pooled.csv", index=False)
    print(f"\n=== pooled estimand, {len(pooled)} rows, 0-100 scale ===")
    print(pooled[["scorer", "split_role", "auroc", "auroc_ci_low",
                  "auroc_ci_high", "fpr95"]].to_string(index=False))

    rows = []
    singleton = results["singleton"]
    for scorer in scorers:
        for group, members in DEFAULT_OOD_GROUPS.items():
            replicates = np.mean(
                [singleton.replicates(scorer, options.protocol, ds, "auroc")
                 for ds in members],
                axis=0,
            )
            point = float(np.mean(
                [singleton.point[(scorer, options.protocol, ds, "auroc")]
                 for ds in members]
            ))
            low, high = percentile_interval(replicates, singleton.level)
            rows.append({"scorer": scorer, "group": group, "auroc": 100 * point,
                         "ci_low": 100 * low, "ci_high": 100 * high})
    averaged = pd.DataFrame(rows)
    averaged.to_csv(tables / f"{stem}-dataset-averaged.csv", index=False)
    print(f"\n=== dataset-averaged estimand (HEADLINE, plan 1.7), 0-100 scale ===")
    print(averaged.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n=== plan 3b: marginal AUROC intervals with an endpoint above "
          f"{CEILING} ===")
    hits = []
    grouped = results["grouped"]
    for scorer in CLAIM_SCORERS:
        for group in grouped.ood_groups:
            interval = grouped.marginal(scorer, options.protocol, group, "auroc")
            if interval.high > CEILING or interval.low > CEILING:
                hits.append((scorer, group, interval.low, interval.high))
                print(f"  {scorer:<32} {group:<9} "
                      f"[{interval.low:.6f}, {interval.high:.6f}]")
    total = len(CLAIM_SCORERS) * len(grouped.ood_groups)
    print(f"  count: {len(hits)} of {total} claim-relevant marginal intervals")
    if not hits:
        print("  The endpoint check closes by measurement: no primary interval came "
              "within 0.001 of the boundary, so percentile intervals stand and "
              "no BCa implementation and no stated limitation is owed.")
    print(f"\nwrote {tables / f'{stem}-pooled.csv'}")
    print(f"wrote {tables / f'{stem}-dataset-averaged.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
