"""Run the two bootstraps the group-AUROC estimand needs, and persist both.

The design requires the bootstrap **twice on the same cached scores**,
once with the default OOD groups and once with singleton per-dataset groups.
One plan cannot hold both: ``build_plan`` refuses a dataset appearing in more
than one group, because its rows would be resampled twice and the groups would
not be independent. At one seed the two runs share their draws, since
``replicate_indices`` draws the ID clusters first and then one draw per dataset
in ``ood_by_dataset`` order, which is what makes them comparable.

**Both ``BootstrapResult`` objects are pickled before any table is built.**
``results_table`` discards the replicate arrays, and the dataset-averaged
interval cannot be constructed without them, so losing them costs a re-run of
the whole bootstrap. Persisting first turns the one expensive-to-undo step of
this analysis into a file read.

Two defaults worth knowing rather than discovering:

* ``run_bootstrap(scorers=None)`` selects **every** non-key column, which
  includes the matched-size-null draws. None of them gets
  an interval. Passing ``--scorers-only`` restricts the run to the fitted
  scorers, which is both faster and closer to what is reported. The default is
  to include them, because the numbers for the fitted scorers are identical
  either way: ``rng`` is consumed only by ``replicate_indices``, never by the
  scorer loop.
* ``B`` comes from ``B_DEFAULT`` and is 1,000, which is what the code ships and
  what is declared after the descriptive branch was elected.

    python3 run_bootstrap_pair.py --run-dir results/runs/<id> --seed INT
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from xai_ood.bootstrap.plan import DEFAULT_OOD_GROUPS, build_plan
from xai_ood.bootstrap.run import B_DEFAULT, run_bootstrap
from xai_ood.schema import CANONICAL_SORT_KEY

CONTROL_PREFIX: str = "control_random_subspace_residual"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run directory, whose scores/ holds the dump")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicates", type=int, default=B_DEFAULT)
    parser.add_argument("--protocols", default="standard")
    parser.add_argument("--scorers-only", action="store_true",
                        help="exclude the control draws from the bootstrap")
    options = parser.parse_args(argv)

    scores_dir = options.run_dir / "scores"
    frame = pd.read_parquet(scores_dir / "per_sample_scores.parquet")
    out = options.run_dir / "bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    protocols = tuple(p for p in options.protocols.split(",") if p)

    scorers = None
    if options.scorers_only:
        scorers = [
            c for c in frame.columns
            if c not in CANONICAL_SORT_KEY and not c.startswith(CONTROL_PREFIX)
        ]
    print(f"B = {options.replicates}  seed = {options.seed}  rows = {len(frame)}")
    print(f"protocols = {protocols}  scorers = "
          f"{'fitted only, ' + str(len(scorers)) if scorers else 'all columns'}")

    singles = {ds: (ds,) for group in DEFAULT_OOD_GROUPS.values() for ds in group}
    plans = {
        "grouped": build_plan(frame, ood_groups=DEFAULT_OOD_GROUPS,
                              protocols=protocols, allow_missing_csid=True,
                              name="grouped"),
        "singleton": build_plan(frame, ood_groups=singles, protocols=protocols,
                                allow_missing_csid=True, name="singleton"),
    }
    for name, plan in plans.items():
        print(f"{name:<10} ood_groups: {plan.diagnostics['ood_groups']}")

    for name, plan in plans.items():
        print(f"\n--- run_bootstrap [{name}] ---", flush=True)
        result = run_bootstrap(frame, plan, seed=options.seed,
                               n_replicates=options.replicates, scorers=scorers)
        target = out / f"bootstrap_{name}.pkl"
        with target.open("wb") as handle:
            pickle.dump(result, handle)
        print(f"    saved {target}", flush=True)

    print("\nBoth BootstrapResult objects persisted before any table was built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
