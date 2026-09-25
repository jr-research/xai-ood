"""Read every Phase 2 run diagnostic, and fail if a pre-declared one is wrong.

The design requires the diagnostics to be read **before**
any AUROC, which is a blind-analysis step rather than a formality: a number read
after the headline is a number the headline can influence. This script is that
step, made executable so it is evidence rather than a paste in a transcript.

Nothing here forms an AUROC. ``achieved_tpr`` is computed from the ID scores
alone through ``threshold_at_tpr``, never through ``metrics.evaluate``, which
would return an AUROC in the same dict.

Two kinds of output. Every diagnostic is printed. The ones declared in
advance are additionally **asserted**, and a failure sets the exit status, so a
run whose cluster join silently degenerated cannot pass by being read
inattentively.

    python3 read_run_diagnostics.py --run-dir results/runs/<run-id>/scores

Expected values are arguments with defaults rather than literals, because run B
changes three of them: the cs-ID rows stop being zero, the cluster size range
gains a full_spectrum entry of [1, 76], and the protocol list gains a second
member.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xai_ood.bootstrap.plan import build_plan
from xai_ood.metrics import TPR_TARGET, threshold_at_tpr

#: Thirteen scorers plus twenty matched-size-null draws. Both are fixed
#: halves, so a dump of any other width means a stale driver was run.
EXPECTED_COLUMNS: int = 33

#: One cluster per source photograph in the CIFAR-10 test split. Under the
#: standard protocol each carries exactly one row, so the size range is [1, 1];
#: under full spectrum the cs-ID grid puts 76 rows on 2,000 of them.
EXPECTED_ID_CLUSTERS: int = 9_000

#: The row set the degeneracy verdict is declared on. Three others are recorded
#: for transparency and are not the verdict.
EXPECTED_DEGENERACY_ROW_SET: str = "id_test_and_csid"


class Checks:
    """Collects pass/fail on the pre-declared checks so all of them are run."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def that(self, condition: bool, label: str, detail: str) -> None:
        mark = "ok  " if condition else "FAIL"
        print(f"  [{mark}] {label}: {detail}")
        if not condition:
            self.failures.append(f"{label}: {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def provenance(manifest: dict[str, Any], checks: Checks) -> None:
    """What run this is, and that it is not a dry run wearing a real name."""
    section("A. provenance and dump shape")
    dump = manifest["score_dump"]
    print(f"  repo_commit    : {manifest['repo_commit']}")
    print(f"  openood_commit : {manifest['openood_commit']}")
    print(f"  seed           : {manifest['seed']}")
    print(f"  control d      : {manifest['control']['d']}"
          f"  draws: {manifest['control']['draws']}")
    print(f"  scored splits  : {manifest['scored']['splits']}")
    checks.that(
        manifest.get("dry_run") is False,
        "not a dry run",
        f"dry_run={manifest.get('dry_run')}",
    )
    checks.that(
        len(dump["columns"]) == EXPECTED_COLUMNS,
        "dump width",
        f"{len(dump['columns'])} score columns, expected {EXPECTED_COLUMNS}",
    )


def fit_on_train(manifest: dict[str, Any], scorers: list[str], checks: Checks) -> None:
    """Every scorer fitted on the training split.

    ``reference_rows`` is deliberately **not** asserted equal to the fit count.
    The class-mean subspace is fitted to the K class means rather than to the
    rows they were averaged from, so its reference count is K and reads 10 at
    full scale. What must hold is that each scorer was fitted on the train
    split; what it kept from that split varies by family.
    """
    section("B. fit on train")
    reference = manifest["fit_reference"]
    expected_rows = reference["rows"]
    print(f"  fit split      : {reference['split']}, {expected_rows} rows")
    for name in scorers:
        entry = reference["per_scorer"][name]
        print(f"  {name:<34} fit_rows={entry['fit_rows']:>6}"
              f"  reference_rows={entry['reference_rows']:>6}")
    wrong = [
        name
        for name in scorers
        if reference["per_scorer"][name]["fit_rows"] != expected_rows
    ]
    checks.that(
        not wrong,
        "fitted on the training split",
        f"all {len(scorers)} scorers at {expected_rows} rows"
        if not wrong
        else f"{wrong} disagree with {expected_rows}",
    )


def degeneracy(manifest: dict[str, Any], checks: Checks) -> None:
    """The verdict, taken from the declared row set and not from another one."""
    section("C. degeneracy")
    for name, report in sorted(manifest["degeneracy"].items()):
        sets = report["row_sets"]
        print(f"  {name:<34} rho={report['rho_centred_norm']:+.4f}"
              f"  {report['verdict']}")
        print("      " + "  ".join(
            f"{key}={sets[key]['rho_centred_norm']:+.4f}" for key in sorted(sets)
        ))
    wrong = [
        name
        for name, report in manifest["degeneracy"].items()
        if report.get("row_set") != EXPECTED_DEGENERACY_ROW_SET
    ]
    checks.that(
        not wrong,
        "verdict row set",
        f"all on {EXPECTED_DEGENERACY_ROW_SET}"
        if not wrong
        else f"{wrong} report another set",
    )


def conditioning(manifest: dict[str, Any]) -> None:
    """Both shrinkage arms side by side.

    Printed rather than asserted, because an expectation is stated here
    and not a threshold. A small selected intensity that nonetheless moves the
    condition number by orders of magnitude is the case the expectation did not
    anticipate, and it is visible only by reading the two arms together.
    """
    section("D. covariance conditioning, both arms")
    for arm in ("scored_arm", "unscored_arm"):
        print(f"  --- {arm} ---")
        for _, fits in sorted(manifest["shrinkage_diagnostics"][arm].items()):
            for fit_name, diag in sorted(fits.items()):
                print(f"    {fit_name:<40} cond={diag['condition_number']:.3e}"
                      f"  shrink={diag['shrinkage']}"
                      f"  intensity={diag['shrinkage_intensity']:.6f}"
                      f"  n/p={diag['samples_per_dimension']:.1f}")


def resampling(frame: pd.DataFrame, protocols: tuple[str, ...], checks: Checks) -> None:
    """The cluster join, which fails silently and only shows up here.

    A degenerate join gives every cs-ID photograph its own cluster and shrinks
    the full-spectrum interval by up to a factor of sqrt(76). Nothing else in a
    run looks wrong when that happens.
    """
    section("E. resampling plan")
    plan = build_plan(
        frame, protocols=protocols, allow_missing_csid=True, name="diagnostics"
    )
    diagnostics = plan.diagnostics
    for key in (
        "resampling_unit",
        "n_id_test_rows",
        "n_csid_rows",
        "n_csid_photographs",
        "rows_per_protocol",
        "ood_rows",
        "ood_groups",
        "unused_splits",
    ):
        print(f"  {key:<20}: {diagnostics[key]}")
    checks.that(
        diagnostics["n_id_clusters"] == EXPECTED_ID_CLUSTERS,
        "id clusters",
        f"{diagnostics['n_id_clusters']}, expected {EXPECTED_ID_CLUSTERS}",
    )
    if "standard" in protocols:
        got = diagnostics["cluster_size_range"]["standard"]
        checks.that(
            got == [1, 1],
            "standard cluster size range",
            f"{got}, expected [1, 1]",
        )
    if "full_spectrum" in protocols:
        got = diagnostics["cluster_size_range"]["full_spectrum"]
        checks.that(
            got == [1, 76],
            "full spectrum cluster size range",
            f"{got}, expected [1, 76]. A [1, 1] here means the cs-ID rows and "
            f"the clean rows disagree about what an image_id is",
        )


def operating_point(
    frame: pd.DataFrame, scorers: list[str], id_split: str, checks: Checks
) -> None:
    """The TPR actually reached, from the ID scores alone.

    An FPR quoted against an unstated operating point is a different number
    from one quoted against 0.95, so this is read before the metric it
    qualifies. No OOD scores are touched, so no AUROC can be formed here.
    """
    section("F. achieved TPR, ID scores only")
    id_rows = frame[frame["split"] == id_split]
    n_id = len(id_rows)
    rank = int(np.ceil(TPR_TARGET * n_id))
    print(f"  n_id={n_id}  target={TPR_TARGET}  ceil rank={rank}")
    below_target = []
    for name in scorers:
        scores = np.sort(id_rows[name].to_numpy(dtype=np.float64))
        tau = threshold_at_tpr(scores, TPR_TARGET)
        achieved = float(np.count_nonzero(scores <= tau) / scores.size)
        print(f"  {name:<34} achieved_tpr={achieved:.6f}")
        if achieved < TPR_TARGET:
            below_target.append(name)
    checks.that(
        not below_target,
        "achieved TPR clears the target",
        "all scorers at or above 0.95"
        if not below_target
        else f"{below_target} fall below",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--id-split", default="id_test")
    parser.add_argument(
        "--protocols",
        default="standard",
        help="comma separated, e.g. 'standard' or 'standard,full_spectrum'",
    )
    options = parser.parse_args(argv)

    manifest = json.loads((options.run_dir / "run_metadata.json").read_text())
    frame = pd.read_parquet(options.run_dir / "per_sample_scores.parquet")
    scorers = [
        c for c in manifest["score_dump"]["columns"]
        if not c.startswith("control_random_subspace_residual")
    ]
    protocols = tuple(p for p in options.protocols.split(",") if p)

    checks = Checks()
    provenance(manifest, checks)
    fit_on_train(manifest, scorers, checks)
    degeneracy(manifest, checks)
    conditioning(manifest)
    resampling(frame, protocols, checks)
    operating_point(frame, scorers, options.id_split, checks)

    section("pre-declared checks")
    if checks.failures:
        print(f"  {len(checks.failures)} FAILED:")
        for failure in checks.failures:
            print(f"    {failure}")
        return 1
    print("  all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
