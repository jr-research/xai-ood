"""Differential cross-check: our AUROC and FPR@95 against OpenOOD's own.

**Runs only where OpenOOD is importable**, which is the workstation. That is the
whole reason this exists as a script: its output cannot be regenerated anywhere
else, so it must be written to a file rather than read off a terminal.

What it establishes. The design defers the metric
conventions to a later session "with the OpenOOD cross-check as the evidence",
so the number this produces is cited rather than merely reassuring. Two
implementations of two metrics, pointed at an identical score array:

* **AUROC.** Ours is ``P(score_OOD > score_ID) + 0.5 P(tie)``, the Mann-Whitney
  statistic with mid-ranks. OpenOOD's is ``sklearn.metrics.auc`` over
  ``roc_curve``, which is trapezoidal integration of the ROC. These are the same
  statistic by two routes, so the residual difference is summation-order noise.
* **FPR@95TPR.** Ours reads the ``ceil(0.95 * n_id)``-th smallest ID score, an
  observed value with no interpolation. OpenOOD reads ``roc_curve``'s first
  threshold whose empirical TPR is at or above the target. The criterion is the
  same; the routes differ, and this measures whether the routes do.

**Sign convention, which is where this check would silently go wrong.** Every
scorer here returns higher = more OOD. OpenOOD takes a *confidence*, higher =
more ID, with ``label == -1`` marking OOD rows. So ``conf = -score`` and the ID
rows carry a non-negative label. Getting this backwards produces a clean-looking
``1 - AUROC`` rather than an error.

**Blind by construction.** Only ``pca_residual_all_id`` is printed with its
values, because the matched-size null already required reading it. Every other
scorer contributes only ``abs(ours - openood)``, so this can be run before the
headline AUROCs are read without disclosing them.

    python3 crosscheck_openood_metrics.py --run-dir results/runs/<id>/scores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xai_ood.bootstrap.plan import DEFAULT_OOD_GROUPS
from xai_ood.metrics import TPR_TARGET, auroc as our_auroc, fpr_at_tpr as our_fpr
from xai_ood.schema import CANONICAL_SORT_KEY

#: The one scorer printed with its values. Reading it is already required by the
#: matched-size null, so printing it here discloses nothing new.
DISCLOSED: str = "pca_residual_all_id"

#: Difference above which a cell is counted as a real disagreement rather than
#: floating-point noise. Two routes to one statistic differ in the last bits.
NOISE_FLOOR: float = 1e-12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--id-split", default="id_test")
    options = parser.parse_args(argv)

    try:
        from openood.evaluators.metrics import auc as oo_auc, fpr_recall as oo_fpr
    except ImportError:
        raise SystemExit(
            "openood is not importable here. This check runs only where the "
            "pinned OpenOOD checkout is installed, which is the workstation. "
            "Its output cannot be reproduced elsewhere, so copy the saved "
            "output rather than skipping it."
        )

    frame = pd.read_parquet(options.run_dir / "per_sample_scores.parquet")
    scorers = [
        c
        for c in frame.columns
        if c not in CANONICAL_SORT_KEY
        and not c.startswith("control_random_subspace_residual")
    ]
    datasets = [ds for group in DEFAULT_OOD_GROUPS.values() for ds in group]
    id_mask = frame["split"] == options.id_split

    auroc_diffs: list[float] = []
    fpr_diffs: list[float] = []
    worst_auroc = worst_fpr = (0.0, "", "")

    for scorer in scorers:
        id_scores = frame.loc[id_mask, scorer].to_numpy(dtype=np.float64)
        for dataset in datasets:
            ood_scores = frame.loc[frame["split"] == dataset, scorer].to_numpy(
                dtype=np.float64
            )
            # conf is higher = more ID, so it negates the score; OOD rows carry -1.
            conf = np.concatenate([-id_scores, -ood_scores])
            label = np.concatenate(
                [np.zeros(id_scores.size, int), -np.ones(ood_scores.size, int)]
            )
            ours_a = float(our_auroc(id_scores, ood_scores))
            theirs_a = float(oo_auc(conf, label)[0])
            ours_f = float(our_fpr(id_scores, ood_scores, TPR_TARGET))
            theirs_f = float(oo_fpr(conf, label, TPR_TARGET)[0])

            auroc_diffs.append(abs(ours_a - theirs_a))
            fpr_diffs.append(abs(ours_f - theirs_f))
            if auroc_diffs[-1] > worst_auroc[0]:
                worst_auroc = (auroc_diffs[-1], scorer, dataset)
            if fpr_diffs[-1] > worst_fpr[0]:
                worst_fpr = (fpr_diffs[-1], scorer, dataset)
            if scorer == DISCLOSED:
                print(
                    f"  [{DISCLOSED}] {dataset:<10} "
                    f"AUROC ours {ours_a:.6f} openood {theirs_a:.6f} | "
                    f"FPR95 ours {ours_f:.6f} openood {theirs_f:.6f}"
                )

    n = len(auroc_diffs)
    print(f"\n{n} comparisons: {len(scorers)} scorers x {len(datasets)} datasets")
    print(
        f"AUROC |ours - openood| : max {max(auroc_diffs):.3e}  "
        f"median {float(np.median(auroc_diffs)):.3e}  "
        f"worst {worst_auroc[1]}/{worst_auroc[2]}"
    )
    print(
        f"FPR95 |ours - openood| : max {max(fpr_diffs):.3e}  "
        f"median {float(np.median(fpr_diffs)):.3e}  "
        f"worst {worst_fpr[1]}/{worst_fpr[2]}"
    )
    print(
        f"cells disagreeing by more than {NOISE_FLOOR:g}: "
        f"AUROC {sum(1 for x in auroc_diffs if x > NOISE_FLOOR)} of {n}, "
        f"FPR95 {sum(1 for x in fpr_diffs if x > NOISE_FLOOR)} of {n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
