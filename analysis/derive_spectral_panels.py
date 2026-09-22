"""Figure input for the two-panel spectral figure, from a spectral decomposition's output.

Separate from `derive_distributions.py` because the source is different: this reads the
output of a spectral decomposition of the embeddings, not the score dumps. Mixing the
two in one script would put a claim about provenance in a docstring instead of in the
file layout.

The decomposition itself is a one-off harness and is NOT part of this repository, so
this script cannot run from a bare checkout. Point XAI_OOD_SPECTRAL at a directory
holding its output and it will run; without one it refuses and says so.

Nothing is recomputed from the embeddings here. The separation curve is
S_i = mean_i(ood) - mean_i(id) over eigendirection i, restated from its definition
rather than imported, so this script takes no dependency on that harness.

PIN, and it found something. The ablation curve at K = 768 forward is the full-space
score and must reproduce the AUROC in the per-dataset table of record. Pinned PER DATASET,
where all six reproduce at 4.84e-05 AUROC points, which is that table's
four-decimal rounding and not a measured disagreement.

It is pinned per dataset because pinning it against the DATASET-AVERAGED group value
fails by 0.4794 points on far-OOD, and the reason is the estimand: the decomposition's
group rows concatenate the member datasets into one pile and take a single AUROC, which
is the POOLED construction, while the declared headline is the unweighted mean of the
per-dataset AUROCs. Pooled far-OOD is 51.1% MNIST, and three figures once went out on
the wrong side of that distinction.

So both group curves are emitted here, dataset-averaged and pooled, and the figure draws
the dataset-averaged one. The per-dataset companion every group figure needs is emitted
too.

    XAI_OOD_SPECTRAL=<decomposition-output> python derive_spectral_panels.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
#: Output of the spectral decomposition, which is a separate one-off harness and is not
#: part of this repository. There is no sensible default, so the path is required.
SPECTRAL = Path(os.environ.get("XAI_OOD_SPECTRAL", OUT / "spectral-decomposition-output"))
TABLE_OF_RECORD = OUT / "per-dataset-marginals-both-arms-both-protocols.csv"

ARM = "shrunk"
PROTOCOL = "standard"
MODEL = "marginal"
#: The scorer column the `marginal` spectral model corresponds to. The decomposition is
#: of the marginal full-covariance Mahalanobis distance, which is `marginal_full`.
SCORER = "marginal_full"

GROUPS = {"near_ood": ("cifar100", "tin"), "far_ood": ("mnist", "svhn", "texture", "places365")}
DATASETS = tuple(d for members in GROUPS.values() for d in members)
#: The table of record is written to four decimals and the ablation CSV to full float, so
#: the pin can be no tighter than that rounding.
PIN_TOLERANCE = 1e-4


def load_terms() -> dict:
    return json.loads((SPECTRAL / f"terms-{ARM}-{MODEL}.json").read_text())


def separation(payload: dict, group: str) -> np.ndarray:
    ident = np.asarray(payload["curves"]["id_test"]["mean_terms"], dtype=float)
    members = GROUPS[group]
    weights = np.array([payload["curves"][d]["n"] for d in members], dtype=float)
    stack = np.stack([np.asarray(payload["curves"][d]["mean_terms"], dtype=float) for d in members])
    return (weights[:, None] * stack).sum(axis=0) / weights.sum() - ident


def ablation() -> dict:
    rows = defaultdict(dict)
    with (SPECTRAL / "ablation-auroc.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["arm"] != ARM or row["protocol"] != PROTOCOL or row["model"] != MODEL:
                continue
            rows[(row["direction"], row["target"])][int(row["k"])] = float(row["auroc"]) * 100.0
    return rows


def per_dataset_of_record() -> dict:
    values = {}
    with TABLE_OF_RECORD.open() as fh:
        for row in csv.DictReader(fh):
            if (row["arm"], row["protocol"], row["scorer"], row["metric"]) != (
                    ARM, PROTOCOL, SCORER, "auroc"):
                continue
            if row["dataset"] in DATASETS:
                values[row["dataset"]] = float(row["value"])
    if len(values) != len(DATASETS):
        sys.stderr.write(f"REFUSED: {len(values)} of {len(DATASETS)} member datasets found.\n")
        raise SystemExit(2)
    return values


def averaged(curves: dict, direction: str, group: str) -> dict:
    """The dataset-averaged group curve, built from the decomposition's per-dataset rows."""
    members = GROUPS[group]
    grid = sorted(curves[(direction, members[0])])
    return {str(k): float(np.mean([curves[(direction, d)][k] for d in members])) for k in grid}


def main() -> int:
    if not SPECTRAL.is_dir():
        sys.stderr.write(f"REFUSED: no spectral decomposition output at {SPECTRAL}. "
                         f"Set XAI_OOD_SPECTRAL to the directory holding it.\n")
        raise SystemExit(2)

    terms = load_terms()
    curves = ablation()

    of_record = per_dataset_of_record()
    worst, examined = 0.0, 0
    for dataset in DATASETS:
        worst = max(worst, abs(curves[("forward", dataset)][768] - of_record[dataset]))
        examined += 1
    ok = examined == len(DATASETS) and np.isfinite(worst) and worst <= PIN_TOLERANCE
    print(f"PIN, full-space ablation against the table of record, PER DATASET: examined "
          f"{examined} datasets, worst {worst:.4e} AUROC points, tolerance {PIN_TOLERANCE:.0e}")
    if not ok:
        print("REFUSED: the ablation's full-space point does not reproduce the reported AUROC.")
        return 2

    for group in GROUPS:
        pooled = curves[("forward", group)][768]
        mean_of_members = float(np.mean([of_record[d] for d in GROUPS[group]]))
        print(f"  ESTIMAND, {group}: the decomposition's group row is pooled at {pooled:.4f}; the "
              f"dataset-averaged value of record is {mean_of_members:.4f}, a gap of "
              f"{pooled - mean_of_members:+.4f} AUROC points")

    payload = {
        "_source_spectral": str(SPECTRAL),
        "_source_table": str(TABLE_OF_RECORD),
        "_arm": ARM, "_protocol": PROTOCOL, "_model": MODEL, "_scorer": SCORER,
        "_note": "Generated by derive_spectral_panels.py. The figure script draws these and computes nothing.",
        "eigenvalues": [float(v) for v in terms["eigenvalues"]],
        "groups": {},
    }
    for group in GROUPS:
        sep = separation(terms, group)
        # Cumulative share of the total separation, in eigendirection order, which is
        # descending eigenvalue. Negative S_i are kept rather than clipped: a direction
        # that works against the separation is part of the curve and hiding it would draw
        # a monotone picture of something that is not monotone.
        total = float(sep.sum())
        payload["groups"][group] = {
            "members": list(GROUPS[group]),
            "separation": [float(v) for v in sep],
            "cumulative_share": [float(v) for v in np.cumsum(sep) / total],
            "total_separation": total,
            # The drawn curves. Dataset-averaged is the declared headline estimand.
            "forward_dataset_averaged": averaged(curves, "forward", group),
            "backward_dataset_averaged": averaged(curves, "backward", group),
            # The decomposition's own group rows, kept so the two estimands sit side by side
            # rather than one silently replacing the other.
            "forward_pooled": {str(k): v for k, v in sorted(curves[("forward", group)].items())},
            "backward_pooled": {str(k): v for k, v in sorted(curves[("backward", group)].items())},
            "full_space_pooled": curves[("forward", group)][768],
            "full_space_dataset_averaged": float(np.mean([of_record[d] for d in GROUPS[group]])),
        }
    payload["per_dataset"] = {
        d: {
            "forward": {str(k): v for k, v in sorted(curves[("forward", d)].items())},
            "backward": {str(k): v for k, v in sorted(curves[("backward", d)].items())},
            "full_space": curves[("forward", d)][768],
            "table_of_record": of_record[d],
        } for d in DATASETS
    }
    path = OUT / "spectral-figure-input.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {path.name}")

    print("\nWHAT THIS DID NOT CHECK:")
    print("  - the spectral decomposition itself. Its own controls own that")
    print("  - the class_conditional model, which is not drawn")
    print("  - the unshrunk arm and the full-spectrum protocol, neither of which is drawn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
