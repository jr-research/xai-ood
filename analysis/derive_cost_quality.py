"""Cost against quality as one curve, and the Pareto frontier of it.

The thesis reports scoring time in one table and AUROC in another, and the
benchmarking framing the title now commits to needs them on one axis pair. This
script joins them and marks, per protocol and group, the methods for which nothing
else is both cheaper and better.

WHAT THE COST NUMBER IS, AND WHAT IT IS NOT. It is the `timing` block of the run's
own scores/run_metadata.json, and that block is a single unreplicated
measurement: one wall-clock figure per scorer, n = 1, no repeats, no warm-up
discard, no interleaving, the thirteen scorers run sequentially in one process on
one workstation. Every AUROC on
the other axis carries a 1,000-replicate percentile interval and every cost on this
one carries nothing. The manifest's own caveat also applies: nearest-neighbour search
is exact numpy rather than faiss and the covariance factorisations use numpy's
Cholesky rather than scipy's, so any frontier position involving the kNN columns is a
claim about this implementation as much as about the method. A frontier is a
statement about an ordering, and the ordering here spans two orders of magnitude,
which is the only reason an unreplicated measurement can carry it at all.

Fit and score cost are kept apart because they amortise differently: fit is paid once
per deployment over 50,000 reference rows, scoring is paid per image forever. The
frontier is computed on score cost, which is the quantity a practitioner pays, and
fit cost is carried in the table beside it.

Read-only against the repository's results tree. Writes only into this directory.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# The repository root, the directory above this one. Overridable with
# XAI_OOD_ROOT so a run tree held outside the checkout can be read in place.
ROOT = Path(os.environ.get("XAI_OOD_ROOT", Path(__file__).resolve().parents[1]))
RUN = ROOT / "results" / "runs" / "2026-09-10-full-spectrum-shrunk"
OUT = Path(__file__).resolve().parent

NEAR = ("cifar100", "tin")
FAR = ("mnist", "svhn", "texture", "places365")
GROUPS = {"near_ood": NEAR, "far_ood": FAR}
PROTOCOLS = ("standard", "full_spectrum")


class _Stub:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("xai_ood"):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


def main() -> int:
    with open(RUN / "bootstrap" / "bootstrap_singleton.pkl", "rb") as handle:
        reference = _Unpickler(handle).load()
    metadata = json.loads((RUN / "scores" / "run_metadata.json").read_text())
    timing = metadata["timing"]
    scorers = list(reference.scorers)

    missing = [s for s in scorers if s not in timing["score"] or s not in timing["fit"]]
    if missing:
        sys.exit(f"no timing for {missing}; the join cannot be made.")
    rows_scored = {int(timing["score"][s]["rows"]) for s in scorers}
    if len(rows_scored) != 1:
        sys.exit(f"scorers timed over different row counts: {rows_scored}")
    print(f"timing: node {timing['machine']['node']}, numpy {timing['machine']['numpy']}, "
          f"{rows_scored.pop()} rows scored, n = 1 measurement per scorer (entry 38)")

    score_us = {s: float(timing["score"][s]["microseconds_per_sample"]) for s in scorers}
    fit_s = {s: float(timing["fit"][s]["seconds"]) for s in scorers}

    rows = []
    for protocol in PROTOCOLS:
        for group, datasets in GROUPS.items():
            auroc = {
                s: float(np.mean([reference.point[(s, protocol, d, "auroc")] for d in datasets])) * 100.0
                for s in scorers
            }
            # The frontier is an ordering, so the gap between adjacent frontier
            # points decides whether a step on it is a real choice or a coin flip.
            reps = {
                s: np.mean([np.asarray(reference.values[(s, protocol, d, "auroc")], dtype=float)
                            for d in datasets], axis=0) * 100.0
                for s in scorers
            }
            fpr = {
                s: float(np.mean([reference.point[(s, protocol, d, "fpr95")] for d in datasets])) * 100.0
                for s in scorers
            }
            for s in scorers:
                # On the frontier if nothing is both cheaper-or-equal and better,
                # with at least one of the two strict.
                dominated_by = [
                    t for t in scorers
                    if t != s and score_us[t] <= score_us[s] and auroc[t] >= auroc[s]
                    and (score_us[t] < score_us[s] or auroc[t] > auroc[s])
                ]
                low, high = np.percentile(reps[s], [2.5, 97.5])
                rows.append([
                    protocol, group, s, f"{score_us[s]:.4f}", f"{fit_s[s]:.4f}",
                    f"{auroc[s]:.4f}", f"{low:.4f}", f"{high:.4f}", f"{fpr[s]:.4f}",
                    "yes" if not dominated_by else "no",
                    len(dominated_by), ";".join(dominated_by),
                ])
    path = OUT / "cost-quality-pareto.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "protocol", "group", "scorer", "score_microseconds_per_sample",
            "fit_seconds", "auroc", "auroc_ci_low", "auroc_ci_high", "fpr95",
            "on_pareto_frontier", "n_dominating", "dominated_by",
        ])
        writer.writerows(rows)
    print(f"wrote {path.name}  ({len(rows)} rows)")

    steps = []
    for protocol in PROTOCOLS:
        for group, datasets in GROUPS.items():
            reps = {
                s: np.mean([np.asarray(reference.values[(s, protocol, d, "auroc")], dtype=float)
                            for d in datasets], axis=0) * 100.0
                for s in scorers
            }
            frontier = sorted(
                [r for r in rows if r[0] == protocol and r[1] == group and r[9] == "yes"],
                key=lambda r: float(r[3]),
            )
            for lower, upper in zip(frontier, frontier[1:]):
                paired = reps[upper[2]] - reps[lower[2]]
                low, high = np.percentile(paired, [2.5, 97.5])
                steps.append([
                    protocol, group, lower[2], upper[2],
                    f"{float(upper[3]) / float(lower[3]):.2f}",
                    f"{float(upper[5]) - float(lower[5]):.4f}",
                    f"{low:.4f}", f"{high:.4f}",
                    "yes" if low <= 0.0 <= high else "no",
                ])
    steps_path = OUT / "cost-quality-frontier-steps.csv"
    with open(steps_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "protocol", "group", "from_scorer", "to_scorer", "cost_multiple",
            "auroc_gained", "ci_low", "ci_high", "interval_spans_zero",
        ])
        writer.writerows(steps)
    print(f"wrote {steps_path.name}  ({len(steps)} rows)")

    for protocol in PROTOCOLS:
        for group in GROUPS:
            frontier = [r for r in rows if r[0] == protocol and r[1] == group and r[9] == "yes"]
            frontier.sort(key=lambda r: float(r[3]))
            print(f"\n{protocol} / {group}: {len(frontier)} of {len(scorers)} on the frontier")
            for r in frontier:
                print(f"   {r[2]:34s} {float(r[3]):9.2f} us/sample   AUROC {float(r[5]):7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
