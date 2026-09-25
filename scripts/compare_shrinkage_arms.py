"""Check that two dumps differing only in the shrinkage flag are aligned.

``--shrinkage`` reaches only the six Gaussian configurations. ``fit_knn`` and
``fit_pca_residual`` never see the flag, and the matched-size-null draws are
seeded from the run seed alone. So of the thirty-three score columns, **six must
differ and twenty-seven must be bitwise identical**.

That is not a formality, it is the pairing check. The shrunk arm is the headline
and the unshrunk arm is the declared ablation, and an ablation is a paired
comparison on identical rows. If any of the twenty-seven differs, the two runs
did not see the same resample-eligible rows in the same order, the pairing is
invalid, and nothing computed from the pair can be trusted.

**A zero delta on a Gaussian column is a failure too**, in the other direction:
it means ``--shrinkage`` did not reach that fit. Both directions are reported.

Exit status is non-zero if either direction fails.

    python3 compare_shrinkage_arms.py --unshrunk DIR --shrunk DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xai_ood.methods.gaussian import CONFIGURATIONS as GAUSSIAN_CONFIGURATIONS
from xai_ood.schema import CANONICAL_SORT_KEY

#: Read off the library rather than typed here, so a seventh Gaussian
#: configuration joins this check by being defined rather than by being
#: remembered.
GAUSSIAN: frozenset[str] = frozenset(c.name for c in GAUSSIAN_CONFIGURATIONS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unshrunk", type=Path, required=True)
    parser.add_argument("--shrunk", type=Path, required=True)
    options = parser.parse_args(argv)

    a = pd.read_parquet(options.unshrunk / "per_sample_scores.parquet")
    b = pd.read_parquet(options.shrunk / "per_sample_scores.parquet")
    key = list(CANONICAL_SORT_KEY)

    failures: list[str] = []
    print(f"rows {len(a)} and {len(b)}, equal: {len(a) == len(b)}")
    if len(a) != len(b):
        failures.append("row counts differ")
    elif not a[key].equals(b[key]):
        failures.append("key columns differ, so the rows are not the same rows")
    print(f"key columns identical: {len(a) == len(b) and a[key].equals(b[key])}")

    identical, differing = [], []
    for column in [c for c in a.columns if c not in CANONICAL_SORT_KEY]:
        delta = float(
            np.abs(a[column].to_numpy(np.float64) - b[column].to_numpy(np.float64)).max()
        )
        (differing if column in GAUSSIAN else identical).append((column, delta))

    print(f"\nMUST be bitwise identical: {len(identical)} columns")
    for column, delta in identical:
        if delta != 0.0:
            print(f"  MISMATCH {column}: {delta:.3e}")
            failures.append(f"{column} differs by {delta:.3e} and must not")
    worst = max((d for _, d in identical), default=0.0)
    print(f"  max abs deviation: {worst:.3e}   (must be 0.000e+00)")

    print(f"\nMUST differ: {len(differing)} Gaussian columns")
    for column, delta in sorted(differing):
        flag = "  <-- ZERO, the flag did not reach this fit" if delta == 0.0 else ""
        print(f"  {column:<32} max abs delta = {delta:.3e}{flag}")
        if delta == 0.0:
            failures.append(f"{column} is identical; --shrinkage did not reach it")

    if failures:
        print(f"\nFAILED: {len(failures)}")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nThe two arms are aligned and the pairing is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
