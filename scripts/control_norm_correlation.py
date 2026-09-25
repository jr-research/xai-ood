"""Rho against the embedding norms, for the matched-size-null draws.

**Post-hoc, and not pre-specified.** rho is declared for the five
PCA-residual scorers through ``MANIFEST_REQUIRES`` and declares nothing about
the control draws. This is an observation the study generated, and it is
reported as values only. The pre-declared 0.95 threshold and its verdict labels
are **not** applied here: borrowing a threshold declared for one quantity and
stamping it on another is exactly the move pre-declaration exists to prevent.

What it answers. The matched-size null shows that a random subspace at the same
dimension still separates ID from OOD to some degree. There is a rival reading
of that: the residual norm to *any* subspace may track the total embedding norm,
in which case the controls measure **scale** rather than direction-spread.
Contrasting their rho against the variance-ordered scorer's settles it, and it
explains the margin mechanistically rather than only reporting it.

Both norms are reported. The centred norm is the one the pre-declared control
uses, since the residual is computed on centred vectors. The plain norm is the
more direct test of the scale reading, because the rival hypothesis is about
total magnitude.

``degeneracy_report`` cannot be pointed at these columns: it reads
``scorer.config.subspace``, which a control has no equivalent of. So rho is
computed here from ``spearman_rho`` directly, on exactly the two inputs
``degeneracy_report`` uses: the score column and ``norm(z - scorer.mean)``. The
control scores come from the dump rather than being recomputed, so these are the
values that were actually scored.

**Two gates run first and the result is meaningless if either fails.** One
recomputes the matched scorer from the cache and compares against its dumped
column, which proves the embeddings were gathered in the dump's row order. Note
that this is compared against a tolerance and **not** against bitwise equality:
the driver scored the full evaluation matrix in one call and this scores the ID
rows alone, so BLAS blocking and summation order differ and agreement is to
machine precision rather than to the last bit. The other reproduces the rho the
manifest recorded, which a misalignment would destroy.

    python3 control_norm_correlation.py --run-dir DIR --cache-root DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from xai_ood.methods.correlation import spearman_rho
from xai_ood.methods.pca_residual import CONFIGURATIONS, fit_pca_residual

#: Agreement required between the refit scorer and its dumped column, relative
#: to the score scale. Not zero: see the docstring.
ALIGNMENT_TOLERANCE: float = 1e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--driver-dir", type=Path, default=Path("scripts"))
    parser.add_argument("--id-split", default="id_test")
    parser.add_argument("--id-cache-split", default="cifar10_test")
    options = parser.parse_args(argv)

    sys.path.insert(0, str(options.driver_dir))
    import run_phase2 as driver

    manifest = json.loads((options.run_dir / "run_metadata.json").read_text())
    frame = pd.read_parquet(options.run_dir / "per_sample_scores.parquet")
    observed = driver.CONTROL_MATCHES
    controls = sorted(c for c in frame.columns if c.startswith(driver.CONTROL_COLUMN))

    train = driver.load_split(options.cache_root, driver.TRAIN_CACHE_SPLIT, "cls")
    test = driver.load_split(options.cache_root, options.id_cache_split, "cls")
    config = next(c for c in CONFIGURATIONS if c.name == observed)
    matched = fit_pca_residual(config, train.embeddings, train.labels)

    rows = frame["split"] == options.id_split
    image_ids = frame.loc[rows, "image_id"].tolist()
    z = test.rows_for(image_ids)
    plain = np.linalg.norm(z, axis=1)
    centred = np.linalg.norm(z - matched.mean, axis=1)
    dumped = frame.loc[rows, observed].to_numpy(dtype=np.float64)

    alignment = float(np.abs(dumped - matched.score(z)).max())
    rho_centred = spearman_rho(dumped, centred, name="observed:centred")
    recorded = manifest["degeneracy"][observed]["rho_centred_norm"]
    ok = alignment <= ALIGNMENT_TOLERANCE and abs(rho_centred - recorded) < 5e-5
    print(f"rows {len(image_ids)}  d {matched.d}  controls {len(controls)}")
    print(f"GATE row alignment, refit vs dump : {alignment:.3e} "
          f"(tolerance {ALIGNMENT_TOLERANCE:g})")
    print(f"GATE rho reproduces the manifest  : {rho_centred:+.4f} vs {recorded:+.4f}")
    if not ok:
        print("\nFAILED: the gates did not pass, so the values below would be "
              "measuring the wrong rows. Nothing here should be reported.")
        return 1

    print("\nPOST-HOC diagnostic, not pre-specified. Values only, no verdicts.")
    print(f"  {observed:<34} rho_centred {rho_centred:+.4f}   "
          f"rho_plain {spearman_rho(dumped, plain, name='observed:plain'):+.4f}")
    for label, norm in (("rho_centred", centred), ("rho_plain", plain)):
        values = [
            spearman_rho(frame.loc[rows, c].to_numpy(dtype=np.float64), norm, name=c)
            for c in controls
        ]
        print(f"  {len(controls)} controls {label:<12} min {min(values):+.4f}  "
              f"median {float(np.median(values)):+.4f}  max {max(values):+.4f}")
        print(f"    all: {[round(v, 4) for v in values]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
