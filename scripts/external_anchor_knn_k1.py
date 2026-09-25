"""Check knn_normalized at k = 1 against the published frozen-DINOv2 anchor.

The only number in this project checkable against another group's published
result on the same protocol. Everything else is validated internally, so this is
the one place the pipeline is tested against something it did not produce.

**Why this needs its own scoring pass, which the anchor record assumed it did
not.** That record says the six AUROCs "exist as a by-product" of the bootstrap's
singleton pass. They do not: the driver fits ``KNN_CONFIGURATIONS`` at
``DEFAULT_K``, which is 50, so the dumped ``knn_normalized`` column is k = 50 and
the anchor is k = 1. Comparing them would quietly restore the axis the record
removed in order to tighten its agreement band from 3 points to 2. ``fit_knn``
takes a ``k`` override that carries the reporting tier forward, so the correct
comparison costs one distance pass and no new library code.

**The configuration must be named every time this is cited.** It is
``knn_normalized`` at k = 1, which is **appendix** tier. The primary kNN
configuration is ``knn_unnormalized`` at k = 50. An anchor quoted beside a
headline number without its configuration named will be read as validating the
headline number, and it does not.

**The fingerprint is the primary check and the means are the backstop.** A
six-dataset mean is dominated by four far sets, three of them near the ceiling,
so it hides exactly the near-OOD movement that matters. The shape carries the
weight: SVHN weakest of all six and below both near-OOD sets, which is the
signature of a frozen encoder scoring outside its pretraining distribution, and
CIFAR-100 below Tiny ImageNet.

**It is a floor, not a target.** The anchor is ViT-S/14 and this project is
ViT-B/14. Within DINOv2 the S to B step improves every published downstream
measurement, so these numbers should land at or above the anchor. Landing
meaningfully below it is the informative failure and the one this is good at
catching.

    python3 external_anchor_knn_k1.py --cache-root DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from xai_ood.methods.knn import fit_knn
from xai_ood.metrics import auroc

#: arXiv:2311.17093 Table 1, the frozen DINOv2 ViT-S/14 row, k = 1, 50,000
#: reference images, unweighted per-dataset means. Keyed by this project's split
#: names so the two can be read against each other without translation.
ANCHOR: dict[str, float] = {
    "cifar100": 94.08,
    "tin": 97.58,
    "svhn": 93.65,
    "places365": 98.88,
    "mnist": 99.88,
    "texture": 99.96,
}

NEAR: tuple[str, ...] = ("cifar100", "tin")
FAR: tuple[str, ...] = ("mnist", "svhn", "texture", "places365")

#: The plausibility band on the six-dataset unweighted mean, from two
#: independent groups. A band, explicitly not a measurement of anything.
BAND: tuple[float, float] = (96.0, 99.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--driver-dir", type=Path, default=Path("scripts"))
    parser.add_argument("--id-cache-split", default="cifar10_test")
    options = parser.parse_args(argv)

    sys.path.insert(0, str(options.driver_dir))
    import run_phase2 as driver

    train = driver.load_split(options.cache_root, driver.TRAIN_CACHE_SPLIT, "cls")
    scorer = fit_knn("knn_normalized", train.embeddings, k=1)
    print(f"configuration: knn_normalized, k = {scorer.config.k}, "
          f"normalize = {scorer.config.normalize}, tier = "
          f"{scorer.config.reporting_tier}, reference rows = {len(train.paths)}")

    # AUROC is invariant to row order within a split, so the embeddings are read
    # straight from the cache rather than gathered into the dump's order.
    id_scores = scorer.score(
        driver.load_split(options.cache_root, options.id_cache_split, "cls").embeddings
    )
    measured: dict[str, float] = {}
    print(f"\n{'dataset':<12}{'ours':>9}{'anchor':>9}{'delta':>9}")
    for dataset in ANCHOR:
        ood = driver.load_split(options.cache_root, dataset, "cls").embeddings
        value = 100.0 * float(auroc(id_scores, scorer.score(ood)))
        measured[dataset] = value
        print(f"{dataset:<12}{value:>9.2f}{ANCHOR[dataset]:>9.2f}"
              f"{value - ANCHOR[dataset]:>+9.2f}")

    near = float(np.mean([measured[d] for d in NEAR]))
    far = float(np.mean([measured[d] for d in FAR]))
    six = float(np.mean(list(measured.values())))
    print(f"\nnear-OOD mean {near:.2f} against floor 95.80  "
          f"({near - 95.80:+.2f})")
    print(f"far-OOD  mean {far:.2f} against floor 98.10  ({far - 98.10:+.2f})")
    print(f"six-dataset mean {six:.2f}, band {BAND[0]} to {BAND[1]}: "
          f"{BAND[0] <= six <= BAND[1]}")

    print("\nfingerprint, the primary check:")
    weakest = min(measured, key=measured.get)
    svhn_lowest = weakest == "svhn"
    svhn_below_near = all(measured["svhn"] < measured[d] for d in NEAR)
    c100_below_tin = measured["cifar100"] < measured["tin"]
    print(f"  SVHN is the weakest of all six          : {svhn_lowest} "
          f"(weakest is {weakest})")
    print(f"  SVHN sits below both near-OOD datasets  : {svhn_below_near}")
    print(f"  CIFAR-100 sits below Tiny ImageNet      : {c100_below_tin}")

    print("\nfailure conditions from the anchor record:")
    print(f"  any value near or below 50 (sign flip)  : "
          f"{any(v <= 55 for v in measured.values())}")
    print(f"  near-OOD below 92                       : {near < 92}")
    print(f"  far-OOD below 96                        : {far < 96}")
    print(f"  near-OOD above 99 (leakage on this protocol): {near > 99}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
