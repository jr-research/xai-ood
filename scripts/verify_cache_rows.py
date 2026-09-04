"""Cross-outage determinism and row correspondence on the cached embeddings.

Two questions, one extraction each, on the splits the linear probe never
covered. The probe validated cifar10 train and test end to end; a label or row
misalignment on any other split produces plausible AUROCs rather than an error.

Determinism: re-extract the first 256 images at batch size 256, which
reproduces the original run's first batch exactly, and compare bitwise against
the stored rows. This matters because the cs-ID embeddings will be extracted on
today's driver while the cached ID and OOD embeddings were extracted earlier on
a different one. If the two differ numerically, the full-spectrum leg compares
two slightly different spaces.

Correspondence: for each re-extracted row, the nearest row in the whole cached
array must be the row it came from. This is immune to floating-point noise, so
it stays valid even where a batch-shape difference makes bitwise equality fail.
A random sample is checked the same way, to cover positions beyond the head.

Writes nothing into the real cache; every extraction goes to a scratch root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SPLITS = (
    "cifar10_val", "cifar100", "tin",
    "mnist", "svhn", "texture", "places365",
)
HEAD_N = 256
RANDOM_N = 20


def extract_subset(
    script: Path, imglist_lines: list[str], data_root: Path,
    scratch: Path, split_name: str, batch_size: int,
) -> np.ndarray:
    """Run the real extraction script on a temporary imglist. Returns cls rows."""
    sub_imglist = scratch / f"{split_name}.txt"
    sub_imglist.write_text("\n".join(imglist_lines) + "\n")
    out_root = scratch / f"out_{split_name}"
    subprocess.run(
        [sys.executable, str(script),
         "--imglist", str(sub_imglist),
         "--data-root", str(data_root),
         "--out-root", str(out_root),
         "--split-name", split_name,
         "--variant", "dinov2_vitb14",
         "--batch-size", str(batch_size)],
        check=True, capture_output=True, text=True,
    )
    return np.load(out_root / "dinov2_vitb14" / "res224_bicubic" / split_name / "cls.npy")


    # Squared distance minus the query's own norm, which is constant across all
    # references for a given query and so cannot change the argmin. Dropping it
    # avoids materialising anything of size (n_query, n_reference).
def nearest_row(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Index of the closest reference row for each query row, in float64."""
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    ref_sq = np.einsum("ij,ij->i", r, r)
    out = np.empty(len(q), dtype=np.int64)
    for i in range(len(q)):
        d = ref_sq - 2.0 * (r @ q[i])
        out[i] = int(np.argmin(d))
    return out


def check_split(
    split: str, cache_root: Path, data_root: Path,
    script: Path, scratch: Path, rng: np.random.Generator,
) -> dict:
    split_dir = cache_root / split
    manifest = json.loads((split_dir / "manifest.json").read_text())
    cached = np.load(split_dir / "cls.npy")
    imglist = Path(manifest["imglist_path"]).read_text().splitlines()
    imglist = [line for line in imglist if line.strip()]

    if len(imglist) != len(cached):
        return {"split": split, "error": f"imglist {len(imglist)} vs cache {len(cached)}"}

    head = extract_subset(script, imglist[:HEAD_N], data_root, scratch,
                          f"{split}_head", HEAD_N)
    bitwise = bool(np.array_equal(head, cached[:HEAD_N]))
    max_abs = float(np.abs(head.astype(np.float64) - cached[:HEAD_N].astype(np.float64)).max())
    head_nn = nearest_row(head, cached)
    head_ok = bool(np.array_equal(head_nn, np.arange(HEAD_N)))

    picks = np.sort(rng.choice(len(cached), size=min(RANDOM_N, len(cached)), replace=False))
    rand = extract_subset(script, [imglist[i] for i in picks], data_root, scratch,
                          f"{split}_rand", RANDOM_N)
    rand_nn = nearest_row(rand, cached)
    rand_ok = bool(np.array_equal(rand_nn, picks))

    return {
        "split": split, "rows": len(cached),
        "bitwise": bitwise, "max_abs_diff": max_abs,
        "head_corr": head_ok, "rand_corr": rand_ok,
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--extract-script", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = []
    with tempfile.TemporaryDirectory(prefix="cacheverify_") as tmp:
        for split in args.splits:
            print(f"[verify] {split} ...", file=sys.stderr)
            results.append(check_split(split, args.cache_root, args.data_root,
                                       args.extract_script, Path(tmp), rng))

    header = f"{'split':<14}{'rows':>8}{'bitwise':>9}{'max|d|':>12}{'head':>7}{'rand':>7}"
    print(header)
    print("-" * len(header))
    failed = False
    for r in results:
        if r["error"]:
            print(f"{r['split']:<14}  ERROR: {r['error']}")
            failed = True
            continue
        ok = r["head_corr"] and r["rand_corr"]
        failed = failed or not ok
        print(f"{r['split']:<14}{r['rows']:>8}{str(r['bitwise']):>9}"
              f"{r['max_abs_diff']:>12.3e}{str(r['head_corr']):>7}{str(r['rand_corr']):>7}")
    print("-" * len(header))
    print("correspondence is the gate; bitwise False with tiny max|d| is a kernel "
          "difference, bitwise False with large max|d| is not.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()