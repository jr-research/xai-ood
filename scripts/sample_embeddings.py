"""Copy a small stratified sample of the DINOv2 embedding cache off the workstation.

The embeddings live only where the cache is mounted. Every verification session so far has
hit the same wall: the score dump carries scores and not features, so no scorer can be
recomputed element-wise anywhere else. This script closes that permanently for a few
megabytes.

RUN THIS ON THE MACHINE THAT MOUNTS THE CACHE. It reads only, and writes one .npz.

    python3 sample_embeddings.py --cache /mnt/data/jreutter/thesis-data/embeddings \
                                 --out embedding-sample.npz

Then copy the .npz to the laptop. A thousand rows per split at 768 float32 is about
3 MB per split.

What it preserves, and why each matters:

  cls          the CLS embedding, which is what every scorer consumes
  image_id     so a sampled row joins to per_sample_scores.parquet and its published
               score can be reproduced element-wise
  label        needed for anything class-conditional
  split        so the sample can be stratified back out
  row_index    the position in the original cache, so the sample is auditable against
               the full cache later

It deliberately does NOT take a contiguous block. A head slice would miss the
class-ordering fault that the dry-run gate caught once already, where taking the first
N rows of a class-ordered file yielded a single class.
"""
import argparse
import pathlib

import numpy as np

# The fitting split first: the in-sample identity mean d^2 = p can only be checked on it,
# and no scores have ever been dumped for it.
DEFAULT_SPLITS = [
    "cifar10_train", "cifar10_test", "cifar100", "tin",
    "mnist", "svhn", "texture", "places365",
]


def read_split(directory):
    cls = np.load(directory / "cls.npy", mmap_mode="r")
    labels_path = directory / "labels.npy"
    labels = np.load(labels_path, mmap_mode="r") if labels_path.exists() else None
    filelist_path = directory / "filelist.txt"
    names = None
    if filelist_path.exists():
        names = filelist_path.read_text().splitlines()
    return cls, labels, names


def sample_indices(n_rows, n_take, labels, rng):
    """Spread the sample across labels where labels exist, at random within each."""
    if n_take >= n_rows:
        return np.arange(n_rows)
    if labels is None:
        return np.sort(rng.choice(n_rows, size=n_take, replace=False))
    labels = np.asarray(labels)
    classes = np.unique(labels)
    per_class = max(1, n_take // len(classes))
    picked = []
    for value in classes:
        members = np.flatnonzero(labels == value)
        take = min(per_class, len(members))
        picked.append(rng.choice(members, size=take, replace=False))
    picked = np.concatenate(picked)
    if len(picked) < n_take:  # top up at random from whatever is left
        remaining = np.setdiff1d(np.arange(n_rows), picked)
        extra = rng.choice(remaining, size=min(n_take - len(picked), len(remaining)), replace=False)
        picked = np.concatenate([picked, extra])
    return np.sort(picked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=pathlib.Path)
    parser.add_argument("--out", default="embedding-sample.npz", type=pathlib.Path)
    parser.add_argument("--per-split", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bundle, manifest = {}, []

    for split in args.splits:
        directory = args.cache / split
        if not directory.is_dir():
            print(f"  SKIP {split}: no directory at {directory}")
            continue
        cls, labels, names = read_split(directory)
        index = sample_indices(len(cls), args.per_split, labels, rng)
        bundle[f"{split}/cls"] = np.asarray(cls[index], dtype=np.float32)
        bundle[f"{split}/row_index"] = index.astype(np.int64)
        if labels is not None:
            bundle[f"{split}/label"] = np.asarray(labels)[index]
        if names is not None:
            bundle[f"{split}/image_id"] = np.array([names[i] for i in index], dtype=object)
        else:
            print(f"  WARNING {split}: no filelist.txt, so these rows CANNOT be joined to the "
                  f"score dump. The sample is much less useful without it.")
        manifest.append((split, len(cls), len(index)))
        print(f"  {split:<16} {len(index):>6} of {len(cls):>7} rows")

    if not bundle:
        raise SystemExit("Nothing sampled. Check --cache.")

    bundle["manifest"] = np.array(
        [f"{s}\t{total}\t{taken}" for s, total, taken in manifest], dtype=object)
    bundle["seed"] = np.array([args.seed])
    np.savez_compressed(args.out, **bundle)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB)")
    print("Copy this file to the laptop. Verify it by joining image_id against")
    print("per_sample_scores.parquet and recomputing one scorer element-wise.")


if __name__ == "__main__":
    main()
