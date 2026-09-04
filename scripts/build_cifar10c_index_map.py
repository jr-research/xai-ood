"""Build and verify the OpenOOD-path to CIFAR-10-C-row index map.

OpenOOD numbers cifar10/test images per class and 1-indexed, so a filename
carries no information about a photograph's position in the original CIFAR-10
test set. CIFAR-10-C is indexed entirely by that original position. Without an
explicit map, every corrupted image would pair with the wrong clean photograph
and nothing would raise.

The map is built by exact pixel match against the canonical CIFAR-10 test set,
which depends on no assumption about how the PNGs were exported. It is written
to CSV as a committed artefact rather than recomputed, so it cannot drift.

Phases:
  1. structural pre-check: per-class counts on disk and in labels.npy
  2. exact-hash map, with collision assertions in both directions
  3. CSV artefact
  4. contact sheet, for a quick visual check

Any deviation from the expected counts stops the run. The counts are named in
advance so this is a check and not a report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
N_TEST = 10_000
PER_CLASS = 1_000


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).hexdigest()


def phase1_structural(test_dir: Path, cifar10c_dir: Path) -> None:
    """Per-class counts on disk and in labels.npy."""
    print("[1] structural pre-check")
    for name in CLASSES:
        n = len(list((test_dir / name).glob("*.png")))
        if n != PER_CLASS:
            sys.exit(f"    FAIL {name}: {n} files on disk, expected {PER_CLASS}")
    print(f"    on disk: {len(CLASSES)} classes x {PER_CLASS} = {N_TEST} files")

    labels = np.load(cifar10c_dir / "labels.npy")
    print(f"    labels.npy shape {labels.shape}")
    block = labels[:N_TEST]
    if len(labels) % N_TEST != 0:
        sys.exit(f"    FAIL labels.npy length {len(labels)} is not a multiple of {N_TEST}")
    if len(labels) > N_TEST and not all(
        np.array_equal(labels[i * N_TEST:(i + 1) * N_TEST], block)
        for i in range(len(labels) // N_TEST)
    ):
        sys.exit("    FAIL labels.npy severity blocks are not identical; ordering assumption is wrong")
    counts = np.bincount(block, minlength=len(CLASSES))
    if not np.all(counts == PER_CLASS):
        sys.exit(f"    FAIL labels.npy per-class counts {counts.tolist()}, expected all {PER_CLASS}")
    print(f"    labels.npy: all {len(CLASSES)} classes at {PER_CLASS}, blocks consistent")


def phase2_hash_map(test_dir: Path, canonical_root: Path) -> dict[str, int]:
    """Exact pixel match. Asserts uniqueness in both directions."""
    print("[2] exact-hash map")
    from torchvision.datasets import CIFAR10

    canonical = CIFAR10(root=str(canonical_root), train=False, download=True)
    if len(canonical) != N_TEST:
        sys.exit(f"    FAIL canonical test set has {len(canonical)} images, expected {N_TEST}")

    by_hash: dict[str, list[int]] = defaultdict(list)
    for i in range(N_TEST):
        by_hash[sha(np.array(canonical[i][0]))].append(i)

    collisions = {h: v for h, v in by_hash.items() if len(v) > 1}
    if collisions:
        for _, idxs in list(collisions.items())[:5]: 
            print(f"    collision: original indices {idxs} are pixel-identical")
        sys.exit(f"    FAIL {len(collisions)} hash collision(s) in the canonical set; "
                 f"a plain dict build would silently mispair these")
    print(f"    canonical: {N_TEST} images, {len(by_hash)} distinct hashes, zero collisions")

    mapping: dict[str, int] = {}
    used: dict[int, str] = {}
    unmatched: list[str] = []
    for name in CLASSES:
        for png in sorted((test_dir / name).glob("*.png")):
            rel = f"cifar10/test/{name}/{png.name}"
            with Image.open(png) as im:
                digest = sha(np.array(im.convert("RGB")))
            hit = by_hash.get(digest)
            if hit is None:
                unmatched.append(rel)
                continue
            original = hit[0]
            if original in used:
                sys.exit(f"    FAIL original index {original} claimed by both "
                         f"{used[original]} and {rel}")
            used[original] = rel
            mapping[rel] = original

    print(f"    matched {len(mapping)} of {N_TEST} files, {len(unmatched)} unmatched")
    if unmatched:
        for rel in unmatched[:5]:
            print(f"    unmatched: {rel}")
        sys.exit("    FAIL every OpenOOD test PNG must match exactly one original index")
    return mapping


def phase3_csv(mapping: dict[str, int], id_imglist: Path, out_csv: Path) -> list[tuple[str, int]]:
    """Write the artefact and confirm the ID-test imglist resolves completely."""
    print("[3] CSV artefact")
    id_paths = [
        line.rsplit(" ", 1)[0]
        for line in id_imglist.read_text().splitlines()
        if line.strip()
    ]
    missing = [p for p in id_paths if p not in mapping]
    if missing:
        sys.exit(f"    FAIL {len(missing)} ID-test imglist entries have no mapping, "
                 f"first: {missing[:3]}")
    print(f"    ID-test imglist: {len(id_paths)} entries, all resolved")
    print(f"    remaining {N_TEST - len(id_paths)} originals are the reserved val split")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["imglist_path", "original_test_index"])
        for path in id_paths:
            writer.writerow([path, mapping[path]])
    print(f"    wrote {out_csv}")
    return [(p, mapping[p]) for p in id_paths]


def phase4_contact_sheet(
    pairs: list[tuple[str, int]],
    data_root: Path,
    cifar10c_dir: Path,
    out_png: Path,
    corruption: str,
    n: int,
) -> None:
    """Clean/corrupted pairs at severity 1."""
    print("[4] contact sheet")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corrupted = np.load(cifar10c_dir / f"{corruption}.npy", mmap_mode="r")
    step = max(1, len(pairs) // n)
    chosen = pairs[::step][:n]

    fig, axes = plt.subplots(2, len(chosen), figsize=(1.4 * len(chosen), 3.2))
    for col, (rel, original) in enumerate(chosen):
        with Image.open(data_root / rel) as im:
            axes[0][col].imshow(np.array(im.convert("RGB")))
        axes[1][col].imshow(np.asarray(corrupted[original]))
        axes[0][col].set_title(f"{original}", fontsize=6)
        for row in (0, 1):
            axes[row][col].axis("off")
    fig.suptitle(f"clean (top) vs {corruption} severity 1 (bottom); same photograph or not?")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"    wrote {out_png}  Inspect it before converting any images: each pair must be the same photograph.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="OpenOOD images_classic root")
    parser.add_argument("--cifar10c-dir", type=Path, required=True,
                        help="unpacked CIFAR-10-C directory holding the .npy files")
    parser.add_argument("--id-imglist", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True,
                        help="scratch dir for the canonical CIFAR-10 download")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-contact-sheet", type=Path, required=True)
    parser.add_argument("--corruption", type=str, default="brightness")
    parser.add_argument("--n-contact", type=int, default=10)
    args = parser.parse_args()

    test_dir = args.data_root / "cifar10" / "test"
    phase1_structural(test_dir, args.cifar10c_dir)
    mapping = phase2_hash_map(test_dir, args.canonical_root)
    pairs = phase3_csv(mapping, args.id_imglist, args.out_csv)
    phase4_contact_sheet(pairs, args.data_root, args.cifar10c_dir,
                         args.out_contact_sheet, args.corruption, args.n_contact)
    print("\nAll phases passed. The map is evidence, not an assumption.")


if __name__ == "__main__":
    main()