"""Visual sanity/reference grid: N random raw images per extracted split.

Reads straight from the raw image files (via filelist.txt + labels.npy in the
embedding cache dirs), not from the cached embeddings themselves -- this is a
pixel-level check, independent of anything the extraction script computed.

Produces:
    results/figures/dataset_samples/<split_name>.png   (one grid per split)
    results/figures/dataset_samples/_overview.png       (one row per split, combined)

Usage:
    python scripts/visualize_samples.py \
        --embeddings-root /mnt/data/jreutter/thesis-data/embeddings/dinov2_vitb14/res224_bicubic \
        --data-root /mnt/data/jreutter/OpenOOD/data/images_classic \
        --out-dir results/figures/dataset_samples \
        --splits cifar10_train cifar10_test cifar10_val cifar100 tin mnist svhn texture places365
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_split(embeddings_root: Path, split: str) -> tuple[list[str], np.ndarray]:
    split_dir = embeddings_root / split
    labels = np.load(split_dir / "labels.npy")
    paths = (split_dir / "filelist.txt").read_text().splitlines()
    assert len(paths) == len(labels), f"{split}: filelist/labels length mismatch"
    return paths, labels


def label_str(split: str, label: int) -> str:
    if split.startswith("cifar10") and 0 <= label < 10:
        return CIFAR10_CLASSES[label]
    if label == -1:
        return "n/a"
    return str(label)


def sample_grid(
    data_root: Path,
    paths: list[str],
    labels: np.ndarray,
    split: str,
    n: int,
    seed: int,
) -> plt.Figure:
    rng = random.Random(seed)
    idxs = rng.sample(range(len(paths)), min(n, len(paths)))
    cols = min(n, 8)
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes = np.atleast_1d(axes).flatten()
    for ax, idx in zip(axes, idxs):
        img_path = data_root / paths[idx]
        try:
            with Image.open(img_path) as im:
                ax.imshow(im.convert("RGB"))
        except FileNotFoundError:
            ax.text(0.5, 0.5, "MISSING FILE", ha="center", va="center", color="red")
        ax.set_title(f"{label_str(split, int(labels[idx]))}\n{Path(paths[idx]).name}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.suptitle(split, fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", type=str, nargs="+", required=True)
    parser.add_argument("--n-per-split", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overview_rows = []
    for split in args.splits:
        try:
            paths, labels = load_split(args.embeddings_root, split)
        except FileNotFoundError as e:
            print(f"[visualize] SKIP {split}: {e}")
            continue

        fig = sample_grid(args.data_root, paths, labels, split, args.n_per_split, args.seed)
        out_path = args.out_dir / f"{split}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[visualize] wrote {out_path}")

        overview_rows.append((split, paths, labels))

    # Combined overview: one row per split, a handful of examples each.
    n_overview_cols = 6
    fig, axes = plt.subplots(
        len(overview_rows), n_overview_cols,
        figsize=(n_overview_cols * 1.6, len(overview_rows) * 1.8),
    )
    if len(overview_rows) == 1:
        axes = axes[None, :]
    rng = random.Random(args.seed)
    for row_idx, (split, paths, labels) in enumerate(overview_rows):
        idxs = rng.sample(range(len(paths)), min(n_overview_cols, len(paths)))
        for col_idx, idx in enumerate(idxs):
            ax = axes[row_idx, col_idx]
            img_path = args.data_root / paths[idx]
            try:
                with Image.open(img_path) as im:
                    ax.imshow(im.convert("RGB"))
            except FileNotFoundError:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center", color="red")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(split, fontsize=8, rotation=0, ha="right", va="center")
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
        for col_idx in range(len(idxs), n_overview_cols):
            axes[row_idx, col_idx].axis("off")
    fig.tight_layout()
    overview_path = args.out_dir / "_overview.png"
    fig.savefig(overview_path, dpi=120)
    plt.close(fig)
    print(f"[visualize] wrote {overview_path}")


if __name__ == "__main__":
    main()
