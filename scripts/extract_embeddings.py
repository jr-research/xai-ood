"""DINOv2 raw-embedding extraction.

Extracts CLS-token and mean-patch-token embeddings for a single dataset split
and caches them as:

    <out_root>/dinov2_vitb14[_reg]/res224_bicubic/<split_name>/
        cls.npy          (N, 768) float32
        patchmean.npy    (N, 768) float32
        labels.npy       (N,) int64
        filelist.txt     N lines, imglist order == row order
        manifest.json

Non-obvious choices:
    - float32 cache: compute may run under autocast, storage must not.
    - torch.inference_mode() and model.eval(); no gradient path anywhere.
    - shuffle=False and drop_last=False stated explicitly, never left to the
      DataLoader default.
    - filelist.txt is the ground truth for row order, not DataLoader order.
    - resize directly to 224 bicubic, no resize-then-crop. This diverges
      deliberately from DINOv2's own make_classification_eval_transform, which
      does Resize(256) then CenterCrop(224): on a 32x32 image that crop would
      discard about 12.5% of an image with no pixels to spare, so the whole
      image is kept instead. The linear-probe anchor reveals what that costs.
    - the patch-token mean excludes CLS and, for a _reg variant, the register
      tokens; the patch count is asserted to be 256 before averaging, so a
      wrong slice fails loudly instead of producing a plausible embedding.
    - normalization constants are passed via --norm-mean/--norm-std so the
      choice is visible in the invocation, and the manifest records what was
      actually used.

Normalization: the defaults are ImageNet's (0.485/0.456/0.406,
0.229/0.224/0.225). Verified against DINOv2's own source, which
defines IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD to exactly those values
in dinov2/data/transforms.py and uses them as the defaults of both
make_normalize_transform and make_classification_eval_transform.

Usage:
    python extract_embeddings.py \
        --imglist <benchmark_imglist>/cifar10/train_cifar10.txt \
        --data-root <data-root> \
        --out-root <data-root>/embeddings \
        --split-name cifar10_train \
        --variant dinov2_vitb14 \
        --thesis-repo-root <repo> \
        --openood-repo-root <openood-checkout>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PATCH_SIZE = 14
INPUT_RES = 224
EXPECTED_NUM_PATCHES = (INPUT_RES // PATCH_SIZE) ** 2  # 256

VALID_VARIANTS = {"dinov2_vitb14", "dinov2_vitb14_reg"}


class ImglistDataset(Dataset):
    """OpenOOD-style imglist: one 'relative/path label' pair per line.

    Row order == file order, and this class's __getitem__ order is what
    filelist.txt records. Do not let a sampler or shuffling touch this.
    """

    def __init__(self, imglist_path: Path, data_root: Path, transform):
        self.data_root = data_root
        self.transform = transform
        self.entries: list[tuple[str, int]] = []
        with open(imglist_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Malformed imglist line (expected 'path label'): {line!r}"
                    )
                rel_path, label_str = parts
                self.entries.append((rel_path, int(label_str)))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        rel_path, label = self.entries[idx]
        img_path = self.data_root / rel_path
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            tensor = self.transform(im)
        return tensor, label, rel_path


def build_transform(norm_mean, norm_std) -> transforms.Compose:
    return transforms.Compose(
        [
            # Direct resize to 224x224, bicubic, no intermediate crop.
            transforms.Resize(
                (INPUT_RES, INPUT_RES),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )


def load_dinov2(variant: str, device: torch.device):
    assert variant in VALID_VARIANTS, f"Unknown variant: {variant}"
    model = torch.hub.load("facebookresearch/dinov2", variant)
    model.eval()
    model.to(device)
    return model


def git_commit_hash(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_pin(variant: str, checkpoints_dir: Path | None = None) -> dict:
    """Content-addressed pin for the weights the backbone loaded.

    torch.hub unpacks a zipball, so there is no revision to record; the cached
    checkpoint's hash is the pin. The match is exact, not a substring, because
    ``dinov2_vitb14`` is a substring of ``dinov2_vitb14_reg_pretrain.pth``.
    Several matches hash to None rather than picking one.

    ``checkpoints_dir`` overrides the search directory, for testing.
    """
    checkpoints = (
        checkpoints_dir
        if checkpoints_dir is not None
        else Path(torch.hub.get_dir()) / "checkpoints"
    )
    matches = (
        sorted(
            p
            for p in checkpoints.glob("*.pth")
            if p.stem in (variant, f"{variant}_pretrain")
        )
        if checkpoints.is_dir()
        else []
    )
    resolved = matches[0] if len(matches) == 1 else None
    if resolved is not None:
        status = "hashed"
    elif matches:
        status = "ambiguous"
    else:
        status = "not-found"
    return {
        "weights_file": str(resolved) if resolved is not None else None,
        "weights_sha256": sha256_of_file(resolved) if resolved is not None else None,
        "weights_pin_status": status,
        "weights_candidates": [p.name for p in matches],
    }


def library_versions() -> dict:
    import torchvision

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


@torch.inference_mode()
def extract(
    model,
    loader: DataLoader,
    device: torch.device,
    use_registers: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cls_chunks: list[np.ndarray] = []
    patchmean_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    filelist: list[str] = []

    for images, labels, rel_paths in loader:
        images = images.to(device, non_blocking=True)

        # forward_features gives back normalized CLS + patch tokens already
        # separated from register tokens for the _reg variant, but assert
        # the patch count explicitly anyway rather than trust it.
        out = model.forward_features(images)
        cls_tok = out["x_norm_clstoken"]  # (B, 768)
        patch_tok = out["x_norm_patchtokens"]  # (B, num_patches, 768)

        num_patches = patch_tok.shape[1]
        if num_patches != EXPECTED_NUM_PATCHES:
            raise RuntimeError(
                f"Expected {EXPECTED_NUM_PATCHES} patch tokens at 224/14, got "
                f"{num_patches}. Do not silently average over the wrong "
                f"slice - this is the classic off-by-a-few-tokens bug."
            )

        patch_mean = patch_tok.mean(dim=1)  # (B, 768)

        # Cast to float32 before leaving GPU/before saving, regardless of
        # whether the forward pass ran under autocast.
        cls_chunks.append(cls_tok.float().cpu().numpy())
        patchmean_chunks.append(patch_mean.float().cpu().numpy())
        label_chunks.append(np.asarray(labels, dtype=np.int64))
        filelist.extend(rel_paths)

    cls_arr = np.concatenate(cls_chunks, axis=0)
    patchmean_arr = np.concatenate(patchmean_chunks, axis=0)
    labels_arr = np.concatenate(label_chunks, axis=0)

    assert cls_arr.dtype == np.float32
    assert patchmean_arr.dtype == np.float32
    assert len(filelist) == cls_arr.shape[0] == patchmean_arr.shape[0] == labels_arr.shape[0]

    return cls_arr, patchmean_arr, labels_arr, filelist


def sanity_check_no_nan_inf(*arrays: np.ndarray) -> None:
    for arr in arrays:
        if not np.isfinite(arr).all():
            raise RuntimeError("NaN or Inf found in extracted embeddings.")


def sanity_check_variance(arr: np.ndarray, name: str) -> None:
    variances = arr.var(axis=0)
    n_zero = int((variances == 0).sum())
    if n_zero > 0:
        raise RuntimeError(
            f"{name}: {n_zero} dimension(s) have zero variance across the "
            f"dataset. This will divide by zero in the diagonal baseline."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imglist", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split-name", type=str, required=True, help="e.g. cifar10_train")
    parser.add_argument("--variant", type=str, default="dinov2_vitb14", choices=sorted(VALID_VARIANTS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--norm-mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--norm-std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument("--thesis-repo-root", type=Path, default=None)
    parser.add_argument("--openood-repo-root", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    use_registers = args.variant.endswith("_reg")

    transform = build_transform(args.norm_mean, args.norm_std)
    dataset = ImglistDataset(args.imglist, args.data_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,       
        drop_last=False,     
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[extract] {len(dataset)} images, variant={args.variant}, device={device}", file=sys.stderr)

    model = load_dinov2(args.variant, device)

    # After the hub load, which populates the cache; before the extraction.
    # The arrays are written before the manifest, so refusing there is too late.
    pin = weights_pin(args.variant)
    if pin["weights_pin_status"] != "hashed":
        raise RuntimeError(
            f"Refusing to extract: weights pin is {pin['weights_pin_status']} "
            f"for {args.variant}. Candidates: {pin['weights_candidates']}"
        )

    cls_arr, patchmean_arr, labels_arr, filelist = extract(model, loader, device, use_registers)

    sanity_check_no_nan_inf(cls_arr, patchmean_arr)
    sanity_check_variance(cls_arr, "cls")
    sanity_check_variance(patchmean_arr, "patchmean")

    out_dir = args.out_root / args.variant / "res224_bicubic" / args.split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "cls.npy", cls_arr)
    np.save(out_dir / "patchmean.npy", patchmean_arr)
    np.save(out_dir / "labels.npy", labels_arr)
    with open(out_dir / "filelist.txt", "w") as f:
        f.write("\n".join(filelist) + "\n")

    manifest = {
        "split_name": args.split_name,
        "backbone_variant": args.variant,
        "uses_registers": use_registers,
        "weights_source": f"torch.hub facebookresearch/dinov2:{args.variant}",
        **pin,
        "input_resolution": INPUT_RES,
        "resize_mode": "direct-resize-224",
        "resize_is_deviation": True,
        "resize_reference_transform": "resize-256-then-center-crop-224",
        "interpolation": "bicubic",
        "normalization_mean": list(args.norm_mean),
        "normalization_std": list(args.norm_std),
        "token_selection": "cls token + mean of patch tokens (registers/cls excluded from mean)",
        "expected_num_patches": EXPECTED_NUM_PATCHES,
        "dtype": "float32",
        "row_count": int(cls_arr.shape[0]),
        "cls_shape": list(cls_arr.shape),
        "patchmean_shape": list(patchmean_arr.shape),
        "imglist_path": str(args.imglist),
        "imglist_sha256": sha256_of_file(args.imglist),
        "extraction_date_utc": datetime.now(timezone.utc).isoformat(),
        "thesis_repo_commit": (
            git_commit_hash(args.thesis_repo_root) if args.thesis_repo_root else None
        ),
        "openood_repo_commit": (
            git_commit_hash(args.openood_repo_root) if args.openood_repo_root else None
        ),
        "library_versions": library_versions(),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[extract] wrote {cls_arr.shape[0]} rows to {out_dir}", file=sys.stderr)
    print(f"[extract] embedding norm range (cls): "
          f"[{np.linalg.norm(cls_arr, axis=1).min():.3f}, "
          f"{np.linalg.norm(cls_arr, axis=1).max():.3f}]", file=sys.stderr)


if __name__ == "__main__":
    main()
