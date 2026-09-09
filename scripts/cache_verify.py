"""Verify the DINOv2 embedding cache.

Per split, checks that cls.npy, patchmean.npy, labels.npy and filelist.txt all
agree on row count, that they agree with the manifest's recorded row_count and
shapes, that neither embedding array contains NaN or Inf, and that the source
imglist still hashes to the value recorded at extraction time. Across splits,
checks that every manifest records the same OpenOOD checkout.

Exit status is non-zero if anything fails.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CACHE_ROOT = Path(
    "/mnt/data/jreutter/thesis-data/embeddings/dinov2_vitb14/res224_bicubic"
)


# Row counts as measured during extraction, hardcoded so this compares against
# a recorded value rather than against whatever is currently on disk.
EXPECTED_ROWS = {
    "cifar10_train": 50_000,
    "cifar10_test": 9_000,
    "cifar10_val": 1_000,
    "cifar100": 9_000,
    "tin": 7_793,
    "mnist": 70_000,
    "svhn": 26_032,
    "texture": 5_640,
    "places365": 35_195,
    "csid": 150_000,
}
EXPECTED_TOTAL = 363_660

# The OpenOOD checkout every split must have been extracted against. Named here
# rather than inferred from the manifests, so a cache built against a different
# checkout fails instead of silently redefining what agreement means.
EXPECTED_OPENOOD_COMMIT = "8d44375"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_split(name: str) -> tuple[int, list[str], str | None]:
    directory = CACHE_ROOT / name
    failures: list[str] = []

    if not (directory / "manifest.json").is_file():
        return 0, [f"no manifest under {directory}"], None

    manifest = json.loads((directory / "manifest.json").read_text())
    cls = np.load(directory / "cls.npy", mmap_mode="r")
    patch = np.load(directory / "patchmean.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    n_lines = sum(1 for _ in (directory / "filelist.txt").open())

    expected = EXPECTED_ROWS[name]
    for label, got in (
        ("cls.npy", len(cls)),
        ("patchmean.npy", len(patch)),
        ("labels.npy", len(labels)),
        ("filelist.txt", n_lines),
        ("manifest.row_count", manifest["row_count"]),
    ):
        if got != expected:
            failures.append(f"{label} has {got} rows, expected {expected}")

    for label, array, shape_key in (
        ("cls.npy", cls, "cls_shape"),
        ("patchmean.npy", patch, "patchmean_shape"),
    ):
        if list(array.shape) != manifest[shape_key]:
            failures.append(
                f"{label} shape {list(array.shape)} != manifest {manifest[shape_key]}"
            )
        if not np.isfinite(array).all():
            n_nan = int(np.isnan(array).sum())
            n_inf = int(np.isinf(array).sum())
            failures.append(f"{label} not finite: {n_nan} NaN, {n_inf} Inf")

    imglist = Path(manifest["imglist_path"])
    if not imglist.exists():
        failures.append(f"imglist missing: {imglist}")
    elif (got := sha256_of(imglist)) != manifest["imglist_sha256"]:
        failures.append(
            f"imglist sha256 {got[:12]} != recorded {manifest['imglist_sha256'][:12]}"
        )

    return len(cls), failures, manifest.get("openood_repo_commit")


def main() -> int:
    total = 0
    all_failures: dict[str, list[str]] = {}
    openood_commits: dict[str, str | None] = {}

    header = f"{'split':<16}{'rows':>8}{'expected':>10}  status"
    print(header)
    print("-" * len(header))

    for name in sorted(EXPECTED_ROWS):
        rows, failures, openood_commit = check_split(name)
        openood_commits[name] = openood_commit
        total += rows
        print(f"{name:<16}{rows:>8}{EXPECTED_ROWS[name]:>10}  "
              f"{'OK' if not failures else 'FAIL'}")
        if failures:
            all_failures[name] = failures

    print("-" * len(header))
    total_ok = total == EXPECTED_TOTAL
    print(f"{'TOTAL':<16}{total:>8}{EXPECTED_TOTAL:>10}  "
          f"{'OK' if total_ok else 'FAIL'}")

    commit_failures: list[str] = []
    missing = sorted(n for n, c in openood_commits.items() if not c)
    if missing:
        commit_failures.append(f"no openood_repo_commit recorded: {', '.join(missing)}")
    distinct = sorted({c for c in openood_commits.values() if c})
    if len(distinct) > 1:
        commit_failures.append(f"splits disagree on the openood commit: {distinct}")
    for commit in distinct:
        if not commit.startswith(EXPECTED_OPENOOD_COMMIT):
            commit_failures.append(f"openood commit {commit[:12]} is not {EXPECTED_OPENOOD_COMMIT}")
    shown = distinct[0][:12] if len(distinct) == 1 else "MIXED"
    print(f"{'OPENOOD':<16}{shown:>12}{EXPECTED_OPENOOD_COMMIT:>10}  "
          f"{'OK' if not commit_failures else 'FAIL'}")

    if all_failures or commit_failures:
        print("\nFAILURES")
        for line in commit_failures:
            print(f"  openood: {line}")
        for name, failures in all_failures.items():
            for line in failures:
                print(f"  {name}: {line}")

    return 0 if total_ok and not all_failures and not commit_failures else 1


if __name__ == "__main__":
    sys.exit(main())