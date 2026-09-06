"""Linear-probe validation anchor.

Trains a multinomial logistic-regression probe on cached CIFAR-10 train CLS
embeddings, evaluates on cached CIFAR-10 test CLS embeddings, and compares
against the published DINOv2 ViT-B/14 CIFAR-10 linear-evaluation figure from
Oquab et al. (2023), Appendix Table 8.

The target accuracy and tolerance are committed constants below rather than CLI
arguments: a hardcoded value with a citation, living in git history, is a more
reproducible record than a flag that can change run to run with no audit trail.
The tolerance is decided before looking at the result, which is the whole point
of having one. Both remain available as optional overrides for debugging.

CPU-only by design; no GPU dependency.

Usage:
    python scripts/train_linear_probe.py \
        --embeddings-root <data-root>/embeddings/dinov2_vitb14/res224_bicubic \
        --out-dir results/probe \
        --thesis-repo-root <repo> \
        --openood-repo-root <openood-checkout>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# Published target, read directly from Oquab et al. (2023), "DINOv2: Learning
# Robust Visual Features without Supervision" (arxiv.org/abs/2304.07193),
# Appendix Table 8 ("Linear evaluation of frozen features on fine-grained
# benchmarks"), ViT-B/14 row, C10 (CIFAR-10) column.
# If this number is ever in question, re-check against the PDF directly.
PUBLISHED_VIT_B14_CIFAR10_LINEAR_EVAL_ACCURACY = 0.987

# Not measurement noise -- this covers hyperparameter mismatch against
# whatever exact training recipe (regularization strength, optimizer,
# possible per-dataset hyperparameter sweep) the published figure used,
# which this script's plain LogisticRegression(C=1.0) is not guaranteed to
# match. Decided before running, not fit to the result after the fact. A
# genuine extraction bug (wrong DINOv2 variant, wrong normalization, wrong
# resize/token selection) produces shortfalls of many points, not fractions
# of one, so this stays tight enough to still catch that class of bug.
DEFAULT_TOLERANCE = 0.02


def load_split(embeddings_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = embeddings_root / split
    cls = np.load(split_dir / "cls.npy")
    labels = np.load(split_dir / "labels.npy")
    assert cls.shape[0] == labels.shape[0], f"{split}: cls/labels row mismatch"
    return cls, labels


def git_commit_hash(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="cifar10_train")
    parser.add_argument("--test-split", type=str, default="cifar10_test")
    parser.add_argument(
        "--target-accuracy", type=float, default=PUBLISHED_VIT_B14_CIFAR10_LINEAR_EVAL_ACCURACY,
        help="DINOv2 ViT-B/14 published CIFAR-10 linear-eval accuracy (0-1 scale). "
             "Defaults to the committed value read from Oquab et al. (2023) Appendix "
             "Table 8 -- override only for debugging, and note if you do.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help="Acceptable absolute deviation from --target-accuracy. Defaults to the "
             "committed value, decided before ever looking at a probe result.",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (sklearn convention).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thesis-repo-root", type=Path, default=None)
    parser.add_argument("--openood-repo-root", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] loading {args.train_split} / {args.test_split} CLS embeddings", file=sys.stderr)
    X_train, y_train = load_split(args.embeddings_root, args.train_split)
    X_test, y_test = load_split(args.embeddings_root, args.test_split)

    print(f"[probe] train: {X_train.shape}, test: {X_test.shape}", file=sys.stderr)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=args.max_iter,
        C=args.C,
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))

    delta = test_acc - args.target_accuracy
    within_tolerance = abs(delta) <= args.tolerance

    print(f"[probe] train accuracy: {train_acc:.4f}", file=sys.stderr)
    print(f"[probe] test accuracy:  {test_acc:.4f}", file=sys.stderr)
    print(f"[probe] target (published ViT-B/14 CIFAR-10 linear eval): {args.target_accuracy:.4f}", file=sys.stderr)
    print(f"[probe] delta: {delta:+.4f} (tolerance ±{args.tolerance:.4f})", file=sys.stderr)
    print(f"[probe] WITHIN TOLERANCE: {within_tolerance}", file=sys.stderr)
    if not within_tolerance:
        print(
            "[probe] Large shortfall from the published figure usually means a bug in "
            "extraction (wrong variant, wrong normalization, wrong token, wrong resize) "
            "rather than a real limitation of the setup --  this is "
            "meant to be a tight check.",
            file=sys.stderr,
        )

    # Could be reused later. Save weights rather than the sklearn object, so the
    # probe stays readable without depending on this exact sklearn version.
    np.save(args.out_dir / "probe_coef.npy", clf.coef_.astype(np.float32))
    np.save(args.out_dir / "probe_intercept.npy", clf.intercept_.astype(np.float32))

    record = {
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "published_target_accuracy": args.target_accuracy,
        "published_target_source": (
            "Oquab et al. (2023), DINOv2 paper, Appendix Table 8 "
            "('Linear evaluation of frozen features on fine-grained benchmarks'), "
            "ViT-B/14 row, C10 (CIFAR-10) column -- confirm this citation matches "
            "what was actually read before treating this record as final."
        ),
        "target_accuracy_overridden_from_cli": (
            args.target_accuracy != PUBLISHED_VIT_B14_CIFAR10_LINEAR_EVAL_ACCURACY
        ),
        "tolerance_overridden_from_cli": args.tolerance != DEFAULT_TOLERANCE,
        "tolerance": args.tolerance,
        "delta": delta,
        "within_tolerance": within_tolerance,
        "sklearn_solver": "lbfgs",
        "sklearn_C": args.C,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "embedding_dim": int(X_train.shape[1]),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "thesis_repo_commit": (
            git_commit_hash(args.thesis_repo_root) if args.thesis_repo_root else None
        ),
        "openood_repo_commit": (
            git_commit_hash(args.openood_repo_root) if args.openood_repo_root else None
        ),
    }
    with open(args.out_dir / "probe_validation_record.json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"[probe] wrote {args.out_dir / 'probe_validation_record.json'}", file=sys.stderr)
    print(f"[probe] wrote probe_coef.npy / probe_intercept.npy to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
