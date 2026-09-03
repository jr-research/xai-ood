#!/usr/bin/env bash
set -euo pipefail

export TORCH_HOME=/mnt/data/jreutter/.cache/torch
mkdir -p "$TORCH_HOME"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xai-ood

cd ~/xai-ood
mkdir -p logs

DATA_ROOT=/mnt/data/jreutter/OpenOOD/data/images_classic
IMGLIST_DIR=/mnt/data/jreutter/OpenOOD/data/benchmark_imglist/cifar10
OUT_ROOT=/mnt/data/jreutter/thesis-data/embeddings

python scripts/extract_embeddings.py \
  --imglist "$IMGLIST_DIR/train_cifar10.txt" \
  --data-root "$DATA_ROOT" --out-root "$OUT_ROOT" \
  --split-name cifar10_train --variant dinov2_vitb14 \
  --thesis-repo-root ~/xai-ood --openood-repo-root /mnt/data/jreutter/OpenOOD \
  2>&1 | tee "logs/extract_cifar10_train_$(date +%Y%m%d_%H%M).log"

python scripts/extract_embeddings.py \
  --imglist "$IMGLIST_DIR/test_cifar10.txt" \
  --data-root "$DATA_ROOT" --out-root "$OUT_ROOT" \
  --split-name cifar10_test --variant dinov2_vitb14 \
  --thesis-repo-root ~/xai-ood --openood-repo-root /mnt/data/jreutter/OpenOOD \
  2>&1 | tee "logs/extract_cifar10_test_$(date +%Y%m%d_%H%M).log"