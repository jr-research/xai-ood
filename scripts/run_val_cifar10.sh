#!/usr/bin/env bash
set -euo pipefail

export TORCH_HOME=/mnt/data/jreutter/.cache/torch
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xai-ood
cd ~/xai-ood
mkdir -p logs

python scripts/extract_embeddings.py \
  --imglist /mnt/data/jreutter/OpenOOD/data/benchmark_imglist/cifar10/val_cifar10.txt \
  --data-root /mnt/data/jreutter/OpenOOD/data/images_classic \
  --out-root /mnt/data/jreutter/thesis-data/embeddings \
  --split-name cifar10_val --variant dinov2_vitb14 \
  --thesis-repo-root ~/xai-ood --openood-repo-root /mnt/data/jreutter/OpenOOD \
  2>&1 | tee "logs/extract_cifar10_val_$(date +%Y%m%d_%H%M).log"