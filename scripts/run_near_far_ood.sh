#!/usr/bin/env bash
set -euo pipefail

export TORCH_HOME=/mnt/data/jreutter/.cache/torch
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xai-ood
cd ~/xai-ood
mkdir -p logs

DATA_ROOT=/mnt/data/jreutter/OpenOOD/data/images_classic
IMGLIST_DIR=/mnt/data/jreutter/OpenOOD/data/benchmark_imglist/cifar10
OUT_ROOT=/mnt/data/jreutter/thesis-data/embeddings

for split in cifar100 tin mnist svhn texture places365; do
  python scripts/extract_embeddings.py \
    --imglist "$IMGLIST_DIR/test_${split}.txt" \
    --data-root "$DATA_ROOT" --out-root "$OUT_ROOT" \
    --split-name "${split}" --variant dinov2_vitb14 \
    --thesis-repo-root ~/xai-ood --openood-repo-root /mnt/data/jreutter/OpenOOD \
    2>&1 | tee "logs/extract_${split}_$(date +%Y%m%d_%H%M).log"
done