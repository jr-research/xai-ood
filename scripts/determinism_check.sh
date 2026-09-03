#!/usr/bin/env bash
set -euo pipefail

export TORCH_HOME=/mnt/data/jreutter/.cache/torch
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xai-ood
cd ~/xai-ood

DATA_ROOT=/mnt/data/jreutter/OpenOOD/data/images_classic
IMGLIST_DIR=/mnt/data/jreutter/OpenOOD/data/benchmark_imglist/cifar10

head -n 100 "$IMGLIST_DIR/test_cifar10.txt" > /tmp/det_check.txt

for i in 1 2; do
  python scripts/extract_embeddings.py \
    --imglist /tmp/det_check.txt --data-root "$DATA_ROOT" \
    --out-root /tmp/det_run$i --split-name det_check --variant dinov2_vitb14
done

python -c "
import numpy as np
a = np.load('/tmp/det_run1/dinov2_vitb14/res224_bicubic/det_check/cls.npy')
b = np.load('/tmp/det_run2/dinov2_vitb14/res224_bicubic/det_check/cls.npy')
print('bitwise identical:', np.array_equal(a, b))
"