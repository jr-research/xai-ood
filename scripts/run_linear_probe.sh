#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xai-ood
cd ~/xai-ood
mkdir -p results/probe

python scripts/train_linear_probe.py \
  --embeddings-root /mnt/data/jreutter/thesis-data/embeddings/dinov2_vitb14/res224_bicubic \
  --out-dir results/probe \
  --thesis-repo-root ~/xai-ood