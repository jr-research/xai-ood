# xai-ood

BSc project on covariance-aware OOD detection.

## Setup

Two conda environments:

- `openood`: OpenOOD reproduction only (Phase 1 Step 1).
- `xai-ood`: thesis development, DINOv2 extraction, custom scorers, analysis (Phase 1 Step 1 onward).

Both are pinned to the same modern stack (Python 3.10, PyTorch 2.13.0+cu130, NumPy 1.26.4, OpenCV 5.0.0.93, faiss-gpu 1.14.3) via `env-lockfiles/constraints.txt`, because the older versions used originally didnt install cleanly on the workstation see my code-workstation-setup.md note for the details.

bash

```bash
conda env create -f environment.yml -n xai-ood
conda activate xai-ood
pip install -e .
```

## OpenOOD

OpenOOD was cloned separately, not into this repo:

bash

Pinned commit: `8d44375e4c695d03d2b97850b754f24fd4bda447` (tag `v1.5`), installed editable into both conda environments from this one checkout to avoid drift to a different OpenOOD version.

## Phase 1 Step 1: OpenOOD reproduction

MDS, RMDS, and kNN reproduced OpenOOD v1.5's published CIFAR-10 standard-protocol AUROC to within 0.01 points.


| Scorer | near-OOD published | near-OOD reproduced | delta | far-OOD published | far-OOD reproduced | delta |
| ------ | ------------------ | ------------------- | ----- | ----------------- | ------------------ | ----- |
| MDS    | 84.20 (±2.40)      | 84.20               | 0.00  | 89.72 (±1.36)     | 89.72              | 0.00  |
| RMDS   | 89.80 (±0.28)      | 89.80               | 0.00  | 92.20 (±0.21)     | 92.20              | 0.00  |
| kNN    | 90.64 (±0.20)      | 90.65               | 0.01  | 92.96 (±0.14)     | 92.96              | 0.00  |

  Published values: OpenOOD v1.5 (Zhang et al., 2024, *Journal of Data-centric Machine Learning Research*)

## Preliminary embedding cache layout 

```
<data-root>/embeddings/
└── dinov2_vitb14/
    └── res224_bicubic/
        ├── cifar10_train/
        │   ├── cls.npy
        │   ├── patchmean.npy
        │   ├── labels.npy
        │   ├── filelist.txt
        │   └── manifest.json
        ├── cifar10_test/
        └── ...
```

## License

MIT.
