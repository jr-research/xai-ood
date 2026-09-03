import numpy as np
import random

d = "/mnt/data/jreutter/thesis-data/embeddings/dinov2_vitb14/res224_bicubic/texture"
labels = np.load(f"{d}/labels.npy")
paths = open(f"{d}/filelist.txt").read().splitlines()
assert len(paths) == len(labels)

for i in random.sample(range(len(paths)), 4):
    print(i, paths[i], labels[i])