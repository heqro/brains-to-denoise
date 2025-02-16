# Stupid converter from .pt to .npy
import numpy as np
import torch

for i in range(1, 10):
    img_gt = torch.load(f"../dataset/im_{i}/gt.pt", weights_only=False)
    np.save(file=f"../dataset/im_{i}/gt", arr=img_gt.numpy())
    for noise in ["0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40"]:
        img_noisy = torch.load(f"../dataset/im_{i}/Std{noise}.pt", weights_only=False)
        np.save(file=f"../dataset/im_{i}/Std{noise}", arr=img_noisy.numpy())
