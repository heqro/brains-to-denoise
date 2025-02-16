# Stupid script to contaminate an image
import numpy as np
import torch
import sys
import os

sys.path.append("../libs/image-utils/src")
from my_io import print_image
from noise import add_rician_noise


for noise in ["0.25", "0.30", "0.35", "0.40"]:
    for i in range(1, 10):
        img_gt = torch.load(f"../dataset/im_{i}/gt.pt", weights_only=False)
        print(f"{i} dim: {img_gt.shape}")
        noisy = add_rician_noise(img_gt, float(noise))
        print_image(
            (noisy.numpy()[0] * 255).astype(np.uint8),
            f"../dataset/im_{i}/Std{noise}.png",
        )
        torch.save(noisy, f"../dataset/im_{i}/Std{noise}.pt")
