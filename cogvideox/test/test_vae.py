# Copyright © 2024 Apple Inc.

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.append(".")

from models import AutoencoderKL
from utils import load_model

hf_repo = "THUDM/CogVideoX-5b"
mlx_repo = "mlx_cogvideox_models"


def compute_pt(x, y):
    import diffusers.models

    model = diffusers.models.AutoencoderKLCogVideoX.from_pretrained(
        hf_repo, subfolder="vae"
    )
    x, y = map(lambda z: torch.tensor(np.moveaxis(z, 4, 1)), (x, y))
    y_hat = model.encode(x).latent_dist.mean
    x_hat = model.decode(y).sample
    return map(
        lambda z: z.moveaxis(1, 4).detach().numpy(),
        (y_hat, x_hat),
    )


def compute_mx(x, y):
    path = Path(mlx_repo) / "vae"
    model = load_model(path, AutoencoderKL)
    y_hat = model.encode(mx.array(x)).mean
    x_hat = model.decode(mx.array(y))
    return y_hat, x_hat


np.random.seed(0)
x = np.random.uniform(size=(2, 10, 32, 32, 3)).astype(np.float32)
y = np.random.uniform(size=(2, 3, 4, 4, 16)).astype(np.float32)
pt_y, pt_x = compute_pt(x, y)
mx_y, mx_x = compute_mx(x, y)
assert np.allclose(pt_y, mx_y, rtol=1e-4, atol=1e-2)
assert np.allclose(pt_x, mx_x, rtol=1e-4, atol=1e-3)
