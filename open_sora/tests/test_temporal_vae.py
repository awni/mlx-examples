# Copyright © 2024 Apple Inc.

import sys

import mlx.core as mx
import numpy as np
import torch

sys.path.append(".")

from models import VideoAutoencoder
from utils import load_model


def compute_pt(x, y):
    from opensora.models.vae import OpenSoraVAE_V1_2

    model = OpenSoraVAE_V1_2(from_pretrained="hpcai-tech/OpenSora-VAE-v1.2")
    model = model.temporal_vae
    x, y = map(lambda z: torch.tensor(np.moveaxis(z, 4, 1)), (x, y))
    y_hat = model.encode(x).mean
    x_hat = model.decode(y, num_frames=x.shape[2])
    return map(
        lambda z: z.moveaxis(1, 4).detach().numpy(),
        (y_hat, x_hat),
    )


def compute_mx(x, y):
    model = load_model("mlx_models/OpenSora-VAE-v1.2", VideoAutoencoder)
    model = model.temporal_vae
    y_hat = model.encode(mx.array(x)).mean
    x_hat = model.decode(mx.array(y), x.shape[1])
    return y_hat, x_hat


np.random.seed(0)
x = np.random.uniform(size=(2, 10, 4, 4, 4)).astype(np.float32)
y = np.random.uniform(size=(2, 4, 3, 4, 4)).astype(np.float32)

pt_y, pt_x = compute_pt(x, y)
mx_y, mx_x = compute_mx(x, y)
assert np.allclose(pt_y, mx_y, atol=1e-1, rtol=1e-3)
assert np.allclose(pt_x, mx_x, atol=1e-1, rtol=1e-3)
