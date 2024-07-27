# Copyright © 2024 Apple Inc.

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.append(".")

from models.autoencoder_kl import AutoencoderKL
from utils import load_model

hf_repo = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"


def fetch_from_hub():
    patterns = ["vae/*.json", "vae/*.safetensors"]
    return Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=patterns,
        )
    )


def compute_pt(x, y):
    import diffusers.models

    model = diffusers.models.AutoencoderKL.from_pretrained(hf_repo, subfolder="vae")
    x, y = map(lambda z: torch.tensor(np.moveaxis(z, 3, 1)), (x, y))
    y_hat = model.encode(torch.tensor(x)).latent_dist.mean
    x_hat = model.decode(torch.tensor(y)).sample
    return map(
        lambda z: z.moveaxis(1, 3).detach().numpy(),
        (y_hat, x_hat),
    )


def compute_mx(x, y):
    path = fetch_from_hub()
    with open(path / "vae/config.json", "r") as fid:
        config = json.load(fid)

    weights = mx.load(str(path / "vae/diffusion_pytorch_model.safetensors"))

    def torch_to_mlx(key, value):
        if "conv" in key and value.ndim == 4:
            value = value.moveaxis(1, 3)
        return key, value

    weights = [torch_to_mlx(k, v) for k, v in weights.items()]

    model = AutoencoderKL(**config)
    model.load_weights(weights)
    y_hat = model.encode(mx.array(x)).mean
    x_hat = model.decode(mx.array(y))
    return y_hat, x_hat


np.random.seed(0)
x = np.random.uniform(size=(2, 32, 32, 3)).astype(np.float32)
y = np.random.uniform(size=(2, 32, 32, 4)).astype(np.float32)
pt_y, pt_x = compute_pt(x, y)
mx_y, mx_x = compute_mx(x, y)
assert np.allclose(pt_y, mx_y, rtol=1e-4, atol=1e-2)
assert np.allclose(pt_x, mx_x, rtol=1e-4, atol=1e-3)
