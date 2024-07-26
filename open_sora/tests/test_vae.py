
import numpy as np
import mlx.core as mx
import json
from huggingface_hub import snapshot_download
from pathlib import Path

import diffusers.models
from autoencoder_kl import AutoencoderKL

def fetch_from_hub(hf_repo: str, patterns=None) -> Path:
    default_patterns = [ "*.json", "*.safetensors"]
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=patterns or default_patterns,
        )
    )
    return model_path

hf_repo = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
path = fetch_from_hub(hf_repo, patterns=["vae/*"])
#print(path)

with open(path / "vae/config.json", 'r') as fid:
    config = json.load(fid)
print(config)
exit(0)
weights = mx.load(str(path / "vae/diffusion_pytorch_model.safetensors"))
def torch_to_mlx(key, value):
    if "conv" in key and value.ndim == 4:
        value = value.moveaxis(1, 3)
    return key, value
weights = [torch_to_mlx(k, v) for k, v in weights.items()]

np.random.seed(0)
x = np.random.uniform(size=(2, 4, 32, 32)).astype(np.float32)
pt_model = diffusers.models.AutoencoderKL.from_pretrained(hf_repo, subfolder="vae")
import torch
#pt_y = pt_model.encode(torch.tensor(x))
pt_y = pt_model.decode(torch.tensor(x))
#print(pt_y.latent_dist.std)
#print(pt_y.latent_dist.mean)

model = AutoencoderKL(**config)
model.load_weights(weights)
mlx_y = model.encode(mx.array(np.moveaxis(x, 1, 3)))
mlx_y = model.decode(mx.array(np.moveaxis(x, 1, 3)))
