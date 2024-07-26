import numpy as np
import mlx.core as mx
import json
from huggingface_hub import snapshot_download
from pathlib import Path

def fetch_from_hub(hf_repo: str, patterns=None) -> Path:
    default_patterns = [ "*.json", "*.safetensors"]
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=patterns or default_patterns,
        )
    )
    return model_path

hf_repo = "hpcai-tech/OpenSora-VAE-v1.2"
path = fetch_from_hub(hf_repo)

from opensora.models.vae import OpenSoraVAE_V1_2
pt_model = OpenSoraVAE_V1_2(from_pretrained=str(path / "model.safetensors"))

with open(path / "config.json", 'r') as fid:
    config = json.load(fid)

hf_repo = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
config_path = fetch_from_hub(hf_repo, patterns=["vae/*.json"])

with open(config_path / "vae/config.json", 'r') as fid:
    spatial_config = json.load(fid)
config["spatial_vae"] = spatial_config

from video_autoencoder import VideoAutoencoder
model = VideoAutoencoder(**config)
weights = mx.load(str(path / "model.safetensors"))
def torch_to_mlx(key, value):
    if "conv" in key:
        if value.ndim == 4:
            value = value.moveaxis(1, 3)
        if value.ndim == 5:
            value = value.moveaxis(1, 4)
    if key.endswith("scale") or key.endswith("shift"):
        value = value.squeeze()
    return key, value
weights = [torch_to_mlx(k, v) for k, v in weights.items()]

np.random.seed(0)
x = np.random.uniform(size=(2, 3, 10, 32, 32)).astype(np.float32)
import torch
pt_y = pt_model.encode(torch.tensor(x))
#pt_y = pt_model.decode(torch.tensor(x))

model.load_weights(weights)
mlx_y = model.encode(mx.array(np.moveaxis(x, 1, 4)))
import pdb
pdb.set_trace()
