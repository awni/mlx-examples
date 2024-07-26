
import torch
import numpy as np
import mlx.core as mx
from huggingface_hub import snapshot_download
import json
from pathlib import Path
from models.stdit3 import STDiT3

def fetch_from_hub(hf_repo: str, patterns=None) -> Path:
    default_patterns = [ "*.json", "*.safetensors"]
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=patterns or default_patterns,
        )
    )
    return model_path

path = fetch_from_hub("hpcai-tech/OpenSora-STDiT-v3")
with open(path / "config.json", 'r') as fid:
    config = json.load(fid)


model = STDiT3(**config)

def convert(key, value):
    if "rope.freqs" in key:
        if value.ndim == 4:
            value = value.moveaxis(1, 3)
        if value.ndim == 5:
            value = value.moveaxis(1, 4)
    if key.endswith("scale") or key.endswith("shift"):
        value = value.squeeze()
    return key, value

np.random.seed(0)
x = np.random.uniform(size=[2, 8, 30, 40, 4]).astype(np.float32)
timestep = np.array([1000., 1000.]).astype(np.float32)
mask = np.zeros((1, 20)).astype(np.int32)
mask[:10 ] = 1
x_mask = np.ones((2, 8)).astype(np.bool_)
y = np.random.uniform(size=[2, 1, 20, 4096])
height = np.array([240])
width = np.array([320])
fps = np.array([24])
from opensora.models.stdit import stdit3
pt_model = stdit3.STDiT3.from_pretrained("hpcai-tech/OpenSora-STDiT-v3")

inputs = [x, timestep, y, mask, x_mask, fps, height, width]
pt_inputs = list(map(torch.tensor, inputs))
pt_inputs[0] = pt_inputs[0].moveaxis(4, 1)

#pt_out = pt_model(*pt_inputs)

weights = mx.load(str(path / "model.safetensors"))
weights.pop("rope.freqs")
v = weights["x_embedder.proj.weight"]
weights["x_embedder.proj.weight"] = mx.moveaxis(v, 1, 4)

model.load_weights(list(weights.items()))
mx_out = model(*map(mx.array, inputs))
import pdb
pdb.set_trace()
