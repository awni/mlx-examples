# Copyright © 2024 Apple Inc.

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.append(".")

import diffusers.models
from models import Transformer3D
from utils import load_model

hf_repo = "THUDM/CogVideoX-5b"
mlx_repo = "mlx_cogvideox_models"


def compute_pt(inputs):
    model = diffusers.models.CogVideoXTransformer3DModel.from_pretrained(
        hf_repo, subfolder="transformer"
    ).to("mps")
    pt_inputs = dict(inputs)
    pt_inputs["hidden_states"] = np.moveaxis(inputs["hidden_states"], 4, 2)
    for k, v in pt_inputs.items():
        pt_inputs[k] = torch.tensor(v).to("mps") if v is not None else v
    output = model(**pt_inputs).sample
    return output.moveaxis(2, 4).detach().cpu().numpy()


def compute_mx(inputs):
    path = Path(mlx_repo) / "transformer"
    model = load_model(path, Transformer3D)
    mx.eval(model)
    mx_inputs = dict(inputs)
    for k, v in mx_inputs.items():
        mx_inputs[k] = mx.array(v) if v is not None else v
    return model(**mx_inputs)


np.random.seed(0)
inputs = {
    "hidden_states": np.random.uniform(size=[2, 2, 30, 45, 16]).astype(np.float32),
    "encoder_hidden_states": np.random.uniform(size=[2, 226, 4096]).astype(np.float32),
    "timestep": np.array([999, 999]),
    "timestep_cond": None,
    "image_rotary_emb": None,
}

pt_out = compute_pt(inputs)
mx_out = compute_mx(inputs)
import pdb

pdb.set_trace()
assert np.allclose(pt_out, mx_out, rtol=1e-4, atol=1e-2)
