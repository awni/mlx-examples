# Copyright © 2024 Apple Inc.

import torch
import numpy as np
import mlx.core as mx

import sys
sys.path.append(".")

from utils import load_model
from models import STDiT3


def get_inputs():
    np.random.seed(0)
    x = np.random.uniform(size=[2, 8, 30, 40, 4]).astype(np.float32)
    timestep = np.array([1000., 1000.]).astype(np.float32)
    mask = np.zeros((1, 20)).astype(np.int32)
    mask[:10 ] = 1
    x_mask = np.ones((2, 8)).astype(np.bool_)
    y = np.random.uniform(size=[2, 20, 4096])
    height = np.array([240])
    width = np.array([320])
    fps = np.array([24])
    return [x, timestep, y, mask, x_mask, fps, height, width]


def compute_pt(inputs):
    from opensora.models.stdit import stdit3
    pt_model = stdit3.STDiT3.from_pretrained("hpcai-tech/OpenSora-STDiT-v3")
    pt_inputs = list(map(torch.tensor, inputs))
    pt_inputs[0] = pt_inputs[0].moveaxis(4, 1)
    pt_inputs[2] = pt_inputs[2].unsqueeze(1)
    pt_out = pt_model(*pt_inputs)
    return pt_out.moveaxis(1, 4).detach().numpy()

def compute_mx(inputs):
    # Load converted weights
    model = load_model("mlx_models/OpenSora-STDiT-v3", STDiT3)
    return model(*map(mx.array, inputs))

# Compare
inputs = get_inputs()
pt_out = compute_pt(inputs)
mx_out = compute_mx(inputs)
mx.eval(mx_out)
np.allclose(mx_out, pt_out, rtol=1e-3, atol=1e-1)
