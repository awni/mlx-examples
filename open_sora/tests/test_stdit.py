# Copyright © 2024 Apple Inc.

import sys

import mlx.core as mx
import numpy as np
import torch

sys.path.append(".")

from models import STDiT3
from utils import load_model


def get_inputs():
    np.random.seed(0)
    x = np.random.uniform(size=[2, 8, 30, 40, 4]).astype(np.float32)
    timestep = np.array([1000.0, 1000.0]).astype(np.float32)
    mask = np.zeros((1, 20)).astype(np.int32)
    mask[:10] = 1
    x_mask = np.ones((2, 8)).astype(np.bool_)
    y = np.random.uniform(size=[2, 20, 4096])
    height = np.array([240])
    width = np.array([320])
    fps = np.array([24])
    return [x, timestep, y, mask, x_mask, fps, height, width]


def compute_pt(inputs):
    from opensora.models.stdit import stdit3

    model = stdit3.STDiT3.from_pretrained("hpcai-tech/OpenSora-STDiT-v3")
    inputs = list(map(torch.tensor, inputs))
    inputs[0] = inputs[0].moveaxis(4, 1)
    inputs[2] = inputs[2].unsqueeze(1)
    out = model(*inputs)
    return out.moveaxis(1, 4).detach().numpy()


def compute_mx(inputs):
    model = load_model("mlx_models/OpenSora-STDiT-v3", STDiT3)
    return model(*map(mx.array, inputs))


# Compare
inputs = get_inputs()
pt_out = compute_pt(inputs)
mx_out = compute_mx(inputs)
mx.eval(mx_out)
assert np.allclose(mx_out, pt_out, rtol=1e-3, atol=1e-1)
