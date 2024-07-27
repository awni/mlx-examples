# Copyright © 2024 Apple Inc.

import sys

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.append(".")

from models import T5
from utils import load_model


def compute_pt(text):
    from opensora.models.text_encoder import T5Encoder

    model = T5Encoder("DeepFloyd/t5-v1_1-xxl", device="mps")
    out = model.encode(text)
    y, mask = out["y"], out["mask"]
    y = y.squeeze(0)
    return map(lambda x: x.cpu().detach().numpy(), (y, mask))


def compute_mx(text):
    text_encoder = load_model("mlx_models/t5-v1_1-xxl", T5)
    tokenizer = AutoTokenizer.from_pretrained("mlx_models/t5-v1_1-xxl")
    inputs = tokenizer(
        text,
        max_length=120,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="mlx",
    )
    input_ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    y = text_encoder.encode(input_ids, mask)
    return y, mask


text = "a beautiful waterfall"
pt_y, pt_mask = compute_pt(text)
mx_y, mx_mask = compute_mx(text)
assert np.allclose(pt_y.squeeze(0), mx_y, rtol=1e-4, atol=1e-1)
assert np.allclose(pt_mask, mx_mask)
