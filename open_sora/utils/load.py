# Copyright © 2024 Apple Inc.

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download


def load_model(path, model_class):

    path = Path(
        snapshot_download(
            repo_id=path,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    with open(path / "config.json", "r") as f:
        config = json.load(f)

    weight_files = glob.glob(str(path / "model*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model = model_class(**config)

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))
    model.eval()
    return model
