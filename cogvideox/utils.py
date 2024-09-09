# Copyright © 2024 Apple Inc.

import glob
import json
from pathlib import Path

import av
import mlx.core as mx
import mlx.nn as nn
import models
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def save_video(x, save_path=None, fps=8):
    """
    Save an MLX array as a video.

    Args:
        x (mx.array): shape [T, H, W, C]
    """

    def normalize(x):
        x = (mx.clip(x, a_min=-1.0, a_max=1.0) + 1.0) * (255.0 / 2.0)
        x = mx.clip(x + 0.5, a_min=0, a_max=255).astype(mx.uint8)
        return x

    x = np.array(normalize(x))

    with av.open(save_path, mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = x.shape[2]
        stream.height = x.shape[1]
        stream.pix_fmt = "yuv420p"

        for img in x:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


def load_model(path, model_class):
    with open(path / "config.json", "r") as f:
        config = json.load(f)

    weight_files = glob.glob(str(path / "model*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model = model_class(**config)

    if (quantization := config.get("quantization", None)) is not None:
        nn.quantize(model, **quantization)

    model.load_weights(list(weights.items()))
    return model


def load(path_or_repo):
    path = Path(path_or_repo)
    if not path.exists():
        path = Path(
            snapshot_download(
                repo_id=path_or_repo,
                allow_patterns=["*.json", "*.safetensors", "*.model"],
            )
        )

    text_encoder = load_model(path / "text_encoder", models.T5)
    vae = load_model(path / "vae", models.AutoencoderKL)
    transformer = load_model(path / "transformer", models.Transformer3D)
    # Silences process fork warnings from transformers
    import tqdm

    tqdm.tqdm([], disable=True)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    with open(path / "scheduler/config.json", "r") as f:
        scheduler_config = json.load(f)
        scheduler = models.CogVideoXDPMScheduler(**scheduler_config)
    return text_encoder, vae, transformer, tokenizer, scheduler
