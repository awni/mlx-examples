# Copyright © 2024 Apple Inc.

import glob
import json
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import models
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from transformers import AutoTokenizer


def fetch_from_hub(hf_repo: str) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
        )
    )
    return model_path


def make_shards(weights: dict, max_file_size_gb: int = 5) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    content = dedent(
        f"""
        ---
        language: en
        license: other
        library: mlx
        tags:
        - mlx
        ---

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was
        converted to MLX format from
        [{hf_path}](https://huggingface.co/{hf_path}).

        This model is intended to be used with the [Open-Sora MLX
        Example](https://github.com/ml-explore/mlx-examples/tree/main/open-sora).
        """
    )

    card = ModelCard(content)
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def load_weights(path: Union[str, Path]):
    weight_files = glob.glob(str(Path(path) / "*.safetensors"))
    weights = {}
    for w in weight_files:
        weights.update(mx.load(w))
    return weights


def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def quantize_model(
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.

    Returns:
        Tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)
    nn.quantize(model, q_group_size, q_bits)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}

    return model, quantized_config


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


# def load_stdit(hf_path):
#    path = fetch_from_hub(hf_path)
#    with open(path / "config.json", "r") as fid:
#        config = json.load(fid)
#    weights = mx.load(str(path / "model.safetensors"))
#    v = weights["x_embedder.proj.weight"]
#    weights["x_embedder.proj.weight"] = mx.moveaxis(v, 1, 4)
#
#    model = models.STDiT3(**config)
#
#    model.load_weights(list(weights.items()))
#    return model, config


def convert_model(
    model,
    config,
    save_path,
    dtype=None,
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
):
    if quantize:
        print("[INFO] Quantizing")
        model, config = quantize_model(model, config, q_group_size, q_bits)

    weights = dict(tree_flatten(model.parameters()))
    del model

    if dtype is not None:
        weights = {k: v.astype(dtype) for k, v in weights.items()}

    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_weights(save_path, weights, donate_weights=True)

    save_config(config, config_path=save_path / "config.json")


def convert_t5(path, save_path):
    replacements = [
        (".layer.0.layer_norm.", ".ln1."),
        (".layer.1.layer_norm.", ".ln2."),
        (
            "block.0.layer.0.SelfAttention.relative_attention_bias.",
            "relative_attention_bias.embeddings.",
        ),
        (".layer.0.SelfAttention.", ".attention."),
        (".layer.1.DenseReluDense.", ".dense."),
        ("shared.", "encoder.embed_tokens."),
    ]

    def replace(k):
        for o, n in replacements:
            k = k.replace(o, n)
        return k

    path = path / "text_encoder"

    with open(path / "config.json", "r") as fid:
        config = json.load(fid)
    weights = load_weights(path)
    weights = [(replace(k), mx.array(v)) for k, v in weights.items()]
    model = models.T5(**config)
    model.load_weights(weights)
    convert_model(model, config, save_path / "text_encoder")


def convert_vae(path, save_path):
    path = path / "vae"

    with open(path / "config.json", "r") as fid:
        config = json.load(fid)

    weights = load_weights(path)
    model = models.AutoencoderKL(**config)

    def convert(key, value):
        if "conv" in key:
            if value.ndim == 4:
                value = value.moveaxis(1, 3)
            if value.ndim == 5:
                value = value.moveaxis(1, 4)
        return key, value

    weights = [convert(k, v) for k, v in weights.items()]
    model.load_weights(weights)
    convert_model(model, config, save_path / "vae")


def convert_transformer(
    path,
    save_path,
    **kwargs,
):
    path = path / "transformer"

    with open(path / "config.json", "r") as fid:
        config = json.load(fid)

    def convert(k, v):
        if v.ndim == 4:
            v = v.moveaxis(1, 3)
        k = k.replace("to_out.0", "to_out")
        k = k.replace("net.2", "net.1")
        return k, v

    weights = load_weights(path)
    model = models.Transformer3D(**config)
    weights = [convert(*a) for a in weights.items()]
    model.load_weights(weights)
    convert_model(model, config, save_path / "transformer", **kwargs)


def convert_tokenizer(path, save_path):
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    tokenizer.save_pretrained(save_path / "tokenizer")


def convert_scheduler(path, save_path):
    with open(path / "scheduler/scheduler_config.json", "r") as fid:
        config = json.load(fid)
    save_path = save_path / "scheduler/"
    save_path.mkdir(exist_ok=True)
    save_config(config, save_path / "config.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CogVideoX models to MLX")
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--upload",
        help="Upload to Hugging Face.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    hf_repo = "THUDM/CogVideoX-5b"
    mlx_repo = "mlx-community/CogVideoX-5b-mlx"
    path = fetch_from_hub(hf_repo)
    save_path = Path("mlx_cogvideox_models")
    # Convert each model separately

    #    {'scheduler': ['diffusers', 'CogVideoXDDIMScheduler'] 'CogVideoXTransformer3DModel']

    #    convert_tokenizer(path, save_path)
    #    convert_t5(path, save_path)
    #    convert_vae(path, save_path)
    convert_transformer(
        path,
        save_path,
        quantize=args.quantize,
        q_group_size=args.q_group_size,
        q_bits=args.q_bits,
    )
    convert_scheduler(path, save_path)

    if args.upload:
        upload_to_hub(save_path, mlx_repo, hf_path)
