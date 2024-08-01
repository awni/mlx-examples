# Copyright © 2024 Apple Inc.

import json
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import models
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from transformers import AutoConfig, AutoModel, AutoTokenizer


def fetch_from_hub(hf_repo: str, patterns=None) -> Path:
    default_patterns = ["*.json", "*.safetensors"]
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=patterns or default_patterns,
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


def load_vae(hf_path):
    path = fetch_from_hub(hf_path)
    with open(path / "config.json", "r") as fid:
        config = json.load(fid)

    spatial_config_path = fetch_from_hub(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        patterns=["vae/*.json"],
    )
    with open(spatial_config_path / "vae/config.json", "r") as fid:
        spatial_config = json.load(fid)

    config["spatial_vae"] = spatial_config
    model = models.VideoAutoencoder(**config)

    def convert(key, value):
        if "conv" in key:
            if value.ndim == 4:
                value = value.moveaxis(1, 3)
            if value.ndim == 5:
                value = value.moveaxis(1, 4)
        if key.endswith("scale") or key.endswith("shift"):
            value = value.squeeze()
        return key, value

    weights = mx.load(str(path / "model.safetensors"))
    weights = [convert(k, v) for k, v in weights.items()]
    model.load_weights(weights)
    return model, config


def load_t5(hf_path):
    from transformers import T5EncoderModel

    config = AutoConfig.from_pretrained(hf_path).to_dict()
    model = T5EncoderModel.from_pretrained(hf_path, torch_dtype="auto")
    replacements = [
        (".layer.0.layer_norm.", ".ln1."),
        (".layer.1.layer_norm.", ".ln2."),
        (
            "block.0.layer.0.SelfAttention.relative_attention_bias.",
            "relative_attention_bias.embeddings.",
        ),
        (".layer.0.SelfAttention.", ".attention."),
        (".layer.1.DenseReluDense.", ".dense."),
    ]

    def replace(k):
        for o, n in replacements:
            k = k.replace(o, n)
        return k

    weights = model.state_dict()
    weights.pop("shared.weight")
    weights = [(replace(k), mx.array(v)) for k, v in weights.items()]
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    model = models.T5(**config)
    model.load_weights(weights)
    return model, tokenizer, config


def load_stdit(hf_path):
    path = fetch_from_hub(hf_path)
    with open(path / "config.json", "r") as fid:
        config = json.load(fid)
    weights = mx.load(str(path / "model.safetensors"))
    v = weights["x_embedder.proj.weight"]
    weights["x_embedder.proj.weight"] = mx.moveaxis(v, 1, 4)

    model = models.STDiT3(**config)

    model.load_weights(list(weights.items()))
    return model, config


def convert(
    model,
    config,
    save_path,
    tokenizer=None,
    dtype=None,
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    upload_repo: str = None,
    hf_path: str = None,
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

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)

    save_config(config, config_path=save_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(save_path, upload_repo, hf_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert OpenSora models to MLX")

    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )

    args = parser.parse_args()

    # Load and convert T5
    t5_hf_path = "DeepFloyd/t5-v1_1-xxl"
    t5_save_path = "mlx_models/t5-v1_1-xxl"
    t5_upload_repo = "mlx-community/DeepFloyd-t5-v1_1-xxl"
    model, tokenizer, config = load_t5(t5_hf_path)
    convert(
        model,
        config,
        save_path=t5_save_path,
        tokenizer=tokenizer,
        dtype=mx.bfloat16,  # Model doesn't work in fp16
        quantize=args.quantize,
        q_group_size=args.q_group_size,
        q_bits=args.q_bits,
        upload_repo=t5_upload_repo,
        hf_path=t5_hf_path,
    )

    # Load and convert VAE
    vae_hf_path = "hpcai-tech/OpenSora-VAE-v1.2"
    vae_save_path = "mlx_models/OpenSora-VAE-v1.2"
    vae_upload_repo = "mlx-community/OpenSora-VAE-v1.2"
    model, config = load_vae(vae_hf_path)
    convert(
        model,
        config,
        save_path=vae_save_path,
        dtype=mx.float16,
        quantize=args.quantize,
        q_group_size=args.q_group_size,
        q_bits=args.q_bits,
        upload_repo=vae_upload_repo,
        hf_path=vae_hf_path,
    )

    # Load and convert STDiT
    stdit_hf_path = "hpcai-tech/OpenSora-STDiT-v3"
    stdit_save_path = "mlx_models/OpenSora-STDiT-v3"
    stdit_upload_repo = "mlx-community/OpenSora-STDiT-v3"
    model, config = load_stdit(stdit_hf_path)
    convert(
        model,
        config,
        save_path=stdit_save_path,
        quantize=args.quantize,
        dtype=mx.float16,
        q_group_size=args.q_group_size,
        q_bits=args.q_bits,
        upload_repo=stdit_upload_repo,
        hf_path=stdit_hf_path,
    )
