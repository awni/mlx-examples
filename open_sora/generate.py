# Copyright © 2024 Apple Inc.

import argparse
from pathlib import Path

import mlx.core as mx
import models
from transformers import AutoTokenizer
from utils import get_image_size, get_num_frames, load_model, process_prompt, save_video


def parse_args(training=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, help="The text prompt.", required=True)
    parser.add_argument(
        "--seed", default=42, type=int, help="RNG seed for reproducibility."
    )
    parser.add_argument(
        "--resolution",
        default="240p",
        type=str,
        choices=["144p", "240p", "360p", "480p", "720p"],
    )
    parser.add_argument(
        "--save-path",
        default="sample.mp4",
        type=str,
        help="Path to save the video.",
    )
    parser.add_argument("--num-frames", default=51, type=str, help="number of frames")
    parser.add_argument("--fps", default=24, type=int, help="fps")
    parser.add_argument(
        "--image-size", default=None, type=int, nargs=2, help="image size"
    )
    parser.add_argument(
        "--aspect-ratio", default="9:16", type=str, help="aspect ratio (h:w)"
    )
    parser.add_argument(
        "--num-sampling-steps", default=30, type=int, help="sampling steps"
    )
    parser.add_argument(
        "--cfg-scale", default=7.0, type=float, help="balance between cond & uncond"
    )
    parser.add_argument("--aes", default=6.5, type=float, help="aesthetic score")
    parser.add_argument("--flow", default=None, type=float, help="flow score")
    parser.add_argument("--camera-motion", default=None, type=str, help="camera motion")
    return parser.parse_args()


def prepare_multi_resolution_info(image_size, num_frames, fps):
    if num_frames <= 1:
        raise ValueError(f"Invalid num_frames {num_frames}")
    height = image_size[0]
    width = image_size[1]
    num_frames = num_frames
    ar = image_size[0] / image_size[1]
    return dict(height=height, width=width, num_frames=num_frames, ar=ar, fps=fps)


def encode_text(text_encoder, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        max_length=300,
        truncation=True,
        add_special_tokens=True,
        return_tensors="mlx",
    )
    return text_encoder.encode(inputs["input_ids"])


def main():
    args = parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    print("[INFO] Loading models")
    hf_path = "mlx-community"
    vae = load_model(f"{hf_path}/OpenSora-VAE-v1.2", models.VideoAutoencoder)
    text_encoder = load_model(f"{hf_path}/DeepFloyd-t5-v1_1-xxl", models.T5)
    model = load_model(f"{hf_path}/OpenSora-STDiT-v3", models.STDiT3)
    tokenizer = AutoTokenizer.from_pretrained(f"{hf_path}/DeepFloyd-t5-v1_1-xxl")

    # == prepare video size ==
    if args.image_size is None:
        resolution = args.resolution
        aspect_ratio = args.aspect_ratio
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(args.num_frames)
    latent_size = vae.get_latent_size((num_frames, *image_size))

    scheduler = models.RFlow(
        num_sampling_steps=args.num_sampling_steps,
        cfg_scale=args.cfg_scale,
    )

    score_kwargs = {
        "aes": args.aes,
        "flow": args.flow,
        "camera_motion": args.camera_motion,
    }

    # == multi-resolution info ==
    model_args = prepare_multi_resolution_info(
        image_size,
        num_frames,
        args.fps,
    )

    prompt = process_prompt(args.prompt, score_kwargs)

    z = mx.random.normal(shape=(1, *latent_size, vae.out_channels))

    text_embeddings = encode_text(text_encoder, tokenizer, prompt)

    samples = scheduler.sample(
        model,
        z,
        text_embeddings,
        additional_args=model_args,
    )
    samples = vae.decode(samples, num_frames=num_frames)
    save_path = save_video(
        samples[0, 17:],
        fps=args.fps,
        save_path=str(args.save_path),
    )


if __name__ == "__main__":
    main()
