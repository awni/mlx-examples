# Copyright © 2024 Apple Inc.

import argparse
import mlx.core as mx
from transformers import AutoTokenizer

import models
from pathlib import Path
from utils import (
    extract_prompts_loop,
    get_image_size,
    get_num_frames,
    load_model,
    prepare_multi_resolution_info,
    process_prompts,
    save_sample,
)

def parse_args(training=False):
    parser = argparse.ArgumentParser()

    # prompt
    parser.add_argument("--prompt", type=str, nargs="+", help="prompt list", required=True)

    parser.add_argument("--seed", default=42, type=int, help="RNG seed for reproducibility")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--model", default="mlx_models", type=str, help="path to the models")
    parser.add_argument("--resolution", default="240p", type=str, help="multi resolution")

    # output
    parser.add_argument("--save-dir", default="samples/", type=str, help="path to save generated samples")
    parser.add_argument("--num-sample", default=1, type=int, help="number of samples to generate for one prompt")

    # image/video
    parser.add_argument("--num-frames", default=51, type=str, help="number of frames")
    parser.add_argument("--fps", default=24, type=int, help="fps")
    parser.add_argument("--save-fps", default=24, type=int, help="save fps")
    parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")
    parser.add_argument("--frame-interval", default=1, type=int, help="frame interval")
    parser.add_argument("--aspect-ratio", default="9:16", type=str, help="aspect ratio (h:w)")

    # hyperparameters
    parser.add_argument("--num-sampling-steps", default=30, type=int, help="sampling steps")
    parser.add_argument("--cfg-scale", default=7.0, type=float, help="balance between cond & uncond")

    # reference
    parser.add_argument("--loop", default=1, type=int, help="loop")
    parser.add_argument("--condition-frame-length", default=5, type=int, help="condition frame length")
    parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
    parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
    parser.add_argument("--aes", default=6.5, type=float, help="aesthetic score")
    parser.add_argument("--flow", default=None, type=float, help="flow score")
    parser.add_argument("--camera-motion", default=None, type=str, help="camera motion")
    return parser.parse_args()

# TODO
def dframe_to_frame(num):
    assert num % 5 == 0, f"Invalid num: {num}"
    return num // 5 * 17


def append_generated(vae, generated_video, refs_x, mask_strategy, loop_i, condition_frame_length, condition_frame_edit):
    ref_x = vae.encode(generated_video)
    for j, refs in enumerate(refs_x):
        if refs is None:
            refs_x[j] = [ref_x[j]]
        else:
            refs.append(ref_x[j])
        if mask_strategy[j] is None or mask_strategy[j] == "":
            mask_strategy[j] = ""
        else:
            mask_strategy[j] += ";"
        mask_strategy[
            j
        ] += f"{loop_i},{len(refs)-1},-{condition_frame_length},0,{condition_frame_length},{condition_frame_edit}"
    return refs_x, mask_strategy


def parse_mask_strategy(mask_strategy):
    MASK_DEFAULT = ["0", "0", "0", "0", "1", "0"]
    mask_batch = []
    if mask_strategy == "" or mask_strategy is None:
        return mask_batch

    mask_strategy = mask_strategy.split(";")
    for mask in mask_strategy:
        mask_group = mask.split(",")
        num_group = len(mask_group)
        assert num_group >= 1 and num_group <= 6, f"Invalid mask strategy: {mask}"
        mask_group.extend(MASK_DEFAULT[num_group:])
        for i in range(5):
            mask_group[i] = int(mask_group[i])
        mask_group[5] = float(mask_group[5])
        mask_batch.append(mask_group)
    return mask_batch


def find_nearest_point(value, point, max_value):
    t = value // point
    if value % point > point / 2 and t < max_value // point - 1:
        t += 1
    return t * point

def apply_mask_strategy(z, refs_x, mask_strategys, loop_i, align=None):
    masks = []
    no_mask = True
    for i, mask_strategy in enumerate(mask_strategys):
        no_mask = False
        T = z.shape[1]
        mask = mx.ones(T)
        mask_strategy = parse_mask_strategy(mask_strategy)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
            if loop_id != loop_i:
                continue
            ref = refs_x[i][m_id]

            if m_ref_start < 0:
                # ref: [T, H, W, C]
                m_ref_start = ref.shape[0] + m_ref_start
            if m_target_start < 0:
                # z: [B, T, H, W, C]
                m_target_start = T + m_target_start
            if align is not None:
                m_ref_start = find_nearest_point(m_ref_start, align, ref.shape[1])
                m_target_start = find_nearest_point(m_target_start, align, T)
            m_length = min(m_length, T - m_target_start, ref.shape[0] - m_ref_start)
            z[i, m_target_start : m_target_start + m_length] = ref[:, m_ref_start : m_ref_start + m_length]
            mask[m_target_start : m_target_start + m_length] = edit_ratio
        masks.append(mask)
    if no_mask:
        return None
    return mx.stack(masks)


def encode_text(text_encoder, tokenizer, prompts, max_length):
    inputs = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="mlx",
    )
    input_ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    n = len(prompts)
    y = text_encoder.encode(input_ids, mask)
    y_null = text_encoder.null(n)
    return mx.concatenate([y, y_null], axis=0), mask


def main():
    args = parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    model_path = Path(args.model)
    print("[INFO] Loading models")
    vae = load_model(model_path / "OpenSora-VAE-v1.2", models.VideoAutoencoder)
    text_encoder = load_model(model_path / "t5-v1_1-xxl", models.T5)
    model = load_model(model_path / "OpenSora-STDiT-v3", models.STDiT3)
    tokenizer = AutoTokenizer.from_pretrained(model_path / "t5-v1_1-xxl")
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
    max_length = model.y_embedder.y_embedding.shape[0]

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

    prompts = args.prompt

    # == prepare reference ==
    reference_path = args.reference_path or [""] * len(prompts)
    mask_strategy = args.mask_strategy or [""] * len(prompts)
    if len(reference_path) != len(prompts):
        raise ValueError("Length of reference must be the same as prompts")
    if len(mask_strategy) != len(prompts):
        raise ValueError("Length of mask_strategy must be the same as prompts")

    # == prepare arguments ==
    fps = args.fps
    save_fps = args.save_fps or fps // args.frame_interval
    batch_size = args.batch_size
    num_sample = args.num_sample
    loop = args.loop
    condition_frame_edit = 0.0
    condition_frame_length = 5
    align = 5

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    score_kwargs = {
        "aes": args.aes,
        "flow": args.flow,
        "camera_motion": args.camera_motion,
    }

    # == Iter over prompts ==
    start_idx = 0
    for i in range(0, len(prompts), batch_size):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            len(batch_prompts), image_size, num_frames, fps,
        )

        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            save_paths = [
                save_dir / f"sample_{start_idx + idx:04d}-{k}"
                for idx in range(len(batch_prompts))
            ]

            batch_prompts = process_prompts(batch_prompts, score_kwargs)

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                mx.random.seed(1024) # TODO why seed here?
                z = mx.random.normal(
                    shape=(len(batch_prompts), *latent_size, vae.out_channels)
                )
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)

                text_embeddings, text_mask = encode_text(text_encoder, tokenizer, batch_prompts, max_length)
                # pass in the inputs and make sure all the args are set right
                samples = scheduler.sample(
                    model,
                    z,
                    text_embeddings,
                    text_mask,
                    additional_args=model_args,
                    mask=masks,
                )
                samples = vae.decode(samples, num_frames=num_frames)
                video_clips.append(samples)

            # == save samples ==
            for idx, batch_prompt in enumerate(batch_prompts):
                save_path = save_paths[idx]
                video = [video_clips[i][idx] for i in range(loop)]
                for i in range(1, loop):
                    video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                video = mx.concatenate(video, axis=1)
                save_path = save_sample(
                    video,
                    fps=save_fps,
                    save_path=str(save_path),
                )

        start_idx += len(batch_prompts)

if __name__ == "__main__":
    main()
