# Copyright © 2024 Apple Inc.

import math
from typing import List, Union

import fire
import mlx.core as mx
import models
import tqdm
import utils


def encode_prompt(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 226,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Union[str, List[str], None] = None,
):

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if do_classifier_free_guidance:
        negative_prompt = negative_prompt or ""
        negative_prompt = (
            len(prompt) * [negative_prompt]
            if isinstance(negative_prompt, str)
            else negative_prompt
        )

        if len(prompt) != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt` has batch size {len(negative_prompt)}, but `prompt`"
                f"has batch size {len(prompt)}."
            )
        prompt.extend(negative_prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="mlx",
    ).input_ids
    return text_encoder.encode(text_inputs)


def generate(
    prompt: str,
    model_path: str = "mlx-community/CogVideoX-5b",
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = None,
    num_frames: int = 49,
):
    """
    Generates a video based on the given prompt and saves it to the specified
    path.

    Args:
        prompt (str): The description of the video to be generated.
        model_path (str): The path of the pre-trained model to be used.
        output_path (str): The path where the generated video will be saved.
        num_inference_steps (int): Number of steps for the inference process.
            More steps can result in better quality.
        guidance_scale (float): The scale for classifier-free guidance. Higher
            values can lead to better alignment with the prompt.
        seed (int): PRNG seed.
        num_frames (int): Number of frames to generate. Default: `49`.
    """
    if seed is not None:
        mx.random.seed(seed)

    # Load the models
    text_encoder, vae, transformer, tokenizer, scheduler = utils.load(model_path)

    # Setup scheduler
    use_dynamic_cfg = isinstance(scheduler, models.CogVideoXDPMScheduler)
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    # Encode input prompt
    prompt_embeds = encode_prompt(
        text_encoder,
        tokenizer,
        prompt,
        do_classifier_free_guidance=(guidance_scale > 1.0),
    )

    # Unload text encoder before evaluating the embeddings
    del text_encoder
    mx.eval(prompt_embeds)

    # Prepare latents
    height = 480
    width = 720
    vae_scale_factor_spatial = 2 ** (len(vae.block_out_channels) - 1)
    latents = mx.random.normal(
        shape=(
            1 if isinstance(prompt, str) else len(prompt),
            (num_frames - 1) // vae.temporal_compression_ratio + 1,
            height // vae_scale_factor_spatial,
            width // vae_scale_factor_spatial,
            transformer.in_channels,
        )
    )

    # Prepare positional encodings
    image_rotary_emb = transformer.compute_rope_freqs(
        height,
        width,
        latents.shape[1],
        vae_scale_factor_spatial,
    )

    old_pred_original_sample = None

    # Run the denoising loop
    for i, t in enumerate(tqdm.tqdm(timesteps)):
        latent_input = latents
        if guidance_scale > 1.0:
            latents_input = mx.repeat(latents, 2, axis=0)

        # predict noise model_output
        noise_pred = transformer(
            hidden_states=latents_input,
            encoder_hidden_states=prompt_embeds,
            timestep=mx.array(t)[None],
            image_rotary_emb=image_rotary_emb,
        )

        # perform guidance
        if guidance_scale > 1.0:
            gd = guidance_scale
            if use_dynamic_cfg:
                gd = 1 + gd * (
                    (
                        1
                        - math.cos(
                            math.pi
                            * ((num_inference_steps - t) / num_inference_steps) ** 5.0
                        )
                    )
                    / 2
                )
            noise_pred_uncond, noise_pred_text = noise_pred.split(2, axis=0)
            noise_pred = noise_pred_uncond + gd * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        if use_dynamic_cfg:
            latents, old_pred_original_sample = scheduler.step(
                noise_pred,
                old_pred_original_sample,
                t,
                timesteps[i - 1] if i > 0 else None,
                latents,
            )
        else:
            latents = scheduler.step(noise_pred, t, latents)[0]
        # eval the graph
        mx.eval(latents)

    # Unload the transformer
    del transformer

    # Decode to video
    latents = (1 / vae.scaling_factor) * latents
    frames = vae.decode(latents).squeeze(0)

    # Export the generated frames to a video file
    utils.save_video(frames, output_path, fps=8)


if __name__ == "__main__":
    fire.Fire(generate)
