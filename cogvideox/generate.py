# Copyright © 2024 Apple Inc.

import argparse

import fire
from transformers import AutoTokenizer
from utils import load, save_video


def generate(
    prompt: str,
    model_path: str,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Args:
        prompt (str): The description of the video to be generated.
        model_path (str): The path of the pre-trained model to be used.
        output_path (str): The path where the generated video will be saved.
        num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
        guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
        seed (int): PRNG seed.
    """
    if seed is not None:
        mx.random.seed(seed)

    # 1. Load the models

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B and `CogVideoXDPMScheduler` for CogVideoX-5B.
    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps,so 48 frames and will plus 1 frame for the first frame.
    # for diffusers `0.30.1` and after version, this should be 49.

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
        num_inference_steps=num_inference_steps,  # Number of inference steps
        num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.31.0` and after.
        use_dynamic_cfg=True,  ## This id used for DPM Sechduler, for DDIM scheduler, it should be False
        guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance, can set to 7 for DPM scheduler
        generator=torch.Generator().manual_seed(42),  # Set the seed for reproducibility
    ).frames[0]

    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    export_to_video(video, output_path, fps=8)


if __name__ == "__main__":
    fire.Fire(generate)
