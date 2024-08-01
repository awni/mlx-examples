# Copyright © 2024 Apple Inc.

import mlx.core as mx


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    height = model_kwargs["height"]
    width = model_kwargs["width"]
    num_frames = model_kwargs["num_frames"]
    t = t / num_timesteps
    resolution = height * width
    ratio_space = (resolution / base_resolution) ** 0.5
    num_frames = num_frames // 17 * 5
    ratio_time = (num_frames / base_num_frames) ** 0.5

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


def add_noise(
    original_samples: mx.array,
    noise: mx.array,
    timesteps: mx.array,
    num_timesteps: int,
):
    timepoints = timesteps / num_timesteps
    return (1 - timepoints) * original_samples + timepoints * noise


class RFlow:
    def __init__(
        self,
        num_sampling_steps=30,
        num_timesteps=1000,
        cfg_scale=7.0,
        use_discrete_timesteps=False,
        use_timestep_transform=True,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

    def sample(
        self,
        model,
        z,
        text_embeddings,
        additional_args=None,
    ):
        guidance_scale = self.cfg_scale
        dtype = model.y_embedder.y_embedding.dtype

        # Add in null embeddings for CFG
        text_embeddings = mx.concatenate(
            [
                text_embeddings.astype(dtype),
                model.y_embedder.y_embedding[: text_embeddings.shape[1]][None],
            ],
            axis=0,
        )

        z = z.astype(dtype)

        # prepare timesteps
        timesteps = [
            (1.0 - i / self.num_sampling_steps) * self.num_timesteps
            for i in range(self.num_sampling_steps)
        ]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(t, additional_args, num_timesteps=self.num_timesteps)
                for t in timesteps
            ]

        for i, t in enumerate(timesteps):
            print(
                f"[INFO] Sampling iteration {i+1}/{self.num_sampling_steps} ...",
                end="\r",
            )
            # classifier-free guidance
            z_in = mx.repeat(z, 2, axis=0)
            t = mx.repeat(mx.array([t]), 2, axis=0)
            pred = model(z_in, t, y=text_embeddings, x_mask=None, **additional_args)
            pred = mx.split(pred, 2, axis=-1)[0]
            pred_cond, pred_uncond = mx.split(pred, 2, axis=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = (
                timesteps[i] - timesteps[i + 1]
                if i < len(timesteps) - 1
                else timesteps[i]
            )
            dt = dt / self.num_timesteps
            z = z + v_pred * dt

            # Eval after each time-step
            mx.eval(z)

        print("\033[K", end="")
        print("[INFO] Done sampling.")
        return z
