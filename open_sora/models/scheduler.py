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
    height = model_kwargs["height"].astype(mx.float32)
    width = model_kwargs["width"].astype(mx.float32)
    num_frames = model_kwargs["num_frames"].astype(mx.float32)
    t = t / num_timesteps
    resolution = height * width
    ratio_space = (resolution / base_resolution).sqrt()
    num_frames = num_frames // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

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
        mask=None,
    ):
        guidance_scale = self.cfg_scale

        # prepare timesteps
        timesteps = [
            (1.0 - i / self.num_sampling_steps) * self.num_timesteps
            for i in range(self.num_sampling_steps)
        ]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [mx.array([t]) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(t, additional_args, num_timesteps=self.num_timesteps)
                for t in timesteps
            ]

        if mask is not None:
            noise_added = mask == 1

        for i, t in enumerate(timesteps):
            print("TIME STEP ", i)
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z
                x_noise = add_noise(
                    x0, mx.random.normal(shape=x0.shape), t, self.num_timesteps
                )
                mask_t_upper = mask_t >= t
                x_mask = mx.repeat(mask_t_upper, 2, axis=0)
                mask_add_noise = mask_t_upper & ~noise_added
                z = mx.where(mx.expand_dims(mask_add_noise, (2, 3, 4)), x_noise, x0)
                noise_added = mask_t_upper
            else:
                x_mask = None

            # classifier-free guidance
            z_in = mx.repeat(z, 2, axis=0)
            t = mx.repeat(t, 2, axis=0)
            pred = model(z_in, t, y=text_embeddings, x_mask=x_mask, **additional_args)
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

            if mask is not None:
                z = mx.where(mx.expand_dims(mask_t_upper, (2, 3, 4)), z, x0)

            # Eval after each time-step
            mx.eval(z)
        return z
