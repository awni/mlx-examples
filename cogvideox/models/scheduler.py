from typing import List, Optional, Union

import mlx.core as mx
import numpy as np


def rescale_zero_terminal_snr(alphas_cumprod):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`mx.array`): Initial betas.

    Returns:
        `mx.array`: rescaled betas with zero terminal SNR
    """

    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0]
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1]

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt

    return alphas_bar


# TODO:
# Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
# Recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B and `CogVideoXDPMScheduler` for CogVideoX-5B.


class CogVideoXDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        snr_shift_scale: float = 3.0,
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset

        if trained_betas is not None:
            self.betas = mx.array(trained_betas)
        elif beta_schedule == "linear":
            self.betas = mx.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = mx.array(
                np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented.")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

        # Modify: SNR shift following SD3
        self.alphas_cumprod = self.alphas_cumprod / (
            snr_shift_scale + (1 - snr_shift_scale) * self.alphas_cumprod
        )

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.alphas_cumprod = rescale_zero_terminal_snr(self.alphas_cumprod)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = (
            mx.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger "
                f"than `self.num_train_timesteps`: {self.num_train_timesteps}."
            )

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(
                np.arange(self.num_train_timesteps, 0, -step_ratio)
            ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. "
                "Choose one of 'leading' or 'trailing'."
            )

        self.timesteps = timesteps.tolist()

    def get_variables(self, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back=None):
        lamb = ((alpha_prod_t / (1 - alpha_prod_t)) ** 0.5).log()
        lamb_next = ((alpha_prod_t_prev / (1 - alpha_prod_t_prev)) ** 0.5).log()
        h = lamb_next - lamb

        if alpha_prod_t_back is not None:
            lamb_previous = ((alpha_prod_t_back / (1 - alpha_prod_t_back)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back):
        mult1 = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * (-h).exp()
        mult2 = mx.expm1(-2 * h) * alpha_prod_t_prev**0.5

        if alpha_prod_t_back is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def step(
        self,
        model_output: mx.array,
        old_pred_original_sample: Optional[mx.array],
        timestep: int,
        timestep_back: Optional[int],
        sample: mx.array,
        eta: float = 0.0,
        variance_noise: Optional[mx.array] = None,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE.
        This function propagates the diffusion process from the learned model
        outputs (most often the predicted noise).

        Args:
            model_output (`mx.array`): The direct output from learned diffusion model.
            timestep (`float`): The current discrete timestep in the diffusion chain.
            sample (`mx.array`): A current instance of a sample created by the
                diffusion process.
            eta (`float`): The weight of noise for added noise in diffusion step.
            variance_noise (`mx.array`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', "
                "run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        alpha_prod_t_back = (
            self.alphas_cumprod[timestep_back] if timestep_back is not None else None
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one "
                " of `epsilon`, `sample`, or `v_prediction`"
            )

        h, r, lamb, lamb_next = self.get_variables(
            alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back
        )
        mult = list(
            self.get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
        )
        mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

        noise = mx.random.normal(shape=sample.shape)
        prev_sample = (
            mult[0] * sample - mult[1] * pred_original_sample + mult_noise * noise
        )

        if old_pred_original_sample is None or prev_timestep < 0:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return prev_sample, pred_original_sample
        else:
            denoised_d = (
                mult[2] * pred_original_sample - mult[3] * old_pred_original_sample
            )
            noise = mx.random.normal(shape=sample.shape)
            x_advanced = mult[0] * sample - mult[1] * denoised_d + mult_noise * noise

            prev_sample = x_advanced

        return (prev_sample, pred_original_sample)
