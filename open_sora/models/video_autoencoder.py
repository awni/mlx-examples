from typing import List

import mlx.core as mx
import mlx.nn as nn
import os

from .vae_temporal import VAETemporal
from .autoencoder_kl import AutoencoderKL


class VideoAutoencoderKL(nn.Module):
    def __init__(
        self,
        config,
        micro_batch_size=4,
        scaling_factor=0.18215,
    ):
        super().__init__()
        self.module = AutoencoderKL(**config)
        self.out_channels = self.module.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size
        self.scaling_factor = scaling_factor

    def encode(self, x):
        # x: (B, T, H, W, C)
        B, T = x.shape[0:2]
        x = x.flatten(0, 1)

        bs = self.micro_batch_size
        x_out = []
        for i in range(0, x.shape[0], bs):
            x_bs = x[i : i + bs]
            x_bs = self.module.encode(x_bs).sample()
            x_out.append(x_bs)
        x = mx.concatenate(x_out, axis=0)
        x = x * self.scaling_factor
        x = x.reshape(B, T, *x.shape[1:])
        return x

    def decode(self, x, **kwargs):
        # x: (B, T, H, W, C)
        B, T = x.shape[0:2]
        x = x.flatten(0, 1)
        bs = self.micro_batch_size
        x_out = []
        for i in range(0, x.shape[0], bs):
            x_bs = x[i : i + bs]
            x_bs = self.module.decode(x_bs / self.scaling_factor)
            x_out.append(x_bs)
        x = mx.concatenate(x_out, axis=0)
        x = x.reshape(B, T, *x.shape[1:])
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size


class VideoAutoencoder(nn.Module):

    def __init__(
            self,
            micro_frame_size: int,
            scale: List[float],
            shift: List[float],
            vae_2d: dict,
            spatial_vae: dict,
            **kwargs,
    ):
        super().__init__()
        self.spatial_vae = VideoAutoencoderKL(
            micro_batch_size=vae_2d["micro_batch_size"],
            config=spatial_vae)
        self.temporal_vae = VAETemporal(
            temporal_downsample=(False, True, True),
        )
        self.micro_frame_size = micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size([micro_frame_size, None, None])[0]
        self.out_channels = self.temporal_vae.out_channels

        # normalization parameters
        self.scale = mx.array(scale)
        self.shift = mx.array(shift)

    def encode(self, x):
        x_z = self.spatial_vae.encode(x)

        z_list = []
        for i in range(0, x_z.shape[1], self.micro_frame_size):
            x_z_bs = x_z[:, i : i + self.micro_frame_size]
            posterior = self.temporal_vae.encode(x_z_bs)
            z_list.append(posterior.sample())
        z = mx.concatenate(z_list, axis=1)
        return (z - self.shift) / self.scale

    def decode(self, z, num_frames=None):
        z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        x_z_list = []
        for i in range(0, z.size(2), self.micro_z_frame_size):
            z_bs = z[:, :, i : i + self.micro_z_frame_size]
            x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))
            x_z_list.append(x_z_bs)
            num_frames -= self.micro_frame_size
        x_z = torch.cat(x_z_list, dim=2)
        x = self.spatial_vae.decode(x_z)
        return x

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))
        else:
            sub_input_size = [self.micro_frame_size, input_size[1], input_size[2]]
            sub_latent_size = self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(sub_input_size))
            sub_latent_size[0] = sub_latent_size[0] * (input_size[0] // self.micro_frame_size)
            remain_temporal_size = [input_size[0] % self.micro_frame_size, None, None]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(remain_temporal_size)
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size
