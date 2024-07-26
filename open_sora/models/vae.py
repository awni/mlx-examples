from typing import Optional, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D, UNetMidBlock2D

class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        num_blocks: int = 4,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = []

        # down
        output_channel = block_out_channels[0]
        for i in range(num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_padding=0,
                resnet_time_scale_shift="default",
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_groups=norm_num_groups, dims=block_out_channels[-1], eps=1e-6,
            pytorch_compatible=True)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)


    def __call__(self, sample: mx.array):

        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        num_blocks: int = 4,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = []

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(num_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift="group",
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_groups=norm_num_groups, dims=block_out_channels[0], eps=1e-6,
            pytorch_compatible=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)


    def __call__(
        self,
        sample: mx.array,
        latent_embeds: Optional[mx.array] = None,
    ):

        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample, latent_embeds)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: mx.array):
        self.mean, self.logvar = mx.split(parameters, 2, axis=-1)
        self.logvar = mx.clip(self.logvar, -30.0, 20.0)
        self.std = mx.exp(0.5 * self.logvar)
        self.var = mx.exp(self.logvar)

    def sample(self):
        sample = mx.random.normal(shape=self.mean.shape)
        sample = sample.astype(self.mean.dtype)
        x = self.mean + self.std * sample
        return x
