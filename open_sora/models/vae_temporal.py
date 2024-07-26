from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .utils import get_activation
from .vae import DiagonalGaussianDistribution


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        strides=None,  # allow custom stride
        **kwargs,
    ):
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else ((kernel_size,) * 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)

        time_pad = (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = [
            (0,0),
            (self.time_pad, 0),
            (height_pad, height_pad),
            (width_pad, width_pad),
            (0, 0),
        ]

        stride = strides if strides is not None else (stride, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, **kwargs)

    def __call__(self, x):
        x = mx.pad(x, self.time_causal_padding)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # SCH: added
        filters,
        conv_fn,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        num_groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activation_fn = activation_fn
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups, in_channels, pytorch_compatible=True)
        self.conv1 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), bias=False)
        self.norm2 = nn.GroupNorm(num_groups, self.filters, pytorch_compatible=True)
        self.conv2 = conv_fn(self.filters, self.filters, kernel_size=(3, 3, 3), bias=False)
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), bias=False)
            else:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(1, 1, 1), bias=False)

    def __call__(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:  # SCH: ResBlock X->Y
            residual = self.conv3(residual)
        return x + residual


class Encoder(nn.Module):
    """Encoder Blocks."""

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,  # num channels for latent vector
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activation_fn = get_activation(activation_fn)
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        # first layer conv
        self.conv_in = self.conv_fn(
            in_out_channels,
            filters,
            kernel_size=(3, 3, 3),
            bias=False,
        )

        # ResBlocks and conv downsample
        self.block_res_blocks = []
        self.conv_blocks = []

        filters = self.filters
        prev_filters = filters  # record for in_channels
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            block_items = []
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.temporal_downsample[i]:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 1
                    self.conv_blocks.append(
                        self.conv_fn(
                            prev_filters, filters, kernel_size=(3, 3, 3), strides=(t_stride, s_stride, s_stride)
                        )
                    )
                    prev_filters = filters  # update in_channels
                else:
                    # if no t downsample, don't add since this does nothing for pipeline models
                    self.conv_blocks.append(nn.Identity(prev_filters))  # Identity
                    prev_filters = filters  # update in_channels

        # last layer res block
        self.res_blocks = []
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(prev_filters, filters, **self.block_args))
            prev_filters = filters  # update in_channels

        self.norm1 = nn.GroupNorm(
            self.num_groups, dims=prev_filters, pytorch_compatible=True)

        self.conv2 = self.conv_fn(prev_filters, self.embedding_dim, kernel_size=(1, 1, 1))

    def __call__(self, x):
        x = self.conv_in(x)

        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)

        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Decoder Blocks."""

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.s_stride = 1

        self.activation_fn = get_activation(activation_fn)
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        filters = self.filters * self.channel_multipliers[-1]
        prev_filters = filters

        # last conv
        self.conv1 = self.conv_fn(self.embedding_dim, filters, kernel_size=(3, 3, 3), bias=True)

        # last layer res block
        self.res_blocks = []
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters, filters, **self.block_args))

        # ResBlocks and conv upsample
        self.block_res_blocks = []
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = []
        # reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = []
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                if self.temporal_downsample[i - 1]:
                    t_stride = 2 if self.temporal_downsample[i - 1] else 1
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters, prev_filters * t_stride * self.s_stride * self.s_stride, kernel_size=(3, 3, 3)
                        ),
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        nn.Identity(prev_filters),
                    )

        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters)

        self.conv_out = self.conv_fn(filters, in_out_channels, 3)

    def __call__(self, x):
        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x)
                x = rearrange(
                    x,
                    "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                    ts=t_stride,
                    hs=self.s_stride,
                    ws=self.s_stride,
                )

        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv_out(x)
        return x

class VAETemporal(nn.Module):
    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(True, True, False),
        num_groups=32,
        activation_fn="swish",
    ):
        super().__init__()

        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        self.patch_size = (self.time_downsample_factor, 1, 1)
        self.out_channels = in_out_channels

        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )
        self.quant_conv = CausalConv3d(2 * latent_embed_dim, 2 * embed_dim, 1)

        self.post_quant_conv = CausalConv3d(embed_dim, latent_embed_dim, 1)
        self.decoder = Decoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                lsize = None
            elif i == 0:
                time_padding = (
                    0
                    if (input_size[i] % self.time_downsample_factor == 0)
                    else self.time_downsample_factor - input_size[i] % self.time_downsample_factor
                )
                lsize = (input_size[i] + time_padding) // self.patch_size[i]
            else:
                lsize = input_size[i] // self.patch_size[i]
            latent_size.append(lsize)
        return latent_size

    def encode(self, x):
        time_padding = (
            0
            if (x.shape[1] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[1] % self.time_downsample_factor
        )

        pad = [(0, 0)] * 5
        pad[1] = (time_padding, 0)
        x = mx.pad(x, pad)

        encoded_feature = self.encoder(x)
        moments = self.quant_conv(encoded_feature)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, num_frames=None):
        time_padding = (
            0
            if (num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - num_frames % self.time_downsample_factor
        )
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        x = x[:, :, time_padding:]
        return x
