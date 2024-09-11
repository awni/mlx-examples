from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .layers import get_activation


def interpolate(x: mx.array, scale_factors: Tuple[int]):
    dims = x.ndim - 2
    shape = list(x.shape)
    for d in range(dims):
        shape.insert(2 + 2 * d, 1)
    x = x.reshape(shape)
    for d in range(dims):
        shape[2 + 2 * d] = scale_factors[d]
    x = mx.broadcast_to(x, shape)
    for d in range(dims):
        shape[d + 1] *= shape[d + 2]
        shape.pop(d + 2)
    x = x.reshape(shape)
    return x


class CogVideoXDownsample3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.compress_time = compress_time
        if self.compress_time:
            self.pool = nn.AvgPool1d(2, 2)

    def __call__(self, x):
        if self.compress_time:
            batch_size, frames, height, width, channels = x.shape

            x = x.moveaxis(1, 3).reshape(batch_size * height * width, frames, channels)

            if x.shape[-2] % 2 == 1:
                x_first, x_rest = x[..., 0, :], x[..., 1:, :]
                if x_rest.shape[-2] > 0:
                    x_rest = self.pool(x_rest)

                x = mx.concatenate([x_first[..., None, :], x_rest], axis=-2)
                x = x.reshape(
                    batch_size, height, width, x.shape[-2], channels
                ).moveaxis(3, 1)
            else:
                x = self.pool(x)
                x = x.reshape(
                    batch_size, height, width, x.shape[-2], channels
                ).moveaxis(3, 1)

        # Pad the tensor
        batch_size, frames, height, width, channels = x.shape
        x = x.reshape(batch_size * frames, height, width, channels)
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.conv(x)
        return x.reshape(batch_size, frames, *x.shape[1:])


class CogVideoXUpsample3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.compress_time = compress_time

    def __call__(self, inputs):
        if self.compress_time:
            if inputs.shape[1] > 1 and inputs.shape[1] % 2 == 1:
                # split first frame
                x_first, x_rest = inputs[:, 0], inputs[:, 1:]
                x_first = interpolate(x_first, scale_factors=(2, 2))
                x_rest = interpolate(x_rest, scale_factors=(2, 2, 2))
                inputs = mx.concatenate([x_first[:, None], x_rest], axis=1)
            elif inputs.shape[1] > 1:
                inputs = interpolate(inputs, scale_factors=(2, 2, 2))
            else:
                inputs = inputs.squeeze(1)
                inputs = interpolate(inputs, scale_factors=(2, 2))
                inputs = inputs[:, None]
        else:
            b, t, h, w, c = inputs.shape
            inputs = inputs.reshape(b * t, h, w, c)
            inputs = interpolate(inputs, scale_factors=(2, 2))
            inputs = inputs.reshape(b, t, *inputs.shape[1:])

        b, t, h, w, c = inputs.shape
        inputs = inputs.reshape(b * t, h, w, c)
        inputs = self.conv(inputs)
        return inputs.reshape(b, t, *inputs.shape[1:])


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


class CogVideoXCausalConv3d(nn.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model.

    Args:
        in_channels (`int`): Number of channels in the input tensor.
        out_channels (`int`): Number of output channels produced by the convolution.
        kernel_size (`int` or `Tuple[int, int, int]`): Kernel size of the convolutional kernel.
        stride (`int`, defaults to `1`): Stride of the convolution.
        dilation (`int`, defaults to `1`): Dilation rate of the convolution.
        pad_mode (`str`, defaults to `"constant"`): Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = (stride, 1, 1)
        assert dilation == 1, "Dilation > 1 NYI"
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.conv_cache = None

    def fake_context_parallel_forward(self, inputs: mx.array) -> mx.array:
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = (
                [self.conv_cache]
                if self.conv_cache is not None
                else [inputs[:, :1]] * (kernel_size - 1)
            )
            inputs = mx.concatenate(cached_inputs + [inputs], axis=1)
        return inputs

    def _clear_fake_context_parallel_cache(self):
        self.conv_cache = None

    def __call__(self, inputs: mx.array) -> mx.array:
        inputs = self.fake_context_parallel_forward(inputs)

        self._clear_fake_context_parallel_cache()
        self.conv_cache = inputs[:, -self.time_kernel_size + 1 :]

        pad = [(0, 0)] * inputs.ndim
        pad[-3] = (self.width_pad, self.width_pad)
        pad[-2] = (self.height_pad, self.height_pad)
        inputs = mx.pad(inputs, pad)
        output = self.conv(inputs)
        return output


class CogVideoXSpatialNorm3D(nn.Module):
    r"""
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002. This implementation is specific
    to 3D-video like data.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
        groups (`int`):
            Number of groups to separate the channels into for group normalization.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
    ):
        super().__init__()

        self.norm_layer = nn.GroupNorm(
            num_groups=groups,
            dims=f_channels,
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv_y = CogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1
        )
        self.conv_b = CogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1
        )

    def __call__(self, f: mx.array, zq: mx.array) -> mx.array:
        scale_factors = list(f.shape[i] // zq.shape[i] for i in range(1, 4))
        if f.shape[1] > 1 and f.shape[1] % 2 == 1:
            scale_factors[0] = (f.shape[1] - 1) // (zq.shape[1] - 1)
            z_first, z_rest = zq[:, 0], zq[:, 1:]
            z_first = interpolate(z_first, scale_factors=scale_factors[1:])
            z_rest = interpolate(z_rest, scale_factors=scale_factors)
            zq = mx.concatenate([z_first[:, None], z_rest], axis=1)
        else:
            zq = interpolate(zq, scale_factors=scale_factors)

        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class CogVideoXResnetBlock3D(nn.Module):
    r"""
    A 3D ResNet block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        non_linearity (`str`, defaults to `"swish"`):
            Activation function to use.
        conv_shortcut (bool, defaults to `False`):
            Whether or not to use a convolution shortcut.
        spatial_norm_dim (`int`, *optional*):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut

        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(
                num_groups=groups,
                dims=in_channels,
                eps=eps,
                affine=True,
                pytorch_compatible=True,
            )
            self.norm2 = nn.GroupNorm(
                num_groups=groups,
                dims=out_channels,
                eps=eps,
                affine=True,
                pytorch_compatible=True,
            )
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )

        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(
                in_features=temb_channels, out_features=out_channels
            )

        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    pad_mode=pad_mode,
                )
            else:
                self.conv_shortcut = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def __call__(
        self,
        inputs: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = inputs

        if zq is not None:
            hidden_states = self.norm1(hidden_states, zq)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = (
                hidden_states
                + self.temb_proj(self.nonlinearity(temb))[:, None, None, None, :]
            )

        if zq is not None:
            hidden_states = self.norm2(hidden_states, zq)
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class CogVideoXDownBlock3D(nn.Module):
    r"""
    A downsampling block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        add_downsample (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = resnets
        self.downsamplers = None

        if add_downsample:
            self.downsamplers = [
                CogVideoXDownsample3D(
                    out_channels,
                    out_channels,
                    padding=downsample_padding,
                    compress_time=compress_time,
                )
            ]

    def __call__(
        self,
        hidden_states: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
    ) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, zq)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class CogVideoXMidBlock3D(nn.Module):
    r"""
    A middle block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, *optional*):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )
        self.resnets = resnets

    def __call__(
        self,
        hidden_states: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
    ) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, zq)

        return hidden_states


class CogVideoXUpBlock3D(nn.Module):
    r"""
    An upsampling block used in the CogVideoX model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`):
            Number of time embedding channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        resnet_groups (`int`, defaults to `32`):
            Number of groups to separate the channels into for group normalization.
        spatial_norm_dim (`int`, defaults to `16`):
            The dimension to use for spatial norm if it is to be used instead of group norm.
        add_upsample (`bool`, defaults to `True`):
            Whether or not to use a upsampling layer. If not used, output dimension would be same as input dimension.
        compress_time (`bool`, defaults to `False`):
            Whether or not to downsample across temporal dimension.
        pad_mode (str, defaults to `"first"`):
            Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = resnets
        self.upsamplers = None

        if add_upsample:
            self.upsamplers = [
                CogVideoXUpsample3D(
                    out_channels,
                    out_channels,
                    padding=upsample_padding,
                    compress_time=compress_time,
                )
            ]

    def __call__(
        self,
        hidden_states: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
    ) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, zq)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class CogVideoXEncoder3D(nn.Module):
    r"""
    The `CogVideoXEncoder3D` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CogVideoXCausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode
        )
        self.down_blocks = []

        # down blocks
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if down_block_type == "CogVideoXDownBlock3D":
                down_block = CogVideoXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    compress_time=compress_time,
                )
            else:
                raise ValueError(
                    "Invalid `down_block_type` encountered. Must be `CogVideoXDownBlock3D`"
                )

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
        )

        self.norm_out = nn.GroupNorm(
            num_groups=norm_num_groups,
            dims=block_out_channels[-1],
            eps=1e-6,
            affine=True,
            pytorch_compatible=True,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode
        )

    def __call__(self, sample: mx.array, temb: Optional[mx.array] = None) -> mx.array:
        hidden_states = self.conv_in(sample)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states, temb, None)

        hidden_states = self.mid_block(hidden_states, temb, None)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class CogVideoXDecoder3D(nn.Module):
    r"""
    The `CogVideoXDecoder3D` layer of a variational autoencoder that decodes its latent representation into an output
    sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = CogVideoXCausalConv3d(
            in_channels,
            reversed_block_out_channels[0],
            kernel_size=3,
            pad_mode=pad_mode,
        )

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
        )

        # up blocks
        self.up_blocks = []

        output_channel = reversed_block_out_channels[0]
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if up_block_type == "CogVideoXUpBlock3D":
                up_block = CogVideoXUpBlock3D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=in_channels,
                    add_upsample=not is_final_block,
                    compress_time=compress_time,
                    pad_mode=pad_mode,
                )
                prev_output_channel = output_channel
            else:
                raise ValueError(
                    "Invalid `up_block_type` encountered. Must be `CogVideoXUpBlock3D`"
                )

            self.up_blocks.append(up_block)

        self.norm_out = CogVideoXSpatialNorm3D(
            reversed_block_out_channels[-1], in_channels, groups=norm_num_groups
        )
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            reversed_block_out_channels[-1],
            out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

    def __call__(self, sample: mx.array, temb: Optional[mx.array] = None) -> mx.array:
        hidden_states = self.conv_in(sample)

        hidden_states = self.mid_block(hidden_states, temb, sample)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, temb, sample)

        hidden_states = self.norm_out(hidden_states, sample)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderKL(nn.Module):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images. Used in
    [CogVideoX](https://github.com/THUDM/CogVideo).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to `1.15258426`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_height: int = 480,
        sample_width: int = 720,
        scaling_factor: float = 1.15258426,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.encoder = CogVideoXEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.decoder = CogVideoXDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.quant_conv = (
            nn.Conv3d(2 * out_channels, 2 * out_channels, 1) if use_quant_conv else None
        )
        self.post_quant_conv = (
            nn.Conv3d(out_channels, out_channels, 1) if use_post_quant_conv else None
        )

        self.use_slicing = False
        self.use_tiling = False

        self.num_latent_frames_batch_size = 2

        # We make the minimum height and width of sample for tiling half that of the generally supported
        self.block_out_channels = block_out_channels
        self.tile_sample_min_height = sample_height // 2
        self.tile_sample_min_width = sample_width // 2
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(
            self.tile_sample_min_width / (2 ** (len(block_out_channels) - 1))
        )
        self.tile_overlap_factor_height = 1 / 6
        self.tile_overlap_factor_width = 1 / 5
        self.scaling_factor = scaling_factor
        self.temporal_compression_ratio = temporal_compression_ratio

    def _clear_fake_context_parallel_cache(self):
        for name, module in self.named_modules():
            if isinstance(module, CogVideoXCausalConv3d):
                module._clear_fake_context_parallel_cache()

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_overlap_factor_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension. Must be between 0 and 1. Setting a higher
                value might cause more tiles to be processed leading to slow down of the decoding process.
            tile_overlap_factor_width (`int`, *optional*):
                The minimum amount of overlap between two consecutive horizontal tiles. This is to ensure that there
                are no tiling artifacts produced across the width dimension. Must be between 0 and 1. Setting a higher
                value might cause more tiles to be processed leading to slow down of the decoding process.
        """
        self.use_tiling = True
        self.tile_sample_min_height = (
            tile_sample_min_height or self.tile_sample_min_height
        )
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(self.block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(
            self.tile_sample_min_width / (2 ** (len(self.block_out_channels) - 1))
        )
        self.tile_overlap_factor_height = (
            tile_overlap_factor_height or self.tile_overlap_factor_height
        )
        self.tile_overlap_factor_width = (
            tile_overlap_factor_width or self.tile_overlap_factor_width
        )

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def encode(self, x: mx.array) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        return DiagonalGaussianDistribution(h)

    def _decode(self, z: mx.array) -> mx.array:
        batch_size, num_frames, height, width, num_channels = z.shape

        if self.use_tiling and (
            width > self.tile_latent_min_width or height > self.tile_latent_min_height
        ):
            return self.tiled_decode(z)

        frame_batch_size = self.num_latent_frames_batch_size
        dec = []
        for i in range(num_frames // frame_batch_size):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = z[:, start_frame:end_frame]
            if self.post_quant_conv is not None:
                z_intermediate = self.post_quant_conv(z_intermediate)
            z_intermediate = self.decoder(z_intermediate)
            dec.append(z_intermediate)

        self._clear_fake_context_parallel_cache()
        return mx.concatenate(dec, axis=1)

    def decode(self, z: mx.array) -> mx.array:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = mx.concatenate(decoded_slices, axis=0)
        else:
            decoded = self._decode(z)

        return decoded

    def blend_v(self, a: mx.array, b: mx.array, blend_extent: int) -> mx.array:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y] = a[:, :, -blend_extent + y] * (1 - y / blend_extent) + b[
                :, :, y
            ] * (y / blend_extent)
        return b

    def blend_h(self, a: mx.array, b: mx.array, blend_extent: int) -> mx.array:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                :, :, :, x
            ] * (x / blend_extent)
        return b

    def tiled_decode(self, z: mx.array) -> mx.array:
        # Rough memory assessment:
        #   - In CogVideoX-2B, there are a total of 24 CausalConv3d layers.
        #   - The biggest intermediate dimensions are: [1, 128, 9, 480, 720].
        #   - Assume fp16 (2 bytes per value).
        # Memory required: 1 * 128 * 9 * 480 * 720 * 24 * 2 / 1024**3 = 17.8 GB
        #
        # Memory assessment when using tiling:
        #   - Assume everything as above but now HxW is 240x360 by tiling in half
        # Memory required: 1 * 128 * 9 * 240 * 360 * 24 * 2 / 1024**3 = 4.5 GB

        batch_size, num_frames, height, width, num_channels = z.shape

        overlap_height = int(
            self.tile_latent_min_height * (1 - self.tile_overlap_factor_height)
        )
        overlap_width = int(
            self.tile_latent_min_width * (1 - self.tile_overlap_factor_width)
        )
        blend_extent_height = int(
            self.tile_sample_min_height * self.tile_overlap_factor_height
        )
        blend_extent_width = int(
            self.tile_sample_min_width * self.tile_overlap_factor_width
        )
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        frame_batch_size = self.num_latent_frames_batch_size

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                time = []
                for k in range(num_frames // frame_batch_size):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (
                        0 if k == 0 else remaining_frames
                    )
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = z[
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    if self.post_quant_conv is not None:
                        tile = self.post_quant_conv(tile)
                    tile = self.decoder(tile)
                    time.append(tile)
                self._clear_fake_context_parallel_cache()
                row.append(mx.concatenate(time, axis=1))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :row_limit_height, :row_limit_width])
            result_rows.append(mx.concatenate(result_row, axis=3))

        return mx.concatenate(result_rows, axis=2)
