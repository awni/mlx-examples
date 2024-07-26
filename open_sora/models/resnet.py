from functools import partial
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .utils import get_activation

def upsample_nearest(x: mx.array, scale_factor: int):
    dims = x.ndim - 2
    shape = list(x.shape)
    for d in range(dims):
        shape.insert(2 + 2 * d, 1)
    x = x.reshape(shape)
    for d in range(dims):
        shape[2 + 2 * d] = scale_factor
    x = mx.broadcast_to(x, shape)
    for d in range(dims):
        shape[d + 1] *= shape[d + 2]
        shape.pop(d + 2)
    x = x.reshape(shape)
    return x


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        padding=1,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.interpolate = interpolate

        kernel_size = kernel_size or 3
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def __call__(self, hidden_states: mx.array):
        if self.interpolate:
            hidden_states = upsample_nearest(hidden_states, scale_factor=2)

        return self.conv(hidden_states)


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.padding = padding
        conv = nn.Conv2d(
            self.channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
        )

        self.conv = conv

    def __call__(self, hidden_states: mx.array):
        if self.padding == 0:
            hidden_states = mx.pad(
                hidden_states,
                ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.conv(hidden_states)


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" for a
            stronger conditioning with scale and shift.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        self.norm1 = nn.GroupNorm(
                num_groups=groups, dims=in_channels, eps=eps, affine=True,
                pytorch_compatible=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(
                num_groups=groups, dims=out_channels, eps=eps, affine=True,
                pytorch_compatible=True)

        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def __call__(self, input_tensor: mx.array, temb: mx.array):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states
        return output_tensor
