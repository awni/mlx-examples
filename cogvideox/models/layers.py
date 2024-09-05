import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def __call__(
        self,
        x: mx.array,
        timestep: Optional[mx.array] = None,
        temb: Optional[mx.array] = None,
    ) -> mx.array:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = mx.split(temb, 2, axis=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = mx.split(temb, 2, axis=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class CogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=elementwise_affine)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array, temb: mx.array
    ) -> Tuple[mx.array, mx.array]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(
            self.silu(temb)
        ).split(6, axis=1)
        hidden_states = (
            self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        )
        encoder_hidden_states = (
            self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :]
            + enc_shift[:, None, :]
        )
        return (
            hidden_states,
            encoder_hidden_states,
            gate[:, None, :],
            enc_gate[:, None, :],
        )


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray,
    theta: float = 10000.0,
    linear_factor=1.0,
    ntk_factor=1.0,
):
    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (mx.arange(0, dim, 2)[: (dim // 2)] / dim)) / linear_factor
    )  # [D/2]
    freqs = pos[:, None] * freqs[None]
    freqs_cos = mx.repeat(mx.cos(freqs), 2, axis=1)
    freqs_sin = mx.repeat(mx.sin(freqs), 2, axis=1)
    return freqs_cos, freqs_sin


def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.

    Returns:
        `mx.array`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    start, stop = crops_coords
    grid_size_h, grid_size_w = grid_size
    grid_h = np.linspace(
        start[0], stop[0], grid_size_h, endpoint=False, dtype=np.float32
    )
    grid_w = np.linspace(
        start[1], stop[1], grid_size_w, endpoint=False, dtype=np.float32
    )
    grid_t = np.linspace(
        0, temporal_size, temporal_size, endpoint=False, dtype=np.float32
    )

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w)

    # Broadcast and concatenate temporal and spaial frequencies (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        shape = (temporal_size, grid_size_h, grid_size_w)
        freqs_t = mx.broadcast_to(freqs_t[:, None, None, :], shape + (dim_t,))
        freqs_h = mx.broadcast_to(freqs_h[None, :, None, :], shape + (dim_h,))
        freqs_w = mx.broadcast_to(freqs_w[None, None, :, :], shape + (dim_w,))
        freqs = mx.concatenate([freqs_t, freqs_h, freqs_w], axis=-1)
        freqs = freqs.reshape(temporal_size * grid_size_h * grid_size_w, -1)
        return freqs

    t_cos, t_sin = freqs_t
    h_cos, h_sin = freqs_h
    w_cos, w_sin = freqs_w
    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin


def apply_rotary_emb(
    x: mx.array,
    freqs_cis: Union[mx.array, Tuple[mx.array]],
) -> Tuple[mx.array, mx.array]:
    """
    Args:
        x (`mx.array`): array to apply rotary embeddings. [B, H, S, D]
        freqs_cis (`Tuple[mx.array]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[mx.array, mx.array]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos, sin = freqs_cis  # [S, D]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).split(
        2, axis=-1
    )  # [B, S, H, D//2]
    x_rotated = mx.stack([-x_imag, x_real], axis=-1).flatten(3)
    return x * cos + x_rotated * sin


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_norm: bool = False,
        eps: float = 1e-5,
        attention_bias: bool = False,
        out_bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=attention_bias)
        self.to_k = nn.Linear(dim, dim, bias=attention_bias)
        self.to_v = nn.Linear(dim, dim, bias=attention_bias)
        self.norm_q = nn.LayerNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_k = nn.LayerNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.to_out = nn.Linear(dim, dim, bias=out_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        freqs: mx.array = None,
    ):
        x = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        text_seq_length = encoder_hidden_states.shape[1]

        B, N, C = x.shape
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = q.reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        q = self.norm_q(q)
        k = self.norm_k(k)
        if freqs is not None:
            q[:, :, text_seq_length:] = apply_rotary_emb(
                q[:, :, text_seq_length:], freqs
            )
            k[:, :, text_seq_length:] = apply_rotary_emb(
                k[:, :, text_seq_length:], freqs
            )

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        x = x.swapaxes(1, 2).reshape(B, N, C)
        x = self.to_out(x)
        encoder_hidden_states, hidden_states = x.split([text_seq_length], axis=1)
        return hidden_states, encoder_hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.gelu = nn.GELU(approx=approximate)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return self.gelu(hidden_states)


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "gelu",
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        else:
            raise ValueError(f"{activation_fn} is not yet implemented.")

        self.net = [act_fn, nn.Linear(inner_dim, dim_out, bias=bias)]

    def __call__(self, hidden_states: mx.array):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (mx.array):
            a 1-D array of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        mx.array: an [N x dim] array of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(
        start=0,
        stop=half_dim,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = timesteps[:, None] * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        pad = [(0, 0)] * emb.ndim
        pad[-2] = (0, 1)
        emb = mx.pad(emb, pad)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> np.ndarray:
    r"""
    Args:
        embed_dim (`int`):
        spatial_size (`int` or `Tuple[int, int]`):
        temporal_size (`int`):
        spatial_interpolation_scale (`float`, defaults to 1.0):
        temporal_interpolation_scale (`float`, defaults to 1.0):
    """
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # 1. Spatial
    grid_h = np.arange(spatial_size[1], dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(spatial_size[0], dtype=np.float32) / spatial_interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # 2. Temporal
    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # 3. Concat
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, temporal_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, spatial_size[0] * spatial_size[1], axis=1
    )  # [T, H*W, D // 4]

    pos_embed = np.concatenate(
        [pos_embed_temporal, pos_embed_spatial], axis=-1
    )  # [T, H*W, D]
    return pos_embed


class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

    def __call__(self, text_embeds: mx.array, image_embeds: mx.array):
        r"""
        Args:
            text_embeds (`mx.array`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`mx.array`):
                Input image embeddings. Expected shape: (batch_size, num_frames, height, width, channels).
        """
        text_embeds = self.text_proj(text_embeds)

        batch, num_frames, height, width, channels = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, height, width, channels)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.reshape(batch, -1, image_embeds.shape[-1])
        out = mx.concatenate([text_embeds, image_embeds], axis=1)
        return out


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def __call__(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb
