# Copyright © 2024 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class RoPE(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.freqs = mx.zeros((dims // 2,))

    def __call__(self, x):
        N = x.shape[-2]
        positions = mx.arange(N)
        theta = positions[:, None] * self.freqs[None, :]
        cos = mx.cos(theta)
        sin = mx.sin(theta)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        lx = x1 * cos - x2 * sin
        rx = x1 * sin + x2 * cos
        return mx.concatenate([lx, rx], axis=-1)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x):
        _, D, H, W, _ = x.shape
        padding = [(0, 0)] * 5
        if D % self.patch_size[0] != 0:
            padding[1] = (0, self.patch_size[0] - D % self.patch_size[0])
        if H % self.patch_size[1] != 0:
            padding[2] = (0, self.patch_size[1] - H % self.patch_size[1])
        if W % self.patch_size[2] != 0:
            padding[3] = (0, self.patch_size[2] - W % self.patch_size[2])
        x = mx.pad(x, padding)
        x = self.proj(x)  # (B T H W C)
        return mx.flatten(x, 1, 3)  # BTHWC -> BNC


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

        self.rope = False
        if rope is not None:
            self.rope = True
            self._rotary_emb = rope

    def __call__(self, x: mx.array):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3 * self.num_heads, self.head_dim)

        qkv = qkv.reshape(qkv_shape).transpose(0, 2, 1, 3)
        q, k, v = qkv.split(3, axis=1)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            q = self._rotary_emb(q)
            k = self._rotary_emb(k)

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)

        x = x.swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / (self.head_dim**0.5)

    def __call__(self, x, cond, mask=None):
        B, N, C = x.shape

        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).reshape(
            B, cond.shape[1], 2 * self.num_heads, self.head_dim
        )
        k, v = kv.split(2, axis=2)
        q, k, v = map(lambda x: x.swapaxes(1, 2), (q, k, v))

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        x = x.swapaxes(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = (
            mx.random.normal(shape=(2, hidden_size)) / hidden_size**0.5
        )
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        B = x.shape[0]
        C = x.shape[-1]
        x = x.reshape(B, T, S, C)
        masked_x = masked_x.reshape(B, T, S, C)
        x = mx.where(x_mask[..., None, None], x, masked_x)
        x = x.flatten(1, 2)
        return x

    def __call__(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).split(2, axis=1)
        x_normed = self.norm_final(x)
        x = t2i_modulate(x_normed, shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).split(
                2, axis=1
            )
            x_zero = t2i_modulate(x_normed, shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: a 1-D array of N indices, one per batch element.
              These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) array of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half) / half)
        args = t[:, None] * freqs[None]
        to_cat = [mx.cos(args), mx.sin(args)]
        if dim % 2:
            to_cat.append(mx.zeros((embedding.shape[0], 1)))
        return mx.concatenate(to_cat, axis=-1)

    def __call__(self, t, dtype):
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = t_emb.astype(dtype)
        for l in self.mlp:
            t_emb = l(t_emb)
        return t_emb


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(
            hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size
        )
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def __call__(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        if s.shape[0] != bs:
            s = mx.tile(s, (bs // s.shape[0], 1))
        b, dims = s.shape[0], s.shape[1]
        s = s.flatten()
        s_emb = self.timestep_embedding(s, self.frequency_embedding_size)
        s_emb = s_emb.astype(self.mlp[0].weight.dtype)
        for l in self.mlp:
            s_emb = l(s_emb)
        return s_emb.reshape(b, dims * self.outdim)


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        act_layer=nn.GELU(approx="precise"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
        )
        self.y_embedding = mx.random.normal(shape=(token_num, in_channels))

    def __call__(self, caption):
        return self.y_proj(caption)


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        self._inv_freq = 1.0 / (10000 ** (mx.arange(0, half_dim, 2) / half_dim))

    def _get_sin_cos_emb(self, t: mx.array):
        out = t[:, None] * self._inv_freq[None]
        emb_cos = mx.cos(out)
        emb_sin = mx.sin(out)
        return mx.concatenate([emb_sin, emb_cos], axis=-1)

    def __call__(
        self,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ):
        grid_h = mx.arange(h) / scale
        grid_w = mx.arange(w) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = mx.meshgrid(
            grid_w,
            grid_h,
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.T.flatten()
        grid_w = grid_w.T.flatten()
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return mx.concatenate([emb_h, emb_w], axis=-1)[None, :]
