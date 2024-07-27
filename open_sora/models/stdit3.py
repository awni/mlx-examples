# Copyright © 2024 Apple Inc.

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .blocks import (
    Attention,
    CaptionEmbedder,
    Mlp,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    RoPE,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    t2i_modulate,
)


class STDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        rope=None,
        qk_norm=False,
        temporal=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=nn.GELU(approx="precise"),
        )
        self.scale_shift_table = (
            mx.random.normal(shape=(6, hidden_size)) / hidden_size**0.5
        )

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

    def __call__(
        self,
        x,
        y,
        t,
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).split(6, axis=1)
        if x_mask is not None:
            (
                shift_msa_zero,
                scale_msa_zero,
                gate_msa_zero,
                shift_mlp_zero,
                scale_mlp_zero,
                gate_mlp_zero,
            ) = (self.scale_shift_table[None] + t0.reshape(B, 6, -1)).split(6, axis=1)
        # modulate (attention)
        x_normed = self.norm1(x)
        x_m = t2i_modulate(x_normed, shift_msa, scale_msa)

        if x_mask is not None:
            x_m_zero = t2i_modulate(x_normed, shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # attention
        if self.temporal:
            x_m = x_m.reshape(B, T, S, C).swapaxes(1, 2).flatten(0, 1)
            x_m = self.attn(x_m)
            x_m = x_m.reshape(B, S, T, C).swapaxes(1, 2).flatten(1, 2)
        else:
            x_m = x_m.reshape(B * T, S, C)
            x_m = self.attn(x_m)
            x_m = x_m.reshape(B, T * S, C)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + x_m_s

        # cross attention
        x = x + self.cross_attn(x, y)

        # modulate (MLP)
        x_normed = self.norm2(x)
        x_m = t2i_modulate(x_normed, shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(x_normed, shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + x_m_s

        return x


class STDiT3(nn.Module):

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=512,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        pred_sigma=True,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        skip_y_embedder=False,
        **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.skip_y_embedder = skip_y_embedder

        # model size related
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # computation related

        # input size related
        self.patch_size = patch_size
        self.input_sq_size = input_sq_size
        self.pos_embed = PositionEmbedding2D(hidden_size)
        self.rope = RoPE(dims=self.hidden_size // self.num_heads)

        # embedding
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = [
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        ]
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            act_layer=nn.GELU(approx="precise"),
            token_num=model_max_length,
        )

        # spatial blocks
        self.spatial_blocks = [
            STDiT3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ]

        # temporal blocks
        self.temporal_blocks = [
            STDiT3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                # temporal
                temporal=True,
                rope=self.rope,
            )
            for i in range(depth)
        ]

        # final layer
        self.final_layer = T2IFinalLayer(
            hidden_size, np.prod(self.patch_size), self.out_channels
        )

    def get_dynamic_size(self, x):
        _, T, H, W, _ = x.shape
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def __call__(
        self, x, timestep, y, x_mask=None, fps=None, height=None, width=None, **kwargs
    ):
        dtype = self.x_embedder.proj.weight.dtype
        B = x.shape[0]
        x = x.astype(dtype)
        timestep = timestep.astype(dtype)
        y = y.astype(dtype)

        # === get pos embed ===
        _, Tx, Hx, Wx, _ = x.shape
        T, H, W = self.get_dynamic_size(x)

        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height * width) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(H, W, scale=scale, base_size=base_size)
        pos_emb = pos_emb.astype(dtype)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=dtype)  # [B, C]

        fps = self.fps_embedder(mx.array([fps], dtype), B)
        t = t + fps
        t_mlp = self.t_block[0](t)
        t_mlp = self.t_block[1](t_mlp)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = mx.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block[0](t0)
            t0_mlp = self.t_block[1](t0_mlp)

        # === get y embed ===
        if not self.skip_y_embedder:
            y = self.y_embedder(y)  # [B, N_token, C]

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = x.reshape(B, T, S, -1)
        x = x + pos_emb

        x = mx.flatten(x, 1, 2)

        # === blocks ===
        for spatial_block, temporal_block in zip(
            self.spatial_blocks, self.temporal_blocks
        ):
            x = spatial_block(x, y, t_mlp, x_mask, t0_mlp, T, S)
            x = temporal_block(x, y, t_mlp, x_mask, t0_mlp, T, S)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.astype(mx.float32)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (mx.array): of shape [B, N, C]

        Return:
            x (mx.array): of shape [B, T, H, W, C_out]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        B = x.shape[0]
        T_p, H_p, W_p = self.patch_size
        x = x.reshape(B, N_t, N_h, N_w, T_p, H_p, W_p, self.out_channels)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        x = x.reshape(B, N_t * T_p, N_h * H_p, N_w * W_p, self.out_channels)

        # unpad
        x = x[:, :R_t, :R_h, :R_w]
        return x
