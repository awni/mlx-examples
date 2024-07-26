from typing import Callable, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        bias: bool = False,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        residual_connection: bool = False,
        out_dim: int = None,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.residual_connection = residual_connection
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale = dim_head**-0.5

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.group_norm = nn.GroupNorm(
            num_groups=norm_num_groups, dims=query_dim, eps=eps, affine=True,
            pytorch_compatible=True)

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)

        self.to_out = [nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        temb: Optional[mx.array] = None,
    ):
        r"""
        The forward method of the `Attention` class.
        """
        residual = hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.flatten(1, 2)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states)

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        B, L_q, D = query.shape
        L_kv = key.shape[1]
        query = query.reshape(B, L_q, self.heads, -1).transpose(0, 2, 1, 3)
        key = key.reshape(B, L_kv, self.heads, -1).transpose(0, 2, 1, 3)
        value = value.reshape(B, L_kv, self.heads, -1).transpose(0, 2, 1, 3)

        hidden_states = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=None
        )

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        if residual.ndim == 4:
            hidden_states = hidden_states.reshape(residual.shape)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states
