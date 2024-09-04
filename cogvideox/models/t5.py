from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    """
    Adapted from HF Tensorflow:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

    Translate relative position to a bucket number for relative attention. The
    relative position is defined as memory_position - query_position, i.e. the
    distance in tokens from the attending position to the attended-to position.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions
    <=-max_distance map to the same bucket.  This should allow for more
    graceful generalization to longer sequences than the model has been trained
    on

    Args:
        relative_position: an int32 array
        num_buckets: an integer
        max_distance: an integer
    """
    relative_buckets = 0
    num_buckets //= 2
    relative_buckets += (relative_position > 0).astype(mx.int16) * num_buckets
    relative_position = mx.abs(relative_position)

    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in
    # positions up to max_distance
    scale = (num_buckets - max_exact) / np.log(max_distance / max_exact)
    relative_position_if_large = max_exact + (
        mx.log(relative_position.astype(mx.float32) / max_exact) * scale
    ).astype(mx.int16)
    relative_position_if_large = mx.minimum(relative_position_if_large, num_buckets - 1)
    relative_buckets += mx.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, num_buckets, max_distance):
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = num_heads
        self.embeddings = nn.Embedding(num_buckets, num_heads)

    def __call__(self, query_length: int, key_length: int, offset: int = 0):
        """Compute binned relative position bias"""
        context_position = mx.arange(offset, query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]

        # shape (query_length, key_length)
        relative_position = memory_position - context_position
        relative_position_bucket = _relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        # shape (query_length, key_length, num_heads)
        values = self.embeddings(relative_position_bucket)

        # shape (num_heads, query_length, key_length)
        return values.transpose(2, 0, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_kv, num_heads):
        super().__init__()
        inner_dim = d_kv * num_heads
        self.num_heads = num_heads
        self.q = nn.Linear(d_model, inner_dim, bias=False)
        self.k = nn.Linear(d_model, inner_dim, bias=False)
        self.v = nn.Linear(d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, d_model, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array],
    ) -> mx.array:
        queries = self.q(inputs)
        keys = self.k(inputs)
        values = self.v(inputs)

        num_heads = self.num_heads
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scores = queries @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)

        scores = mx.softmax(scores, axis=-1, precise=True)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o(values_hat)


class DenseActivation(nn.Module):
    def __init__(self, d_model, d_ff, feed_forward_proj):
        super().__init__()
        mlp_dims = d_ff or d_model * 4
        self.gated = feed_forward_proj.startswith("gated")
        if self.gated:
            self.wi_0 = nn.Linear(d_model, mlp_dims, bias=False)
            self.wi_1 = nn.Linear(d_model, mlp_dims, bias=False)
        else:
            self.wi = nn.Linear(d_model, mlp_dims, bias=False)
        self.wo = nn.Linear(mlp_dims, d_model, bias=False)
        activation = feed_forward_proj.removeprefix("gated-")
        if activation == "relu":
            self.act = nn.relu
        elif activation == "gelu":
            self.act = nn.gelu
        elif activation == "silu":
            self.act = nn.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x):
        if self.gated:
            hidden_act = self.act(self.wi_0(x))
            hidden_linear = self.wi_1(x)
            x = hidden_act * hidden_linear
        else:
            x = self.act(self.wi(x))
        return self.wo(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        feed_forward_proj,
        d_kv,
        num_heads,
        layer_norm_epsilon,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, d_kv, num_heads)
        self.ln1 = nn.RMSNorm(d_model, eps=layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(d_model, eps=layer_norm_epsilon)
        self.dense = DenseActivation(d_model, d_ff, feed_forward_proj)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y = self.attention(y, mask=mask)
        x = x + y
        y = self.ln2(x)
        y = self.dense(y)
        return x + y


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        d_ff,
        feed_forward_proj,
        d_kv,
        num_heads,
        num_layers,
        layer_norm_epsilon,
        relative_attention_num_buckets,
        relative_attention_max_distance,
        **kwargs,
    ):
        super().__init__()
        self.block = [
            TransformerEncoderLayer(
                d_model,
                d_ff,
                feed_forward_proj,
                d_kv,
                num_heads,
                layer_norm_epsilon,
            )
            for i in range(num_layers)
        ]
        self.final_layer_norm = nn.RMSNorm(d_model, eps=layer_norm_epsilon)
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.relative_attention_bias = RelativePositionBias(
            num_heads,
            relative_attention_num_buckets,
            relative_attention_max_distance,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        x = self.embed_tokens(x)
        pos_bias = self.relative_attention_bias(x.shape[1], x.shape[1])
        if mask is not None:
            mask = pos_bias + mx.log(mask)
        else:
            mask = pos_bias
        for e, block in enumerate(self.block):
            x = block(x, mask=mask)
        return self.final_layer_norm(x)


class T5(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        **kwargs,
    ):
        self.encoder = TransformerEncoder(
            vocab_size,
            d_model,
            **kwargs,
        )

    def encode(self, inputs: mx.array, mask: Optional[mx.array] = None):
        return self.encoder(inputs, mask=mask)
