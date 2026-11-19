# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# This module 'transformer.py' is inspired by https://github.com/wenwenyu/MASTER-pytorch and Decoder is borrowed

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["Decoder", "PositionalEncoding", "EncoderBlock", "MultiHeadAttention", "PositionwiseFeedForward"]


class PositionalEncoding(nn.Module):
    """Compute positional encoding"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Forward pass

        Args:
            x: embeddings (batch, max_len, d_model)
            start: position of the first embedding in the sequence (e.g. for incremental decoding)

        Returns:
            positional embeddings (batch, max_len, d_model)
        """
        x = x + self.pe[:, start : start + x.size(1)]  # type: ignore[index]
        return self.dropout(x)


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, None]:
    """Scaled Dot-Product Attention, with the fused kernels of PyTorch (the attention weights are not materialized)

    Args:
        query: queries of shape (*, L, E)
        key: keys of shape (*, S, E)
        value: values of shape (*, S, E)
        mask: optional mask broadcastable to (*, L, S), where 0 marks the keys a query must not attend to

    Returns:
        the attention output of shape (*, L, E), and None in place of the attention weights
    """
    # NOTE: the boolean mask is derived with an int comparison, to ensure the ONNX compatibility
    attn_mask = None if mask is None else mask != 0
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask), None


class PositionwiseFeedForward(nn.Sequential):
    """Position-wise Feed-Forward Network"""

    def __init__(
        self, d_model: int, ffd: int, dropout: float = 0.1, activation_fct: Callable[[Any], Any] = nn.ReLU()
    ) -> None:
        super().__init__(  # type: ignore[call-overload]
            nn.Linear(d_model, ffd),
            activation_fct,
            nn.Dropout(p=dropout),
            nn.Linear(ffd, d_model),
            nn.Dropout(p=dropout),
        )


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (N, L, d_model) --> (N, num_heads, L, d_k)
        return x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

    def project_kv(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the keys and values, e.g. once for the steps of an incremental decoding

        Args:
            key: keys of shape (N, S, d_model)
            value: values of shape (N, S, d_model)

        Returns:
            the projected keys and values, of shape (N, num_heads, S, d_k)
        """
        return self._split_heads(self.linear_layers[1](key)), self._split_heads(self.linear_layers[2](value))

    def attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Attention of the queries over keys and values projected with `project_kv`

        Args:
            query: queries of shape (N, L, d_model)
            key: projected keys of shape (N, num_heads, S, d_k)
            value: projected values of shape (N, num_heads, S, d_k)
            mask: optional mask broadcastable to (N, num_heads, L, S), where 0 marks the masked keys

        Returns:
            the attention output of shape (N, L, d_model)
        """
        x, _ = scaled_dot_product_attention(self._split_heads(self.linear_layers[0](query)), key, value, mask=mask)
        # Concat attention heads
        x = x.transpose(1, 2).contiguous().view(query.size(0), -1, self.num_heads * self.d_k)
        return self.output_linear(x)

    def attend_transposed(
        self, query: torch.Tensor, key_t: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Attention (inference) over keys transposed beforehand, of shape (N, num_heads, d_k, S): an exported decoding
        loop then has no transpose of the keys at each step (cf. `attend` for the other arguments)"""
        scores = torch.matmul(self._split_heads(self.linear_layers[0](query)), key_t) * self.d_k**-0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        x = torch.matmul(scores.softmax(-1), value)
        return self.output_linear(x.transpose(1, 2).flatten(2))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        return self.attend(query, *self.project_kv(key, value), mask=mask)


class EncoderBlock(nn.Module):
    """Transformer Encoder Block"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        dff: int,  # hidden dimension of the feedforward network
        dropout: float,
        activation_fct: Callable[[Any], Any] = nn.ReLU(),
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.layer_norm_input = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_attention = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_output = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.ModuleList([
            MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)
        ])
        self.position_feed_forward = nn.ModuleList([
            PositionwiseFeedForward(d_model, dff, dropout, activation_fct) for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        output = x

        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, mask))
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))

        # (batch_size, seq_len, d_model)
        return self.layer_norm_output(output)


class Decoder(nn.Module):
    """Transformer Decoder"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        vocab_size: int,
        dropout: float = 0.2,
        dff: int = 2048,  # hidden dimension of the feedforward network
        maximum_position_encoding: int = 50,
    ) -> None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.layer_norm_input = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_masked_attention = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_attention = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_output = nn.LayerNorm(d_model, eps=1e-5)

        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, maximum_position_encoding)

        self.attention = nn.ModuleList([
            MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)
        ])
        self.source_attention = nn.ModuleList([
            MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)
        ])
        self.position_feed_forward = nn.ModuleList([
            PositionwiseFeedForward(d_model, dff, dropout) for _ in range(self.num_layers)
        ])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        cache: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] | None = None,
    ) -> torch.Tensor:
        """Decode the target tokens

        Args:
            tgt: target tokens of shape (N, L)
            memory: encoded source of shape (N, S, d_model)
            source_mask: optional mask of the source
            target_mask: optional mask of the target
            cache: for incremental decoding, an initially empty list that keeps the keys and values of every layer
                between the calls: `tgt` then only holds the new tokens, which follow the ones of the previous calls
                and attend to all of them (the source is projected once, at the first call)

        Returns:
            the decoded features of shape (N, L, d_model)
        """
        start = 0 if not cache else cache[0]["self"][0].size(2)
        tgt = self.embed(tgt) * math.sqrt(self.d_model)
        output = self.positional_encoding(tgt, start)

        for i in range(self.num_layers):
            self_attention = cast(MultiHeadAttention, self.attention[i])
            source_attention = cast(MultiHeadAttention, self.source_attention[i])
            normed_output = self.layer_norm_input(output)
            if cache is None:
                attended = self_attention(normed_output, normed_output, normed_output, target_mask)
            else:
                if len(cache) <= i:
                    cache.append({"source": source_attention.project_kv(memory, memory)})
                key, value = self_attention.project_kv(normed_output, normed_output)
                if "self" in cache[i]:
                    key = torch.cat([cache[i]["self"][0], key], dim=2)
                    value = torch.cat([cache[i]["self"][1], value], dim=2)
                cache[i]["self"] = (key, value)
                attended = self_attention.attend(normed_output, key, value, target_mask)
            output = output + self.dropout(attended)

            normed_output = self.layer_norm_masked_attention(output)
            if cache is None:
                attended = source_attention(normed_output, memory, memory, source_mask)
            else:
                attended = source_attention.attend(normed_output, *cache[i]["source"], source_mask)
            output = output + self.dropout(attended)
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))

        # (batch_size, seq_len, d_model)
        return self.layer_norm_output(output)

    def decode_step(
        self,
        token: torch.Tensor,
        step: torch.Tensor,
        source_kv: list[tuple[torch.Tensor, torch.Tensor]],
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Decode one token with key / value buffers of a fixed size (inference), as the loop state of a
        `torch.while_loop`, which the `torch.export`-based ONNX exporter turns into an ONNX `Loop`

        Args:
            token: the token to decode, of shape (N, 1)
            step: its position in the sequence, a scalar tensor
            source_kv: the transposed keys, of shape (N, num_heads, d_model / num_heads, S), and the values of the
                source, for each layer (cf. `MultiHeadAttention.project_kv`)
            keys: the transposed key buffer of each layer, of shape (N, num_heads, d_model / num_heads, L), filled
                before `step`
            values: the value buffer of each layer, of shape (N, num_heads, L, d_model / num_heads)

        Returns:
            the decoded token of shape (N, 1, d_model), and the buffers with its key and value written at `step`
        """
        idx = step.view(1)
        pe = cast(torch.Tensor, self.positional_encoding.pe)
        output = self.embed(token) * math.sqrt(self.d_model) + pe.index_select(1, idx)
        # The token attends to itself and to the previous ones
        mask = (torch.arange(values[0].shape[2], device=token.device) <= step).view(1, 1, 1, -1)
        new_keys, new_values = [], []
        for i in range(self.num_layers):
            self_attention = cast(MultiHeadAttention, self.attention[i])
            source_attention = cast(MultiHeadAttention, self.source_attention[i])
            normed_output = self.layer_norm_input(output)
            key, value = self_attention.project_kv(normed_output, normed_output)
            new_keys.append(keys[i].index_copy(3, idx, key.transpose(-2, -1)))
            new_values.append(values[i].index_copy(2, idx, value))
            output = output + self_attention.attend_transposed(normed_output, new_keys[-1], new_values[-1], mask)
            normed_output = self.layer_norm_masked_attention(output)
            output = output + source_attention.attend_transposed(normed_output, *source_kv[i])
            output = output + self.position_feed_forward[i](self.layer_norm_attention(output))
        return self.layer_norm_output(output), new_keys, new_values
