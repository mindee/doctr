# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# This module 'transformer.py' is inspired by https://github.com/wenwenyu/MASTER-pytorch and Decoder is borrowed

import math
from typing import Any, Callable, Optional, Tuple

import torch
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
        ----
            x: embeddings (batch, max_len, d_model)

        Returns
        -------
            positional embeddings (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention"""
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        # NOTE: to ensure the ONNX compatibility, masked_fill works only with int equal condition
        scores = scores.masked_fill(mask == 0, float("-inf"))  # type: ignore[attr-defined]
    p_attn = torch.softmax(scores, dim=-1)  # type: ignore[call-overload]
    return torch.matmul(p_attn, value), p_attn


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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size = query.size(0)

        # linear projections of Q, K, V
        query, key, value = [
            linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]

        # apply attention on all the projected vectors in batch
        x, attn = scaled_dot_product_attention(query, key, value, mask=mask)

        # Concat attention heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.output_linear(x)


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

        self.attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        )
        self.position_feed_forward = nn.ModuleList(
            [PositionwiseFeedForward(d_model, dff, dropout, activation_fct) for _ in range(self.num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        self.attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        )
        self.source_attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        )
        self.position_feed_forward = nn.ModuleList(
            [PositionwiseFeedForward(d_model, dff, dropout) for _ in range(self.num_layers)]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt = self.embed(tgt) * math.sqrt(self.d_model)
        pos_enc_tgt = self.positional_encoding(tgt)
        output = pos_enc_tgt

        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, target_mask))
            normed_output = self.layer_norm_masked_attention(output)
            output = output + self.dropout(self.source_attention[i](normed_output, memory, memory, source_mask))
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))

        # (batch_size, seq_len, d_model)
        return self.layer_norm_output(output)
