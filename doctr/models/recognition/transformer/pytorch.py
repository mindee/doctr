# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import torch
from torch import nn
from typing import Optional

__all__ = ['Decoder', 'positional_encoding']


def positional_encoding(position: int, d_model: int = 512, dtype=torch.float32) -> torch.Tensor:
    """Implementation borrowed from this pytorch tutorial:
    <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_.

    Args:
        position: Number of positions to encode
        d_model: depth of the encoding

    Returns:
        2D positional encoding as described in Transformer paper.
    """
    pe = torch.zeros(position, d_model)
    pos = torch.arange(0, position, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe.unsqueeze(0)


class Decoder(nn.Module):

    pos_encoding: torch.Tensor

    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 512,
        num_heads: int = 8,
        dff: int = 2048,
        vocab_size: int = 120,
        maximum_position_encoding: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size + 3, d_model)  # 3 more classes EOS/SOS/PAD
        self.register_buffer('pos_encoding', positional_encoding(maximum_position_encoding, d_model))

        self.dec_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dff,
                dropout=dropout,
                activation='relu',
            ) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        seq_len = x.shape[1]  # Batch first = True

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= math.sqrt(self.d_model)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Batch first = False in decoder
        x = x.permute(1, 0, 2)
        for i in range(self.num_layers):
            x = self.dec_layers[i](
                tgt=x, memory=enc_output, tgt_mask=look_ahead_mask, memory_mask=padding_mask
            )

        # shape (batch_size, target_seq_len, d_model)
        x = x.permute(1, 0, 2)
        return x
