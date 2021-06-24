# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import torch
from torch import nn
from typing import Tuple

__all__ = ['Decoder', 'positional_encoding', 'create_look_ahead_mask', 'create_padding_mask']


def positional_encoding(position: int, d_model: int = 512) -> torch.Tensor:
    """Implementation borrowed from this pytorch tutorial:
    <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_.

    Args:
        position: Number of positions to encode
        d_model: depth of the encoding

    Returns:
        2D positional encoding as described in Transformer paper.
    """
    pe = torch.zeros(position, d_model)
    position = torch.arange(0, position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe


def decoder_layer(d_model, num_heads, dff):
    return nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=dff,
        dropout=0.1,
        activation='relu',
        layer_norm_eps=1e-05,
        batch_first=False,
    )


def create_look_ahead_mask():
    pass


def create_padding_mask():
    pass


class Decoder(nn.Module):

    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 512,
        num_heads: int = 8,
        dff: int = 2048,
        vocab_size: int = 120,
        maximum_position_encoding: int = 50,
    ) -> None:
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size + 2, d_model)  # 2 more classes EOS/SOS
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [decoder_layer(d_model, num_heads, dff)
                           for _ in range(num_layers)]

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

        # x.shape == (batch_size, target_seq_len, d_model)
        return x
