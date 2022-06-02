# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import copy
import math
from typing import Optional

import torch
from torch import nn

__all__ = ['Decoder', 'PositionalEncoding']


class PositionalEncoding(nn.Module):
    """ Compute positional encoding """

    def __init__(self, d_model: int, dropout_prob: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: embeddings (batch, max_len, d_model)

        Returns:
            positional embeddings (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# TODO: ---------------------------------------------------------------------------------------


def clones(_to_clone_module, _clone_times, _is_deep=True):
    """Produce N identical layers."""
    copy_method = copy.deepcopy if _is_deep else copy.copy
    return nn.ModuleList([copy_method(_to_clone_module) for _ in range(_clone_times if _is_deep else 1)])

class MultiHeadAttention(nn.Module):
    def __init__(self, _multi_attention_heads, _dimensions, _dropout=0.1):
        """
        :param _multi_attention_heads: number of self attention head
        :param _dimensions: dimension of model
        :param _dropout:
        """
        super(MultiHeadAttention, self).__init__()

        assert _dimensions % _multi_attention_heads == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(_dimensions / _multi_attention_heads)
        self.h = _multi_attention_heads
        self.linears = clones(nn.Linear(_dimensions, _dimensions), 4)  # (q, k, v, last output layer)
        self.attention = None
        self.dropout = nn.Dropout(p=_dropout)

    def dot_product_attention(self, _query, _key, _value, _mask):
        """
        Compute 'Scaled Dot Product Attention
        :param _query: (N, h, seq_len, d_q), h is multi-head
        :param _key: (N, h, seq_len, d_k)
        :param _value: (N, h, seq_len, d_v)
        :param _mask: None or (N, 1, seq_len, seq_len), 0 will be replaced with -1e9
        :return:
        """

        d_k = _value.size(-1)
        score = torch.matmul(_query, _key.transpose(-2, -1)) / math.sqrt(d_k)  # (N, h, seq_len, seq_len)
        if _mask is not None:
            score = score.masked_fill(_mask == 0, -1e9)  # score (N, h, seq_len, seq_len)
        p_attn = torch.softmax(score, dim=-1)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        return torch.matmul(p_attn, _value), p_attn

    def forward(self, _query, _key, _value, _mask):
        batch_size = _query.size(0)

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        _query, _key, _value = \
            [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (_query, _key, _value))]

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(_query, _key, _value, _mask=_mask)
        x = product_and_attention[0]
        # self.attention = self.dropout(product_and_attention[1])

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, _dimensions, _feed_forward_dimensions, _dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(_dimensions, _feed_forward_dimensions)
        self.w_2 = nn.Linear(_feed_forward_dimensions, _dimensions)
        self.dropout = nn.Dropout(p=_dropout)

    def forward(self, _input_tensor):
        return self.w_2(self.dropout(torch.relu(self.w_1(_input_tensor))))


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_layers: int=6,
                 num_heads: int=8,
                 d_model: int=512,
                 dropout: float = 0.2,
                 dff: int = 2048,
                 max_length: int = 100,
                 padding_symbol: int = 0
                 ) -> None:

        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.attention = nn.ModuleList([
            MultiHeadAttention(num_heads, d_model, dropout)
            for _ in range(self.num_layers)
        ])
        self.source_attention = nn.ModuleList([
            MultiHeadAttention(num_heads, d_model, dropout)
            for _ in range(self.num_layers)
        ])
        self.position_feed_forward = nn.ModuleList([
            PositionwiseFeedForward(d_model, dff, dropout)
            for _ in range(self.num_layers)
        ])
        self.position = PositionalEncoding(d_model, dropout, max_length)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.sqrt_model_size = math.sqrt(d_model)
        self.padding_symbol = padding_symbol

    def _generate_target_mask(self, _source, _target):
        target_pad_mask = (_target != self.padding_symbol).unsqueeze(1).unsqueeze(3)  # (b, 1, len_src, 1)
        target_length = _target.size(1)
        target_sub_mask = torch.tril(
            torch.ones((target_length, target_length), dtype=torch.uint8, device=_source.device)
        )
        source_mask = torch.ones((target_length, _source.size(1)), dtype=torch.uint8, device=_source.device)
        target_mask = target_pad_mask & target_sub_mask.bool()
        return source_mask, target_mask


    def forward(self, _target_result, _memory):
        target = self.embedding(_target_result) * self.sqrt_model_size
        target = self.position(target)
        source_mask, target_mask = self._generate_target_mask(_memory, _target_result)
        output = target
        for i in range(self.num_layers):
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.attention[i](normed_output, normed_output, normed_output, target_mask)
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.source_attention[i](normed_output, _memory, _memory, source_mask))
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))
        return self.layer_norm(output)
