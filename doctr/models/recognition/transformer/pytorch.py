import copy
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout

__all__ = ['Decoder', 'PositionalEncoding']


def clones(module: nn.Module, times: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(times)])


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.2) -> None:
        super(MultiHeadAttention, self).__init__()

        self.d_k = int(d_model / num_heads)
        self.heads = num_heads
        self.linears = (nn.Linear(d_model, d_model), 4)  # (q, k, v, last output layer)
        self.dropout = nn.Dropout(dropout)

    def dot_product_attention(self, q, k, v, mask) -> Tuple[torch.Tensor, torch.Tensor]:

        d_k = v.size(-1)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (N, h, seq_len, seq_len)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # score (N, h, seq_len, seq_len)
        p_attn = F.softmax(score, dim=-1)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        return (torch.matmul(p_attn, v), p_attn)

    def forward(self, q, k, v, mask) -> torch.Tensor:
        batch_size = q.size(0)

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        q, k, v = [linear(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
                   for linear, x in zip(self.linears, (q, k, v))]

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(q, k, v, mask)
        x = product_and_attention[0]
        # self.attention = self.dropout(product_and_attention[1])

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int = 512, feed_forward_dimensions: int = 2048, dropout: float = 0.2):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, feed_forward_dimensions)
        self.w_2 = nn.Linear(feed_forward_dimensions, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 512, dropout: float = 0.2, max_len: int = 50):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('feature_pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.feature_pe[:, :x.size(1)]  # pe 1 5000 512
        return self.dropout(x)


class Decoder(nn.Module):

    def __init__(self,
                 num_layers: int = 3,
                 d_model: int = 512,
                 num_heads: int = 8,
                 feed_forward_dimensions: int = 2048,
                 vocab_size: int = 10000,
                 dropout: float = 0.2
                 ) -> None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.attention = nn.ModuleList(
            [
                MultiHeadAttention(d_model, num_heads, dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.source_attention = nn.ModuleList(
            [
                MultiHeadAttention(d_model, num_heads, dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.position_feed_forward = nn.ModuleList(
            [
                PositionwiseFeedForward(d_model, feed_forward_dimensions, dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.position = PositionalEncoding(d_model, dropout)
        self.dropout = Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.sqrt_model_size = math.sqrt(d_model)
        self.padding_symbol = 0

    def _generate_target_mask(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_pad_mask = (target != self.padding_symbol).unsqueeze(1).unsqueeze(3)  # (b, 1, len_src, 1)
        target_length = target.size(1)
        target_sub_mask = torch.tril(
            torch.ones((target_length, target_length), dtype=torch.uint8, device=source.device)
        )
        source_mask = torch.ones((target_length, source.size(1)), dtype=torch.uint8, device=source.device)
        target_mask = target_pad_mask & target_sub_mask.bool()
        return source_mask, target_mask

    def forward(self, target_result: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        target = self.embedding(target_result) * self.sqrt_model_size
        target = self.position(target)
        source_mask, target_mask = self._generate_target_mask(memory, target_result)
        output = target
        for i in range(self.num_layers):
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.attention[i](normed_output, normed_output, normed_output, target_mask)
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.source_attention[i](normed_output, memory, memory, source_mask))
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))
        return self.layer_norm(output)
