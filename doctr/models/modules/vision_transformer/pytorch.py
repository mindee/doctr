# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Tuple

import torch
from torch import nn

from ..transformer.pytorch import MultiHeadAttention, PositionwiseFeedForward

__all__ = ["VisionTransformer"]


class PatchEmbedding(nn.Module):
    """Compute 2D patch embedding"""

    # Inpired by: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        channels: int,
        embed_dim: int = 768,
    ) -> None:

        super().__init__()
        self.img_size = img_size
        self.patch_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"

        return self.proj(x)  # BCHW


class VisionTransformer(nn.Module):
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_."""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        channels: int = 3,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = num_layers

        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size, channels, d_model)
        self.num_patches = self.patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # type: ignore[attr-defined]
        self.positions = nn.Parameter(  # type: ignore[attr-defined]
            torch.randn(1, self.num_patches + 1, d_model) * 0.02
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        )
        self.position_feed_forward = nn.ModuleList(
            [PositionwiseFeedForward(d_model, d_model, dropout, use_gelu=True) for _ in range(self.num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(x)

        B, C, H, W = patches.shape
        patches = patches.view(B, C, -1).permute(0, 2, 1)  # (batch_size, num_patches, d_model)
        cls_tokens = self.cls_token.repeat(B, 1, 1)  # (batch_size, 1, d_model)
        # concate cls_tokens to patches
        embeddings = torch.cat([cls_tokens, patches], dim=1)  # (batch_size, num_patches + 1, d_model)
        # add positions to embeddings
        embeddings += self.positions  # (batch_size, num_patches + 1, d_model)

        output = embeddings

        for i in range(self.num_layers):
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output))
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))

        # (batch_size, seq_len + cls token, d_model)
        return self.layer_norm(output)
