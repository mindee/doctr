# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from typing import Tuple

import torch
from torch import nn

from ..transformer.pytorch import PositionalEncoding

__all__ = ["PatchEmbedding"]


class PatchEmbedding(nn.Module):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    # Inpired by: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        channels: int,
        embed_dim: int,
    ) -> None:

        super().__init__()
        self.img_size = img_size
        self.patch_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # type: ignore[attr-defined]
        # self.positions = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  # type: ignore[attr-defined]
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=self.num_patches + 1)
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"

        patches = self.proj(x)  # BCHW

        B, C, H, W = patches.size()
        patches = patches.view(B, C, -1).permute(0, 2, 1)  # (batch_size, num_patches, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch_size, 1, d_model)
        # concate cls_tokens to patches
        embeddings = torch.cat([cls_tokens, patches], dim=1)  # (batch_size, num_patches + 1, d_model)
        # add positions to embeddings
        embeddings = self.positional_encoding(embeddings)  # (batch_size, num_patches + 1, d_model)

        return embeddings
