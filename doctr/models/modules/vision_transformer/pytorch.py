# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Optional, Tuple

import torch
from torch import nn

from doctr.models.modules import MultiHeadAttention, PositionwiseFeedForward

__all__ = ["VisionTransformer"]


class PatchEmbedding(nn.Module):
    """Compute 2D patch embedding"""

    def __init__(
        self,
        img_size: Tuple[int],
        patch_size: Tuple[int] = (4, 8),  # different from paper to match with 32x128 input
        channels: int = 3,
        embed_dim: int = 768,
    ) -> None:

        super().__init__()
        self.img_size = img_size
        self.patch_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"

        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # BCHW -> BNC
