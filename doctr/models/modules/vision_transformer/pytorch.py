# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from typing import Tuple

import torch
from torch import nn

__all__ = ["PatchEmbedding"]


class PatchEmbedding(nn.Module):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        embed_dim: int,
    ) -> None:

        super().__init__()
        channels, height, width = input_shape
        self.patch_size = patch_size
        self.grid_size = (height // patch_size[0], width // patch_size[1])
        self.num_patches = (height // patch_size[0]) * (width // patch_size[1])

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # type: ignore[attr-defined]
        self.positions = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  # type: ignore[attr-defined]
        self.proj = nn.Linear((channels * self.patch_size[0] * self.patch_size[1]), embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"

        # patchify image without convolution
        # adapted from:
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
        # NOTE: patchify with Conv2d works only with padding="valid" correctly on smaller images
        # and has currently no ONNX support so we use this workaround
        x = x.reshape(
            B, C, (H // self.patch_size[0]), self.patch_size[0], (W // self.patch_size[1]), self.patch_size[1]
        )
        # (B, H', W', C, ph, pw) -> (B, H'*W', C*ph*pw)
        patches = x.permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2, 4)
        patches = self.proj(patches)  # (batch_size, num_patches, d_model)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch_size, 1, d_model)
        # concate cls_tokens to patches
        embeddings = torch.cat([cls_tokens, patches], dim=1)  # (batch_size, num_patches + 1, d_model)
        # add positions to embeddings
        embeddings += self.positions  # (batch_size, num_patches + 1, d_model)

        return embeddings
