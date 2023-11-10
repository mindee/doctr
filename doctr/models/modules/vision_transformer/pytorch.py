# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Tuple

import torch
from torch import nn

__all__ = ["PatchEmbedding"]


class PatchEmbedding(nn.Module):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    def __init__(self, input_shape: Tuple[int, int, int], embed_dim: int, patch_size: Tuple[int, int]) -> None:
        super().__init__()
        channels, height, width = input_shape
        self.patch_size = patch_size
        self.interpolate = True if patch_size[0] == patch_size[1] else False
        self.grid_size = tuple([s // p for s, p in zip((height, width), self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.projection = nn.Conv2d(channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """100 % borrowed from:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py

        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.positions.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.positions
        class_pos_embed = self.positions[:, 0]
        patch_pos_embed = self.positions[:, 1:]
        dim = embeddings.shape[-1]
        h0 = float(height // self.patch_size[0])
        w0 = float(width // self.patch_size[1])
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )
        assert int(h0) == patch_pos_embed.shape[-2], "height of interpolated patch embedding doesn't match"
        assert int(w0) == patch_pos_embed.shape[-1], "width of interpolated patch embedding doesn't match"

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"

        # patchify image
        patches = self.projection(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch_size, 1, d_model)
        # concate cls_tokens to patches
        embeddings = torch.cat([cls_tokens, patches], dim=1)  # (batch_size, num_patches + 1, d_model)
        # add positions to embeddings
        if self.interpolate:
            embeddings += self.interpolate_pos_encoding(embeddings, H, W)
        else:
            embeddings += self.positions

        return embeddings  # (batch_size, num_patches + 1, d_model)
