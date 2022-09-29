# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import math
from typing import Tuple

import torch
from torch import nn

__all__ = ["PatchEmbedding"]


class PatchEmbedding(nn.Module):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    def __init__(self, input_shape: Tuple[int, int, int], embed_dim: int) -> None:

        super().__init__()
        channels, height, width = input_shape
        # calculate patch size
        # NOTE: this is different from the original implementation
        self.patch_size = (height // (height // 8), width // (width // 8))

        self.grid_size = (self.patch_size[0], self.patch_size[1])
        self.num_patches = self.patch_size[0] * self.patch_size[1]

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # type: ignore[attr-defined]
        self.positions = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  # type: ignore[attr-defined]
        self.proj = nn.Linear((channels * self.patch_size[0] * self.patch_size[1]), embed_dim)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        100 % borrowed from:
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
        embeddings += self.interpolate_pos_encoding(embeddings, H, W)  # (batch_size, num_patches + 1, d_model)

        return embeddings
