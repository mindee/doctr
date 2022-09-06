# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from doctr.models.modules import MultiHeadAttention, PositionwiseFeedForward
from doctr.utils.repr import NestedObject

__all__ = ["VisionTransformer"]

tf.config.run_functions_eagerly(True)


class PatchEmbedding(layers.Layer, NestedObject):
    """Compute 2D patch embedding"""

    def __init__(
        self,
        img_size: Tuple[int],
        patch_size: Tuple[int] = (4, 8),  # different from paper to match with 32x128 input
        embed_dim: int = 768,
    ) -> None:

        super().__init__()
        self.img_size = img_size
        self.patch_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            kernel_initializer="he_normal",
        )

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        B, W, H, C = x.shape

        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"

        x = self.proj(x, **kwargs)
        return tf.reshape(x, (x.shape[0], (x.shape[1] * x.shape[2]), x.shape[3]))  # BCHW -> BNC
