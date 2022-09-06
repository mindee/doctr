# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from doctr.utils.repr import NestedObject

from ..transformer.tensorflow import MultiHeadAttention, PositionwiseFeedForward

__all__ = ["VisionTransformer"]

tf.config.run_functions_eagerly(True)


class PatchEmbedding(layers.Layer, NestedObject):
    """Compute 2D patch embedding"""

    def __init__(
        self,
        img_size: Tuple[int],
        patch_size: Tuple[int],
        embed_dim: int = 768,
    ) -> None:

        super().__init__()
        self.img_size = img_size
        self.patch_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
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

        return self.proj(x, **kwargs)  # BHWC


class VisionTransformer(layers.Layer, NestedObject):
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_."""

    def __init__(
        self,
        img_size: Tuple[int] = (32, 128),  # different from original paper to match our sizes
        patch_size: Tuple[int] = (4, 8),  # different from original paper to match our sizes
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = num_layers

        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size, d_model)
        self.num_patches = self.patch_embedding.num_patches
        self.cls_token = self.add_weight(
            shape=(1, 1, d_model),
            initializer="zeros",
            trainable=True ,
            name="cls_token"
        )
        self.positions = self.add_weight(
            shape=(1, self.num_patches + 1, d_model),
            initializer="zeros",
            trainable=True,
            name="positions"
        )

        self.layer_norm = layers.LayerNormalization(epsilon=1e-5)
        self.dropout = layers.Dropout(rate=dropout)

        self.attention = [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        self.position_feed_forward = [
            PositionwiseFeedForward(d_model, d_model, dropout, use_gelu=True) for _ in range(self.num_layers)
        ]

    def __call__(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        patches = self.patch_embedding(x)

        B, C, H, W = patches.shape
        patches = tf.reshape(patches, (B, (C * H), W))  # (batch_size, num_patches, d_model)

        cls_tokens = tf.repeat(self.cls_token, B, axis=0)  # (batch_size, num_patches, d_model)
        # concate cls_tokens to patches
        embeddings = tf.concat([cls_tokens, patches], axis=1)  # (batch_size, num_patches + 1, d_model)
        # add positions to embeddings
        embeddings += self.positions  # (batch_size, num_patches + 1, d_model)

        output = embeddings

        for i in range(self.num_layers):
            normed_output = self.layer_norm(output, **kwargs)
            output = output + self.dropout(
                self.attention[i](normed_output, normed_output, normed_output, **kwargs),
                **kwargs,
            )
            normed_output = self.layer_norm(output, **kwargs)
            output = output + self.dropout(self.position_feed_forward[i](normed_output, **kwargs), **kwargs)

        # (batch_size, seq_len + cls token, d_model)
        return self.layer_norm(output, **kwargs)
