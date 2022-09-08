# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from doctr.utils.repr import NestedObject

from ..transformer.tensorflow import PositionalEncoding

__all__ = ["PatchEmbedding"]


tf.config.run_functions_eagerly(True)


class PatchEmbedding(layers.Layer, NestedObject):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        embed_dim: int,
    ) -> None:

        super().__init__()
        self.img_size = img_size
        print(self.img_size)
        self.patch_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer="zeros", trainable=True, name="cls_token")
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=self.num_patches + 1)
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
        )

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        B, W, H, C = x.shape
        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"

        patches = self.proj(x, **kwargs)  # BHWC
        print(patches.shape)

        B, H, W, C = patches.shape
        patches = tf.reshape(patches, (B, (H * W), C))  # (batch_size, num_patches, d_model)

        cls_tokens = tf.repeat(self.cls_token, B, axis=0)  # (batch_size, num_patches, d_model)
        # concate cls_tokens to patches
        embeddings = tf.concat([cls_tokens, patches], axis=1)  # (batch_size, num_patches + 1, d_model)
        # add positions to embeddings
        embeddings = self.positional_encoding(embeddings, **kwargs)  # (batch_size, num_patches + 1, d_model)
        print(embeddings.shape)

        return embeddings
