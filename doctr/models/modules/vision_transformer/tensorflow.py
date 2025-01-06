# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from doctr.utils.repr import NestedObject

__all__ = ["PatchEmbedding"]


class PatchEmbedding(layers.Layer, NestedObject):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    def __init__(self, input_shape: tuple[int, int, int], embed_dim: int, patch_size: tuple[int, int]) -> None:
        super().__init__()
        height, width, _ = input_shape
        self.patch_size = patch_size
        self.interpolate = True if patch_size[0] == patch_size[1] else False
        self.grid_size = tuple(s // p for s, p in zip((height, width), self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer="zeros", trainable=True, name="cls_token")
        self.positions = self.add_weight(
            shape=(1, self.num_patches + 1, embed_dim),
            initializer="zeros",
            trainable=True,
            name="positions",
        )
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="projection",
        )

    def interpolate_pos_encoding(self, embeddings: tf.Tensor, height: int, width: int) -> tf.Tensor:
        """100 % borrowed from:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_tf_vit.py

        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py
        """
        seq_len, dim = embeddings.shape[1:]
        num_patches = seq_len - 1

        num_positions = self.positions.shape[1] - 1

        if num_patches == num_positions and height == width:
            return self.positions
        class_pos_embed = self.positions[:, :1]
        patch_pos_embed = self.positions[:, 1:]
        h0 = height // self.patch_size[0]
        w0 = width // self.patch_size[1]
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(
                patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
            ),
            size=(h0, w0),
            method="bilinear",
        )

        shape = patch_pos_embed.shape
        assert h0 == shape[-3], "height of interpolated patch embedding doesn't match"
        assert w0 == shape[-2], "width of interpolated patch embedding doesn't match"

        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        B, H, W, C = x.shape
        assert H % self.patch_size[0] == 0, "Image height must be divisible by patch height"
        assert W % self.patch_size[1] == 0, "Image width must be divisible by patch width"
        # patchify image
        patches = self.projection(x, **kwargs)  # (batch_size, num_patches, d_model)
        patches = tf.reshape(patches, (B, self.num_patches, -1))  # (batch_size, num_patches, d_model)

        cls_tokens = tf.repeat(self.cls_token, B, axis=0)  # (batch_size, 1, d_model)
        # concate cls_tokens to patches
        embeddings = tf.concat([cls_tokens, patches], axis=1)  # (batch_size, num_patches + 1, d_model)
        # add positions to embeddings
        if self.interpolate:
            embeddings += self.interpolate_pos_encoding(embeddings, H, W)
        else:
            embeddings += self.positions

        return embeddings  # (batch_size, num_patches + 1, d_model)
