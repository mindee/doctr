# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from doctr.utils.repr import NestedObject

__all__ = ["PatchEmbedding"]


class PatchEmbedding(layers.Layer, NestedObject):
    """Compute 2D patch embeddings with cls token and positional encoding"""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        embed_dim: int,
    ) -> None:

        super().__init__()
        height, width, _ = input_shape
        # fix patch size if recognition task with 32x128 input
        self.patch_size = (4, 8) if height != width else patch_size
        self.grid_size = (height // patch_size[0], width // patch_size[1])
        self.num_patches = (height // patch_size[0]) * (width // patch_size[1])

        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer="zeros", trainable=True, name="cls_token")
        self.positions = self.add_weight(
            shape=(1, self.num_patches + 1, embed_dim),
            initializer="zeros",
            trainable=True,
            name="positions",
        )
        self.proj = layers.Dense(embed_dim, kernel_initializer="he_normal")

    def interpolate_pos_encoding(self, embeddings: tf.Tensor, height: int, width: int) -> tf.Tensor:
        """
        100 % borrowed from:
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
            method="bicubic",
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
        # patchify image without convolution
        # adapted from:
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
        # NOTE: tf.image.extract_patches has no ONNX support and Conv2D with padding=valid consumes to much memory
        patches = tf.reshape(
            x, (B, H // self.patch_size[0], self.patch_size[0], W // self.patch_size[1], self.patch_size[1], C)
        )
        patches = tf.transpose(a=patches, perm=(0, 1, 3, 2, 4, 5))
        # (B, H', W', C, ph, pw) -> (B, H'*W', C*ph*pw)
        patches = tf.reshape(tensor=patches, shape=(B, -1, self.patch_size[0] * self.patch_size[1] * C))
        patches = self.proj(patches, **kwargs)  # (batch_size, num_patches, d_model)

        cls_tokens = tf.repeat(self.cls_token, B, axis=0)  # (batch_size, 1, d_model)
        # concate cls_tokens to patches
        embeddings = tf.concat([cls_tokens, patches], axis=1)  # (batch_size, num_patches + 1, d_model)
        # add positions to embeddings
        embeddings += self.interpolate_pos_encoding(embeddings, H, W)  # (batch_size, num_patches + 1, d_model)

        return embeddings
