# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import MultiHeadAttention, PositionwiseFeedForward
from doctr.utils.repr import NestedObject

from ...utils import load_pretrained_params

__all__ = ["vit"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "vit": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
}

tf.config.run_functions_eagerly(True)


class PatchEmbedding(layers.Layer, NestedObject):
    """Compute 2D patch embedding"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        embed_dim: int,
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
        patch_size: Tuple[int, int],
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.include_top = include_top

        self.patch_embedding = PatchEmbedding(input_shape[:-1], self.patch_size, d_model)
        self.num_patches = self.patch_embedding.num_patches
        self.cls_token = self.add_weight(shape=(1, 1, d_model), initializer="zeros", trainable=True, name="cls_token")
        self.positions = self.add_weight(
            shape=(1, self.num_patches + 1, d_model), initializer="zeros", trainable=True, name="positions"
        )

        self.layer_norm = layers.LayerNormalization(epsilon=1e-5)
        self.dropout = layers.Dropout(rate=dropout)

        self.attention = [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        self.position_feed_forward = [
            PositionwiseFeedForward(d_model, d_model, dropout, use_gelu=True) for _ in range(self.num_layers)
        ]

        if self.include_top:
            self.head = layers.Dense(num_classes, kernel_initializer="he_normal")

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
        output = self.layer_norm(output, **kwargs)
        if self.include_top:
            # (batch_size, num_classes) cls token
            output = self.head(output[:, 0], **kwargs)

        return output


def _vit(
    arch: str,
    pretrained: bool,
    patch_size: Tuple[int, int],
    embed_dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    dropout: float = 0.0,
    include_top: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> VisionTransformer:

    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    kwargs.pop("classes")

    # Build the model
    model = VisionTransformer(
        patch_size=patch_size,
        d_model=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        include_top=include_top,
        cfg=_cfg,
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def vit(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import vit
    >>> model = vit(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 32, 32, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A feature extractor model
    """

    return _vit(
        "vit",
        pretrained,
        patch_size=[4, 4],
        **kwargs,
    )
