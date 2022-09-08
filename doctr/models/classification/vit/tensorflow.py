# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Sequential, layers

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import EncoderBlock
from doctr.models.modules.vision_transformer import PatchEmbedding

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


class VisionTransformer(Sequential):
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_."""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        input_shape: Tuple[int, int, int],
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.include_top = include_top

        self.patch_embedding = PatchEmbedding(img_size, patch_size, d_model)
        self.encoder = EncoderBlock(num_layers, num_heads, d_model, dropout, use_gelu=True)

        if self.include_top:
            self.head = layers.Dense(num_classes, kernel_initializer="he_normal")

    def __call__(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:

        embeddings = self.patch_embedding(x, **kwargs)
        encoded = self.encoder(embeddings, **kwargs)

        if self.include_top:
            # (batch_size, num_classes) cls token
            return self.head(encoded[:, 0], **kwargs)

        return encoded


def _vit(
    arch: str,
    pretrained: bool,
    img_size: Tuple[int, int],
    patch_size: Tuple[int, int],
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float = 0.0,
    include_top: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> VisionTransformer:

    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = VisionTransformer(
        img_size=img_size,
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
        img_size=(32, 32),
        patch_size=(4, 4),
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        **kwargs,
    )
