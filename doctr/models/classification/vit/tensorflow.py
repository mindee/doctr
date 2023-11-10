# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Sequential, layers

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import EncoderBlock
from doctr.models.modules.vision_transformer.tensorflow import PatchEmbedding
from doctr.utils.repr import NestedObject

from ...utils import load_pretrained_params

__all__ = ["vit_s", "vit_b"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "vit_s": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.6.0/vit_s-6300fcc9.zip&src=0",
    },
    "vit_b": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.6.0/vit_b-57158446.zip&src=0",
    },
}


class ClassifierHead(layers.Layer, NestedObject):
    """Classifier head for Vision Transformer

    Args:
    ----
        num_classes: number of output classes
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.head = layers.Dense(num_classes, kernel_initializer="he_normal", name="dense")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # (batch_size, num_classes) cls token
        return self.head(x[:, 0])


class VisionTransformer(Sequential):
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    Args:
    ----
        d_model: dimension of the transformer layers
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffd_ratio: multiplier for the hidden dimension of the feedforward layer
        patch_size: size of the patches
        input_shape: size of the input image
        dropout: dropout rate
        num_classes: number of output classes
        include_top: whether the classifier head should be instantiated
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffd_ratio: int,
        patch_size: Tuple[int, int] = (4, 4),
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        dropout: float = 0.0,
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        _layers = [
            PatchEmbedding(input_shape, d_model, patch_size),
            EncoderBlock(
                num_layers,
                num_heads,
                d_model,
                d_model * ffd_ratio,
                dropout,
                activation_fct=layers.Activation("gelu"),
            ),
        ]
        if include_top:
            _layers.append(ClassifierHead(num_classes))

        super().__init__(_layers)
        self.cfg = cfg


def _vit(
    arch: str,
    pretrained: bool,
    **kwargs: Any,
) -> VisionTransformer:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = VisionTransformer(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def vit_s(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """VisionTransformer-S architecture
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_. Patches: (H, W) -> (H/8, W/8)

    NOTE: unofficial config used in ViTSTR and ParSeq

    >>> import tensorflow as tf
    >>> from doctr.models import vit_s
    >>> model = vit_s(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 32, 32, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the VisionTransformer architecture

    Returns:
    -------
        A feature extractor model
    """
    return _vit(
        "vit_s",
        pretrained,
        d_model=384,
        num_layers=12,
        num_heads=6,
        ffd_ratio=4,
        **kwargs,
    )


def vit_b(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """VisionTransformer-B architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_. Patches: (H, W) -> (H/8, W/8)

    >>> import tensorflow as tf
    >>> from doctr.models import vit_b
    >>> model = vit_b(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 32, 32, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the VisionTransformer architecture

    Returns:
    -------
        A feature extractor model
    """
    return _vit(
        "vit_b",
        pretrained,
        d_model=768,
        num_layers=12,
        num_heads=12,
        ffd_ratio=4,
        **kwargs,
    )
