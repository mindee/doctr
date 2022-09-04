# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from doctr.datasets import VOCABS
from doctr.models.modules import Encoder, PatchEmbedding

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


class VisionTransformer(layers.Layer):
    """Implements An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, as described in
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    Args:
        TODO !!
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(**kwargs)

        # TODO

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        # TODO
        return model


def _vit(
    arch: str,
    pretrained: bool,
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
    model = VisionTransformer(cfg=_cfg, **kwargs)
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
    >>> input_tensor = tf.random.uniform(shape=[1, 224, 224, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A vit model
    """

    return _vit(
        "vit",
        pretrained,
        **kwargs,
    )
