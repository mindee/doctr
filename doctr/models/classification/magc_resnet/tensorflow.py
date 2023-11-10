# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params
from ..resnet.tensorflow import ResNet

__all__ = ["magc_resnet31"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "magc_resnet31": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.6.0/magc_resnet31-addbb705.zip&src=0",
    },
}


class MAGC(layers.Layer):
    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
    ----
        inplanes: input channels
        headers: number of headers to split channels
        attn_scale: if True, re-scale attention to counteract the variance distibutions
        ratio: bottleneck ratio
        **kwargs
    """

    def __init__(
        self,
        inplanes: int,
        headers: int = 8,
        attn_scale: bool = False,
        ratio: float = 0.0625,  # bottleneck ratio of 1/16 as described in paper
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.headers = headers  # h
        self.inplanes = inplanes  # C
        self.attn_scale = attn_scale
        self.planes = int(inplanes * ratio)

        self.single_header_inplanes = int(inplanes / headers)  # C / h

        self.conv_mask = layers.Conv2D(filters=1, kernel_size=1, kernel_initializer=tf.initializers.he_normal())

        self.transform = Sequential(
            [
                layers.Conv2D(filters=self.planes, kernel_size=1, kernel_initializer=tf.initializers.he_normal()),
                layers.LayerNormalization([1, 2, 3]),
                layers.ReLU(),
                layers.Conv2D(filters=self.inplanes, kernel_size=1, kernel_initializer=tf.initializers.he_normal()),
            ],
            name="transform",
        )

    def context_modeling(self, inputs: tf.Tensor) -> tf.Tensor:
        b, h, w, c = (tf.shape(inputs)[i] for i in range(4))

        # B, H, W, C -->> B*h, H, W, C/h
        x = tf.reshape(inputs, shape=(b, h, w, self.headers, self.single_header_inplanes))
        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        x = tf.reshape(x, shape=(b * self.headers, h, w, self.single_header_inplanes))

        # Compute shorcut
        shortcut = x
        # B*h, 1, H*W, C/h
        shortcut = tf.reshape(shortcut, shape=(b * self.headers, 1, h * w, self.single_header_inplanes))
        # B*h, 1, C/h, H*W
        shortcut = tf.transpose(shortcut, perm=[0, 1, 3, 2])

        # Compute context mask
        # B*h, H, W, 1
        context_mask = self.conv_mask(x)
        # B*h, 1, H*W, 1
        context_mask = tf.reshape(context_mask, shape=(b * self.headers, 1, h * w, 1))
        # scale variance
        if self.attn_scale and self.headers > 1:
            context_mask = context_mask / math.sqrt(self.single_header_inplanes)
        # B*h, 1, H*W, 1
        context_mask = tf.keras.activations.softmax(context_mask, axis=2)

        # Compute context
        # B*h, 1, C/h, 1
        context = tf.matmul(shortcut, context_mask)
        context = tf.reshape(context, shape=(b, 1, c, 1))
        # B, 1, 1, C
        context = tf.transpose(context, perm=(0, 1, 3, 2))
        # Set shape to resolve shape when calling this module in the Sequential MAGCResnet
        batch, chan = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[-1]
        context.set_shape([batch, 1, 1, chan])
        return context

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Context modeling: B, H, W, C  ->  B, 1, 1, C
        context = self.context_modeling(inputs)
        # Transform: B, 1, 1, C  ->  B, 1, 1, C
        transformed = self.transform(context)
        return inputs + transformed


def _magc_resnet(
    arch: str,
    pretrained: bool,
    num_blocks: List[int],
    output_channels: List[int],
    stage_downsample: List[bool],
    stage_conv: List[bool],
    stage_pooling: List[Optional[Tuple[int, int]]],
    origin_stem: bool = True,
    **kwargs: Any,
) -> ResNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    kwargs.pop("classes")

    # Build the model
    model = ResNet(
        num_blocks,
        output_channels,
        stage_downsample,
        stage_conv,
        stage_pooling,
        origin_stem,
        attn_module=partial(MAGC, headers=8, attn_scale=True),
        cfg=_cfg,
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def magc_resnet31(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet31 architecture with Multi-Aspect Global Context Attention as described in
    `"MASTER: Multi-Aspect Non-local Network for Scene Text Recognition",
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import magc_resnet31
    >>> model = magc_resnet31(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 224, 224, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the ResNet architecture

    Returns:
    -------
        A feature extractor model
    """
    return _magc_resnet(
        "magc_resnet31",
        pretrained,
        [1, 2, 5, 3],
        [256, 256, 512, 512],
        [False] * 4,
        [True] * 4,
        [(2, 2), (2, 1), None, None],
        False,
        stem_channels=128,
        **kwargs,
    )
