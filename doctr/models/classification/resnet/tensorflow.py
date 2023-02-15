# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential

from doctr.datasets import VOCABS

from ...utils import conv_sequence, load_pretrained_params

__all__ = ["ResNet", "resnet18", "resnet31", "resnet34", "resnet50", "resnet34_wide"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/resnet18-d4634669.zip&src=0",
    },
    "resnet31": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.5.0/resnet31-5a47a60b.zip&src=0",
    },
    "resnet34": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.5.0/resnet34-5dcc97ca.zip&src=0",
    },
    "resnet50": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.5.0/resnet50-e75e4cdf.zip&src=0",
    },
    "resnet34_wide": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 32, 3),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.5.0/resnet34_wide-c1271816.zip&src=0",
    },
}


class ResnetBlock(layers.Layer):

    """Implements a resnet31 block with shortcut

    Args:
        conv_shortcut: Use of shortcut
        output_channels: number of channels to use in Conv2D
        kernel_size: size of square kernels
        strides: strides to use in the first convolution of the block
    """

    def __init__(self, output_channels: int, conv_shortcut: bool, strides: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        if conv_shortcut:
            self.shortcut = Sequential(
                [
                    layers.Conv2D(
                        filters=output_channels,
                        strides=strides,
                        padding="same",
                        kernel_size=1,
                        use_bias=False,
                        kernel_initializer="he_normal",
                    ),
                    layers.BatchNormalization(),
                ]
            )
        else:
            self.shortcut = layers.Lambda(lambda x: x)
        self.conv_block = Sequential(self.conv_resnetblock(output_channels, 3, strides))
        self.act = layers.Activation("relu")

    @staticmethod
    def conv_resnetblock(
        output_channels: int,
        kernel_size: int,
        strides: int = 1,
    ) -> List[layers.Layer]:
        return [
            *conv_sequence(output_channels, "relu", bn=True, strides=strides, kernel_size=kernel_size),
            *conv_sequence(output_channels, None, bn=True, kernel_size=kernel_size),
        ]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        clone = self.shortcut(inputs)
        conv_out = self.conv_block(inputs)
        out = self.act(clone + conv_out)

        return out


def resnet_stage(
    num_blocks: int, out_channels: int, shortcut: bool = False, downsample: bool = False
) -> List[layers.Layer]:
    _layers: List[layers.Layer] = [ResnetBlock(out_channels, conv_shortcut=shortcut, strides=2 if downsample else 1)]

    for _ in range(1, num_blocks):
        _layers.append(ResnetBlock(out_channels, conv_shortcut=False))

    return _layers


class ResNet(Sequential):
    """Implements a ResNet architecture

    Args:
        num_blocks: number of resnet block in each stage
        output_channels: number of channels in each stage
        stage_downsample: whether the first residual block of a stage should downsample
        stage_conv: whether to add a conv_sequence after each stage
        stage_pooling: pooling to add after each stage (if None, no pooling)
        origin_stem: whether to use the orginal ResNet stem or ResNet-31's
        stem_channels: number of output channels of the stem convolutions
        attn_module: attention module to use in each stage
        include_top: whether the classifier head should be instantiated
        num_classes: number of output classes
        input_shape: shape of inputs
    """

    def __init__(
        self,
        num_blocks: List[int],
        output_channels: List[int],
        stage_downsample: List[bool],
        stage_conv: List[bool],
        stage_pooling: List[Optional[Tuple[int, int]]],
        origin_stem: bool = True,
        stem_channels: int = 64,
        attn_module: Optional[Callable[[int], layers.Layer]] = None,
        include_top: bool = True,
        num_classes: int = 1000,
        cfg: Optional[Dict[str, Any]] = None,
        input_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        inplanes = stem_channels
        if origin_stem:
            _layers = [
                *conv_sequence(inplanes, "relu", True, kernel_size=7, strides=2, input_shape=input_shape),
                layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),
            ]
        else:
            _layers = [
                *conv_sequence(inplanes // 2, "relu", True, kernel_size=3, input_shape=input_shape),
                *conv_sequence(inplanes, "relu", True, kernel_size=3),
                layers.MaxPool2D(pool_size=2, strides=2, padding="valid"),
            ]

        for n_blocks, out_chan, down, conv, pool in zip(
            num_blocks, output_channels, stage_downsample, stage_conv, stage_pooling
        ):
            _layers.extend(resnet_stage(n_blocks, out_chan, out_chan != inplanes, down))
            if attn_module is not None:
                _layers.append(attn_module(out_chan))
            if conv:
                _layers.extend(conv_sequence(out_chan, activation="relu", bn=True, kernel_size=3))
            if pool:
                _layers.append(layers.MaxPool2D(pool_size=pool, strides=pool, padding="valid"))
            inplanes = out_chan

        if include_top:
            _layers.extend(
                [
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(num_classes),
                ]
            )

        super().__init__(_layers)
        self.cfg = cfg


def _resnet(
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
        num_blocks, output_channels, stage_downsample, stage_conv, stage_pooling, origin_stem, cfg=_cfg, **kwargs
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def resnet18(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet-18 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import resnet18
    >>> model = resnet18(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A classification model
    """

    return _resnet(
        "resnet18",
        pretrained,
        [2, 2, 2, 2],
        [64, 128, 256, 512],
        [False, True, True, True],
        [False] * 4,
        [None] * 4,
        True,
        **kwargs,
    )


def resnet31(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet31 architecture with rectangular pooling windows as described in
    `"Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition",
    <https://arxiv.org/pdf/1811.00751.pdf>`_. Downsizing: (H, W) --> (H/8, W/4)

    >>> import tensorflow as tf
    >>> from doctr.models import resnet31
    >>> model = resnet31(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A classification model
    """

    return _resnet(
        "resnet31",
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


def resnet34(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet-34 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import resnet34
    >>> model = resnet34(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A classification model
    """

    return _resnet(
        "resnet34",
        pretrained,
        [3, 4, 6, 3],
        [64, 128, 256, 512],
        [False, True, True, True],
        [False] * 4,
        [None] * 4,
        True,
        **kwargs,
    )


def resnet50(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet-50 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import resnet50
    >>> model = resnet50(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A classification model
    """

    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs["resnet50"]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs["resnet50"]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs["resnet50"]["classes"])

    _cfg = deepcopy(default_cfgs["resnet50"])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    kwargs.pop("classes")

    model = ResNet50(
        weights=None,
        include_top=True,
        pooling=True,
        input_shape=kwargs["input_shape"],
        classes=kwargs["num_classes"],
        classifier_activation=None,
    )

    model.cfg = _cfg

    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs["resnet50"]["url"])

    return model


def resnet34_wide(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet-34 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_ with twice as many output channels for each stage.

    >>> import tensorflow as tf
    >>> from doctr.models import resnet34_wide
    >>> model = resnet34_wide(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A classification model
    """

    return _resnet(
        "resnet34_wide",
        pretrained,
        [3, 4, 6, 3],
        [128, 256, 512, 1024],
        [False, True, True, True],
        [False] * 4,
        [None] * 4,
        True,
        stem_channels=128,
        **kwargs,
    )
