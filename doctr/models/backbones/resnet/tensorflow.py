# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Tuple, Dict, Optional, Any, List
from ...utils import conv_sequence, load_pretrained_params

__all__ = ['ResNet', 'resnet31', 'ResnetStage']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'resnet31': {'num_blocks': (1, 2, 5, 3), 'output_channels': (256, 256, 512, 512),
                 'conv_seq': (True, True, True, True), 'pooling': ((2, 2), (2, 1), None, None),
                 'url': None},
}


class ResnetBlock(layers.Layer):

    """Implements a resnet31 block with shortcut

    Args:
        conv_shortcut: Use of shortcut
        output_channels: number of channels to use in Conv2D
        kernel_size: size of square kernels
        strides: strides to use in the first convolution of the block
    """
    def __init__(
        self,
        output_channels: int,
        conv_shortcut: bool,
        strides: int = 1,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        if conv_shortcut:
            self.shortcut = Sequential(
                [
                    layers.Conv2D(
                        filters=output_channels,
                        strides=strides,
                        padding='same',
                        kernel_size=1,
                        use_bias=False,
                        kernel_initializer='he_normal'
                    ),
                    layers.BatchNormalization()
                ]
            )
        else:
            self.shortcut = layers.Lambda(lambda x: x)
        self.conv_block = Sequential(
            self.conv_resnetblock(output_channels, 3, strides)
        )
        self.act = layers.Activation('relu')

    @staticmethod
    def conv_resnetblock(
        output_channels: int,
        kernel_size: int,
        strides: int = 1,
    ) -> List[layers.Layer]:
        return [
            *conv_sequence(output_channels, activation='relu', bn=True, strides=strides, kernel_size=kernel_size),
            layers.Conv2D(output_channels, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
        ]

    def call(
        self,
        inputs: tf.Tensor
    ) -> tf.Tensor:
        clone = self.shortcut(inputs)
        conv_out = self.conv_block(inputs)
        out = self.act(clone + conv_out)

        return out


class ResnetStage(Sequential):

    """Implements a resnet31 stage

    Args:
        num_blocks: number of blocks inside the stage
        output_channels: number of channels to use in Conv2D
        downsample: if true, performs a /2 downsampling at the first block of the stage
    """
    def __init__(
        self,
        num_blocks: int,
        output_channels: int,
        downsample: bool = False,
    ) -> None:

        super().__init__()
        final_blocks = [
            ResnetBlock(output_channels, conv_shortcut=False) for _ in range(1, num_blocks)
        ]
        if downsample is True:
            self.add(ResnetBlock(output_channels, conv_shortcut=True, strides=2))
        else:
            self.add(ResnetBlock(output_channels, conv_shortcut=True))
        for final_block in final_blocks:
            self.add(final_block)


class ResNet(Sequential):

    """Resnet class with two convolutions and a maxpooling before the first stage

    Args:
        num_blocks: number of resnet block in each stage
        output_channels: number of channels in each stage
        conv_seq: wether to add a conv_sequence after each stage
        pooling: pooling to add after each stage (if None, no pooling)
        input_shape: shape of inputs
        include_top: whether the classifier head should be instantiated
    """

    def __init__(
        self,
        num_blocks: Tuple[int, int, int, int],
        output_channels: Tuple[int, int, int, int],
        conv_seq: Tuple[bool, bool, bool, bool],
        pooling: Tuple[
            Optional[Tuple[int, int]],
            Optional[Tuple[int, int]],
            Optional[Tuple[int, int]],
            Optional[Tuple[int, int]]
        ],
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        include_top: bool = False,
    ) -> None:

        _layers = [
            *conv_sequence(out_channels=64, activation='relu', bn=True, kernel_size=3, input_shape=input_shape),
            *conv_sequence(out_channels=128, activation='relu', bn=True, kernel_size=3),
            layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        ]
        for n_blocks, out_channels, conv, pool in zip(num_blocks, output_channels, conv_seq, pooling):
            _layers.append(ResnetStage(n_blocks, out_channels))
            if conv:
                _layers.extend(conv_sequence(out_channels, activation='relu', bn=True, kernel_size=3))
            if pool:
                _layers.append(layers.MaxPool2D(pool_size=pool, strides=pool, padding='valid'))
        super().__init__(_layers)


def _resnet(arch: str, pretrained: bool, **kwargs: Any) -> ResNet:

    # Build the model
    model = ResNet(
        default_cfgs[arch]['num_blocks'],
        default_cfgs[arch]['output_channels'],
        default_cfgs[arch]['conv_seq'],
        default_cfgs[arch]['pooling'],
        **kwargs
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def resnet31(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet31 architecture with rectangular pooling windows as described in
    `"Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition",
    <https://arxiv.org/pdf/1811.00751.pdf>`_. Downsizing: (H, W) --> (H/8, W/4)

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import resnet31
        >>> model = resnet31(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 224, 224, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A resnet31 model
    """

    return _resnet('resnet31', pretrained, **kwargs)
