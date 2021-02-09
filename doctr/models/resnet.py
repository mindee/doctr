# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Tuple
from utils import conv_sequence

__all__ = ['Resnet31']


class ResnetBlock(layers.Layer):

    """Implements a resnet31 block with shortcut

    Args:
        conv_shortcut: Use of shortcut
        output_channels: number of channels to use in Conv2D
        kernel_size: size of square kernels
    """
    def __init__(
        self,
        output_channels: int,
        conv_shortcut: bool,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        if conv_shortcut:
            self.shortcut = Sequential(
                [
                    layers.Conv2D(
                        filters=output_channels,
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
            self.conv_resnetblock(output_channels, 3)
        )
        self.act = layers.Activation('relu')

    @staticmethod
    def conv_resnetblock(
        output_channels: int,
        kernel_size: int
    ) -> list[layers.Layer]:
        return [
            *conv_sequence(output_channels, activation='relu', bn=True, kernel_size=kernel_size)
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
    """
    def __init__(
        self,
        num_blocks: int,
        output_channels: int
    ) -> None:

        super().__init__()
        final_blocks = [
            ResnetBlock(output_channels, conv_shortcut=False) for _ in range(1, num_blocks)
        ]
        self.add(ResnetBlock(output_channels, conv_shortcut=True))
        for final_block in final_blocks:
            self.add(final_block)


class Resnet(Sequential):

    """Resnet31 architecture with rectangular pooling windows as described in
    `"Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition",
    <https://arxiv.org/pdf/1811.00751.pdf>`_. Downsizing: (H, W) --> (H/8, W/4)

    Args:
        input_size: size of the images
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (640, 640, 3),
        num_blocks: Tuple[int, int, int, int],
        output_channels: Tuple[int, int, int, int],
        conv_seq: Tuple[bool, bool, bool, bool],
        pooling: Tuple[
            Optional[Tuple[int, int]],
            Optional[Tuple[int, int]],
            Optional[Tuple[int, int]],
            Optional[Tuple[int, int]]
        ],

    ) -> None:

        _layers = [
            *conv_sequence(output_channels=64, activation='relu', bn=True, kernel_size=3, input_shape=input_size),
            *conv_sequence(output_channels=128, activation='relu', bn=True, kernel_size=3),
            layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        ]
        for n_blocks, out_channels, conv, pool in zip(num_blocks, output_channels, conv_seq, pooling):
            _layers.append(ResnetStage(n_blocks, out_channels))
            if conv:
                _layers.append(*conv_sequence(out_channels, activation='relu', bn=True, kernel_size=3))
            if pool:
                _layers.append(layers.MaxPool2D(pool_size=pool, strides=pool, padding='valid'))
        super().__init__(_layers)


default_cfgs: Dict[str, Dict[str, Any]] = {
    'resnet31': {'num_blocks': (1, 2, 5, 3), 'output_channels': (256, 256, 512, 512),
                 'conv_seq': (True, True, True, True), 'pooling': ((2, 2), (2, 1), None, None)
                 'url': None},
}