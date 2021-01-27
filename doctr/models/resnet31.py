# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Tuple

__all__ = ['Resnet31']


class ResnetBlock(layers.Layer):
    """Implements a resnet31 block with shortcut

    Args:
        conv_shortcut: Use of shortcut
        output_channels: number of channels to use in Conv2D
        kernel_size: size of square kernels

    """
    def init(
        self,
        conv_shortcut: bool,
        output_channels: int,
        kernel_size: int
    ) -> None:

        if conv_shortcut:
            self.shortcut = Sequential(
                [
                    layers.Conv2D(filters=n, kernel_size=1),
                    layers.BatchNormalization()
                ]
            )
        else:
            self.shortcut = layers.Lambda(lambda x: x)
        self.conv_block = Sequential(
            self.conv_resnetblock(output_channels, kernel_size)
        )
        self.act = layers.Activation('relu')

    @staticmethod
    def conv_resnetblock(output_channels, kernel_size, **kwargs):
        return [
            layers.Conv2D(output_channels, kernel_size),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(output_channels, kernel_size),
            layers.BatchNormalization(),
        ]

    def __call__(
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

        final_blocks = [
            ResnetBlock(
                conv_shortcut=False, output_channels=output_channels, kernel_size=3
            ) for _ in range(1, num_layers)
        ]
        super.__init__(
            [
                ResnetBlock(conv_shortcut=True, output_channels=output_channels, kernel_size=3),
                *final_blocks,
            ]
        )


class Resnet31(Sequential):
    """Resnet31 architecture with rectangular pooling windows as described in
    `"Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition",
    <https://arxiv.org/pdf/1811.00751.pdf>`_. Downsizing: (H, W) --> (H/8, W/2)

    Args:
        input_size: size of the images

    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (640, 640, 3)
    ) -> None:

        _layers = [
            conv_bn_act(output_channels=64, kernel_size=3, input_shape=input_size),
            conv_bn_act_pool(output_channels=128, kernel_size=3, p_size=2),
            ResnetStage(num_blocks=1, output_channels=256),
            conv_bn_act_pool(output_channels=256, kernel_size=3, p_size=2),
            ResnetStage(num_blocks=2, output_channels=256),
            conv_bn_act_pool(output_channels=256, kernel_size=3, p_size=(2, 1)),
            ResnetStage(num_blocks=5, output_channels=512),
            conv_bn_act(output_channels=512, kernel_size=3),
            ResnetStage(num_blocks=3, output_channels=512),
            conv_bn_act(output_channels=512, kernel_size=3),
        ]
        super.__init__(_layers)

    @staticmethod
    def conv_bn_act(output_channels, kernel_size, **kwargs):
        return [
            layers.Conv2D(output_channels, kernel_size, **kwargs),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ]

    @staticmethod
    def conv_bn_act_pool(output_channels, kernel_size, p_size, **kwargs):
        return [
            layers.Conv2D(output_channels, kernel_size, **kwargs),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=p_size, strides=p_size, padding='valid'),
        ]
