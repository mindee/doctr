# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Tuple


__all__ = ['VGG16BN']


class VGG16BN(Sequential):
    """VGG-16 architecture as described in `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_., modified by adding batch normalization.

    Args:
        input_shape: shapes of the images

    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (640, 640, 3)
    ) -> None:
        _layers = [
            *self.conv_bn_act(64, 3, padding='same', input_shape=input_size),
            *self.conv_bn_act(64, 3, padding='same'),
            layers.MaxPooling2D((2, 2)),
            *self.conv_bn_act(128, 3, padding='same'),
            *self.conv_bn_act(128, 3, padding='same'),
            layers.MaxPooling2D((2, 2)),
            *self.conv_bn_act(256, 3, padding='same'),
            *self.conv_bn_act(256, 3, padding='same'),
            *self.conv_bn_act(256, 3, padding='same'),
            layers.MaxPooling2D((2, 1)),
            *self.conv_bn_act(512, 3, padding='same'),
            *self.conv_bn_act(512, 3, padding='same'),
            *self.conv_bn_act(512, 3, padding='same'),
            layers.MaxPooling2D((2, 1)),
            *self.conv_bn_act(512, 3, padding='same'),
            *self.conv_bn_act(512, 3, padding='same'),
            *self.conv_bn_act(512, 3, padding='same'),
            layers.MaxPooling2D((2, 1)),
        ]
        super().__init__(_layers)

    @staticmethod
    def conv_bn_act(output_channels, kernel_size, **kwargs):
        return [
            layers.Conv2D(output_channels, kernel_size, **kwargs),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ]
