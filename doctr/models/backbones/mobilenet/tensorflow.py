# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Optional, Tuple
from ...utils import conv_sequence


def hard_swish(x: tf.Tensor) -> tf.Tensor:
    return x * tf.nn.relu6(x + 3.) / 6.0


class Squeeze(layers.Layer):
    """Squeeze and Excitation.
    """
    def __init__(self, chan: int) -> None:
        super().__init__()
        self.chan = chan
        self.squeeze_seq = Sequential(
            [
                layers.GlobalAveragePooling2D(),
                layers.Dense(chan, activation='relu'),
                layers.Dense(chan, activation='hard_sigmoid'),
            ]
        )

    def call(self,inputs: tf.Tensor) -> tf.Tensor:
        x = self.squeeze_seq(inputs)
        x = tf.reshape(x, (1, 1, self.chan))
        x = tf.math.multiply(inputs, x)
        return x


class Bottleneck(layers.Layer):
    
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        e: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        squeeze: Boolean, Whether to use the squeeze.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """
    def __init__(
        self,
        out_chan: int,
        kernel: int,
        exp_chan: int,
        strides: int,
        use_squeeze: bool,
        use_swish: bool,
    ) -> None:

        self.out_chan = out_chan
        self.strides = strides
        if use_swish:
            _layers = [*conv_sequence(exp_chan, activation=hard_swish, kernel_size=1)]
        else:
            _layers = [*conv_sequence(exp_chan, activation=tf.nn.relu6, kernel_size=1)]

        _layers.append([
            layers.DepthwiseConv2D(kernel, strides, depth_multiplier=1, padding='same'),
            layers.BatchNormalization(),
        ])

        if use_swish:
            _layers.append(layers.Activation(hard_swish))
        else:
            _layers.append(layers.Activation(tf.nn.relu6))

        if use_squeeze:
            _layers.append(Squeeze(exp_chan))

        _layers.append(
            [
                layers.Conv2D(out_chan, 1, strides=(1, 1), padding='same'),
                layers.BatchNormalization(),

            ]
        )
        self.bottleneck_sequence = Sequential(_layers)

    def call(
        self,
        inputs: tf.Tensor
    ) -> tf.Tensor:

        in_chan = inputs.shape[3]
        use_residual = (self.strides == 1 and in_chan == self.out_chan)
        x = self.bottleneck_sequence(inputs)
        if use_residual:
            x = tf.add(x, inputs)

        return x


class MobileNetV3_Large(Sequential):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: Optional[int] = None,
        include_top: bool = True
    ):
        """Init.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.
        # Returns
            MobileNetv3 model.
        """
        _layers = [
            *conv_sequence(16, strides=2, activation=hard_swish, kernel_size=3, input_shape=(*input_shape, 3)),
            Bottleneck(16, 3, 16, 1, use_squeeze=False, use_swish=False),
            Bottleneck(24, 3, 64, 2, use_squeeze=False, use_swish=False),
            Bottleneck(24, 3, 72, 1, use_squeeze=False, use_swish=False),
            Bottleneck(40, 5, 72, 2, use_squeeze=True, use_swish=False),
            Bottleneck(40, 5, 120, 1, use_squeeze=True, use_swish=False),
            Bottleneck(40, 5, 120, 1, use_squeeze=True, use_swish=False),
            Bottleneck(80, 3, 240, 2, use_squeeze=False, use_swish=True),
            Bottleneck(80, 3, 200, 1, use_squeeze=False, use_swish=True),
            Bottleneck(80, 3, 184, 1, use_squeeze=False, use_swish=True),
            Bottleneck(80, 3, 184, 1, use_squeeze=False, use_swish=True),
            Bottleneck(112, 3, 480, 1, use_squeeze=True, use_swish=True),
            Bottleneck(112, 3, 672, 1, use_squeeze=True, use_swish=True),
            Bottleneck(160, 5, 672, 2, use_squeeze=True, use_swish=True),
            Bottleneck(160, 5, 960, 1, use_squeeze=True, use_swish=True),
            Bottleneck(160, 5, 960, 1, use_squeeze=True, use_swish=True),
            *conv_sequence(960, strides=1, activation=hard_swish, kernel_size=1),
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, 960)),
            layers.Conv2D(1280, 1, padding='same'),
            layers.Activation(hard_swish)
        ]

        if include_top:
            _layers.append([
                layers.Conv2D(num_classes, 1, padding='same', activation='softmax'),
            ])

        super().__init__(_layers)
