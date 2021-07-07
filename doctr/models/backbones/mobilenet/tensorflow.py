# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
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
        filters: int,
        kernel: int,
        e: int,
        s: int,
        squeeze: bool,
        nl: str,
        alpha: float,
    ) -> None:

        cchannel = int(alpha * filters)
        self.filters = filters
        _layers = [
            *conv_sequence(e, activation=nl, kernel_size=1),
            layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same'),
            layers.BatchNormalization(),
            activation(nl),
        ]
        if squeeze:
            _layers.append(Squeeze(chan=e))
        _layers.append(
            [
                layers.Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same'),
                layers.BatchNormalization(),

            ]
        )
        self.bottleneck_sequence = Sequential(_layers)

    def call(
        self,
        inputs: tf.Tensor
    ) -> tf.Tensor:

        input_shape = inputs.shape
        r = (s == 1 and input_shape[3] == self.filters)
        x = self.bottleneck_sequence(inputs)
        if r:
            x = tf.add(x, inputs)

        return x
