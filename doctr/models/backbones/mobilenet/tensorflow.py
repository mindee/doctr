# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Optional, Tuple, Any, Dict
from ...utils import conv_sequence, load_pretrained_params


__all__ = ["MobileNetV3_Large", "MobileNetV3_Small", "mobilenetv3_large", "mobilenetv3_small"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    'mobilenetv3_large': {
        'input_shape': (512, 512),
        'url': None
    },
    'mobilenetv3_small': {
        'input_shape': (512, 512),
        'url': None
    }
}


def hard_swish(x: tf.Tensor) -> tf.Tensor:
    return x * tf.nn.relu6(x + 3.) / 6.0


class Squeeze(layers.Layer):
    """Squeeze and Excitation.
    """
    def __init__(self, chan: int) -> None:
        super().__init__()
        self.squeeze_seq = Sequential(
            [
                layers.GlobalAveragePooling2D(),
                layers.Dense(chan, activation='relu'),
                layers.Dense(chan, activation='hard_sigmoid'),
                layers.Reshape((1, 1, chan))
            ]
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.squeeze_seq(inputs)
        x = tf.math.multiply(inputs, x)
        return x


class Bottleneck(layers.Layer):

    """Bottleneck for mobilenet

    Args:
        out_chan: the dimensionality of the output space.
        kernel: kernel size for depthwise conv
        exp_chan: expanded channels, used in squeeze and first conv
        strides: strides in depthwise conv
        use_squeeze: whether to use the squeeze & sequence module
        use_swish: activation type, relu6 or hard_swish

    Returns:
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
        super().__init__()
        self.out_chan = out_chan
        self.strides = strides
        if use_swish:
            _layers = [*conv_sequence(exp_chan, activation=hard_swish, kernel_size=1)]
        else:
            _layers = [*conv_sequence(exp_chan, activation=tf.nn.relu6, kernel_size=1)]

        _layers.extend([
            layers.DepthwiseConv2D(kernel, strides, depth_multiplier=1, padding='same'),
            layers.BatchNormalization(),
        ])

        if use_swish:
            _layers.append(layers.Activation(hard_swish))
        else:
            _layers.append(layers.Activation(tf.nn.relu6))

        if use_squeeze:
            _layers.append(Squeeze(exp_chan))

        _layers.extend(
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

    """Implements large version of MobileNetV3, inspired from both:
    <https://github.com/xiaochus/MobileNetV3/tree/master/model>`_.
    and <https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html>`_.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: Optional[int] = None,
        include_top: bool = False,
    ) -> None:

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


class MobileNetV3_Small(Sequential):

    """Implements large version of MobileNetV3, inspired from both:
    <https://github.com/xiaochus/MobileNetV3/tree/master/model>`_.
    and <https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html>`_.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: Optional[int] = None,
        include_top: bool = False,
    ) -> None:

        _layers = [
            *conv_sequence(16, strides=2, activation=hard_swish, kernel_size=3, input_shape=(*input_shape, 3)),
            Bottleneck(16, 3, 16, 2, use_squeeze=True, use_swish=False),
            Bottleneck(24, 3, 72, 2, use_squeeze=False, use_swish=False),
            Bottleneck(24, 3, 88, 1, use_squeeze=False, use_swish=False),
            Bottleneck(40, 5, 96, 2, use_squeeze=True, use_swish=True),
            Bottleneck(40, 5, 240, 1, use_squeeze=True, use_swish=True),
            Bottleneck(40, 5, 240, 1, use_squeeze=True, use_swish=True),
            Bottleneck(48, 5, 120, 1, use_squeeze=True, use_swish=True),
            Bottleneck(48, 5, 144, 1, use_squeeze=True, use_swish=True),
            Bottleneck(96, 5, 288, 2, use_squeeze=True, use_swish=True),
            Bottleneck(96, 5, 576, 1, use_squeeze=True, use_swish=True),
            Bottleneck(96, 5, 576, 1, use_squeeze=True, use_swish=True),
            *conv_sequence(576, strides=1, activation=hard_swish, kernel_size=1),
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, 576)),
            layers.Conv2D(1280, 1, padding='same'),
            layers.Activation(hard_swish)
        ]

        if include_top:
            _layers.append([
                layers.Conv2D(num_classes, 1, padding='same', activation='softmax'),
            ])

        super().__init__(_layers)


def _mobilenetv3_large(arch: str, pretrained: bool) -> MobileNetV3_Large:

    # Build the model
    model = MobileNetV3_Large(
        default_cfgs[arch]['input_shape'],
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def _mobilenetv3_small(arch: str, pretrained: bool) -> MobileNetV3_Small:

    # Build the model
    model = MobileNetV3_Small(
        default_cfgs[arch]['input_shape'],
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def mobilenetv3_large(pretrained: bool = False) -> MobileNetV3_Large:
    """MobileNetV3 architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import mobilenetv3_large
        >>> model = mobilenetv3_large(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A  mobilenetv3_large model
    """

    return _mobilenetv3_large('mobilenetv3_large', pretrained)


def mobilenetv3_small(pretrained: bool = False) -> MobileNetV3_Small:
    """MobileNetV3 architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import mobilenetv3_small
        >>> model = mobilenetv3_small(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A  mobilenetv3_small model
    """

    return _mobilenetv3_small('mobilenetv3_small', pretrained)
