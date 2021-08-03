# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Optional, Tuple, Any, Dict, List
from ...utils import conv_sequence, load_pretrained_params


__all__ = ["MobileNetV3", "mobilenetv3_small", "mobilenetv3_large"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    'mobilenetv3_large': {
        'input_shape': (512, 512),
        'out_chans': [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160],
        'kernels': [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5],
        'exp_chans': [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960],
        'strides': [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
        'use_squeeze': [False, False, False, True, True, True, False, False, False, False,
                        True, True, True, True, True],
        'use_swish': [False, False, False, False, False, False, True, True, True, True, True, True, True, True, True],
        'url': None
    },
    'mobilenetv3_small': {
        'input_shape': (512, 512),
        'out_chans': [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
        'kernels': [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
        'exp_chans': [16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576],
        'strides': [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
        'use_squeeze': [True, False, False, True, True, True, True, True, True, True, True],
        'use_swish': [False, False, False, True, True, True, True, True, True, True, True],
        'url': None
    }
}


def hard_swish(x: tf.Tensor) -> tf.Tensor:
    return x * tf.nn.relu6(x + 3.) / 6.0


class SqueezeExcitation(layers.Layer):
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


class InvertedResidual(layers.Layer):

    """InvertedResidual for mobilenet

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

        _layers = [*conv_sequence(exp_chan, activation=hard_swish if use_swish else tf.nn.relu6, kernel_size=1)]

        _layers.extend([
            layers.DepthwiseConv2D(kernel, strides, depth_multiplier=1, padding='same'),
            layers.BatchNormalization(),
        ])

        _layers.append(layers.Activation(hard_swish) if use_swish else layers.Activation(tf.nn.relu6))

        if use_squeeze:
            _layers.append(SqueezeExcitation(exp_chan))

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


class MobileNetV3(Sequential):

    """Implements MobileNetV3, inspired from both:
    <https://github.com/xiaochus/MobileNetV3/tree/master/model>`_.
    and <https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html>`_.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        out_chans: List[int],
        kernels: List[int],
        exp_chans: List[int],
        strides: List[int],
        use_squeeze: List[bool],
        use_swish: List[bool],
        num_classes: Optional[int] = None,
        include_top: bool = False,

    ) -> None:

        _layers = [
            *conv_sequence(16, strides=2, activation=hard_swish, kernel_size=3, input_shape=(*input_shape, 3))
        ]

        for out, k, exp, s, use_sq, use_sw in zip(out_chans, kernels, exp_chans, strides, use_squeeze, use_swish):
            _layers.append(
                InvertedResidual(out, k, exp, s, use_sq, use_sw),
            )

        _layers.extend(
            [
                *conv_sequence(exp_chans[-1], strides=1, activation=hard_swish, kernel_size=1),
                layers.GlobalAveragePooling2D(),
                layers.Reshape((1, 1, exp_chans[-1])),
                layers.Conv2D(1280, 1, padding='same'),
                layers.Activation(hard_swish)
            ]
        )

        if include_top:
            _layers.append([
                layers.Conv2D(num_classes, 1, padding='same', activation='softmax'),
            ])

        super().__init__(_layers)


def _mobilenetv3(arch: str, pretrained: bool) -> MobileNetV3:

    # Build the model
    model = MobileNetV3(
        default_cfgs[arch]['input_shape'],
        default_cfgs[arch]['out_chans'],
        default_cfgs[arch]['kernels'],
        default_cfgs[arch]['exp_chans'],
        default_cfgs[arch]['strides'],
        default_cfgs[arch]['use_squeeze'],
        default_cfgs[arch]['use_swish'],
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def mobilenetv3_small(pretrained: bool = False) -> MobileNetV3:
    """MobileNetV3 architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import mobilenetv3_large
        >>> model = mobilenetv3_small(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A  mobilenetv3_small model
    """
    return _mobilenetv3('mobilenetv3_small', pretrained)


def mobilenetv3_large(pretrained: bool = False) -> MobileNetV3:
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
    return _mobilenetv3('mobilenetv3_large', pretrained)
