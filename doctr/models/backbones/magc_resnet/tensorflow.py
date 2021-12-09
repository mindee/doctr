# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import math
from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from ...utils import conv_sequence, load_pretrained_params
from ..resnet import ResnetStage

__all__ = ['MAGCResNet', 'magc_resnet31']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'magc_resnet31': {
        'num_blocks': (1, 2, 5, 3),
        'output_channels': (256, 512, 512, 512),
        'url': None
    },
}


class MAGC(layers.Layer):
    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
        inplanes: input channels
        headers: number of headers to split channels
        att_scale: if True, re-scale attention to counteract the variance distibutions
        ratio: bottleneck ratio
        **kwargs
    """

    def __init__(
        self,
        inplanes: int,
        headers: int = 8,
        att_scale: bool = False,
        ratio: float = 0.0625,  # bottleneck ratio of 1/16 as described in paper
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.headers = headers  # h
        self.inplanes = inplanes  # C
        self.att_scale = att_scale
        self.planes = int(inplanes * ratio)

        self.single_header_inplanes = int(inplanes / headers)  # C / h

        self.conv_mask = layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_initializer=tf.initializers.he_normal()
        )

        self.transform = Sequential(
            [
                layers.Conv2D(
                    filters=self.planes,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal()
                ),
                layers.LayerNormalization([1, 2, 3]),
                layers.ReLU(),
                layers.Conv2D(
                    filters=self.inplanes,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal()
                ),
            ],
            name='transform'
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
        if self.att_scale and self.headers > 1:
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


class MAGCResNet(Sequential):

    """Implements the modified resnet with MAGC layers, as described in paper.

    Args:
        num_blocks: number of residual blocks in each stage
        output_channels: number of output channels in each stage
        headers: number of header to split channels in MAGC layers
        input_shape: shape of the model input (without batch dim)
    """

    def __init__(
        self,
        num_blocks: Tuple[int, int, int, int],
        output_channels: Tuple[int, int, int, int],
        headers: int = 8,
        input_shape: Tuple[int, int, int] = (32, 128, 3),
    ) -> None:
        _layers = [
            # conv_1x
            *conv_sequence(out_channels=64, activation='relu', bn=True, kernel_size=3, input_shape=input_shape),
            *conv_sequence(out_channels=128, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # Stage 1
            ResnetStage(num_blocks[0], output_channels[0]),
            MAGC(output_channels[0], headers=headers, att_scale=True),
            *conv_sequence(out_channels=output_channels[0], activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # Stage 2
            ResnetStage(num_blocks[1], output_channels[1]),
            MAGC(output_channels[1], headers=headers, att_scale=True),
            *conv_sequence(out_channels=output_channels[1], activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 1), (2, 1)),
            # Stage 3
            ResnetStage(num_blocks[2], output_channels[2]),
            MAGC(output_channels[2], headers=headers, att_scale=True),
            *conv_sequence(out_channels=output_channels[2], activation='relu', bn=True, kernel_size=3),
            # Stage 4
            ResnetStage(num_blocks[3], output_channels[3]),
            MAGC(output_channels[3], headers=headers, att_scale=True),
            *conv_sequence(out_channels=output_channels[3], activation='relu', bn=True, kernel_size=3),
        ]
        super().__init__(_layers)


def _magc_resnet(arch: str, pretrained: bool, **kwargs: Any) -> MAGCResNet:

    # Build the model
    model = MAGCResNet(
        default_cfgs[arch]['num_blocks'],
        default_cfgs[arch]['output_channels'],
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def magc_resnet31(pretrained: bool = False, **kwargs: Any) -> MAGCResNet:
    """Resnet31 architecture with Multi-Aspect Global Context Attention as described in
    `"MASTER: Multi-Aspect Non-local Network for Scene Text Recognition",
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import magc_resnet31
        >>> model = magc_resnet31(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 224, 224, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A feature extractor model
    """

    return _magc_resnet('magc_resnet31', pretrained, **kwargs)
