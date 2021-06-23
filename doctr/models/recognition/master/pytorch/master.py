# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torch import nn
from torch.nn import functional as F
from ...datasets import VOCABS

__all__ = ['MASTER', 'MASTERPostProcessor', 'master']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'master': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'input_shape': (3, 48, 160),
        'post_processor': 'MASTERPostProcessor',
        'vocab': VOCABS['french'],
        'url': None,
    },
}


class MAGC(layers.Layer):

    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
        inplanes: input channels
        headers: number of headers to split channels
        att_scale: if True, re-scale attention to counteract the variance distibutions
        **kwargs
    """

    def __init__(
        self,
        inplanes: int,
        headers: int = 1,
        att_scale: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.headers = headers  # h
        self.inplanes = inplanes  # C
        self.att_scale = att_scale

        self.single_header_inplanes = int(inplanes / headers)  # C / h

        self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
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
        # B*h, H, W, 1,
        context_mask = self.conv_mask(x)
        # B*h, 1, H*W, 1
        context_mask = tf.reshape(context_mask, shape=(b * self.headers, 1, h * w, 1))
        # scale variance
        if self.att_scale and self.headers > 1:
            context_mask = context_mask / tf.sqrt(self.single_header_inplanes)
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


decoder_layer = nn.TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    layer_norm_eps=1e-05,
    batch_first=False,
)

decoder = nn.TransformerDecoder(
    decoder_layer,
    num_layers=3,
    norm=None
)