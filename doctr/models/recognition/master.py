# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, List

from .core import RecognitionModel
from ..backbones.resnet import ResnetStage
from ..utils import conv_sequence
from .transformer import Decoder, positional_encoding, create_look_ahead_mask, create_padding_mask

__all__ = ['MASTER']


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
        super().__init__(name='MAGC', **kwargs)

        self.headers = headers  # h
        self.inplanes = inplanes  # C
        self.att_scale = att_scale

        self.single_header_inplanes = int(inplanes / headers)  # C / h

        self.conv_mask = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_initializer=tf.initializers.he_normal()
        )

        self.transform = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.inplanes,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal()
                ),
                tf.keras.layers.LayerNormalization([1, 2, 3]),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=self.inplanes,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal()
                ),
            ],
            name='transform'
        )

    @tf.function
    def context_modeling(self, inputs: tf.Tensor) -> tf.Tensor:
        B, H, W, C = tf.shape(inputs)

        # B, H, W, h, C/h
        x = tf.reshape(inputs, shape=(B, H, W, self.headers, self.single_header_inplanes))
        # B, h, H, W, C/h
        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        # B*h, H, W, C/h
        x = tf.reshape(x, shape=(B * self.headers, H, W, self.single_header_inplanes))
        input_x = x
        # B*h, 1, H*W, C/h
        input_x = tf.reshape(input_x, shape=(B * self.headers, 1, H * W, self.single_header_inplanes))
        # B*h, 1, C/h, H*W
        input_x = tf.transpose(input_x, perm=[0, 1, 3, 2])
        # B*h, H, W, 1,
        context_mask = self.conv_mask(x)
        # B*h, 1, H*W, 1
        context_mask = tf.reshape(context_mask, shape=(B * self.headers, 1, H * W, 1))
        # scale variance
        if self.att_scale and self.headers > 1:
            context_mask = context_mask / tf.sqrt(self.single_header_inplanes)
        # B*h, 1, H*W, 1
        context_mask = tf.keras.activations.softmax(context_mask, axis=2)
        # B*h, 1, C/h, 1
        context = tf.matmul(input_x, context_mask)
        context = tf.reshape(context, shape=(B, 1, C, 1))
        # B, 1, 1, C
        context = tf.transpose(context, perm=(0, 1, 3, 2))
        return context

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Context modeling: B, H, W, C  ->  B, 1, 1, C
        context = self.context_modeling(inputs)

        # Transform: B, 1, 1, C  ->  B, 1, 1, C
        transformed = self.transform(context)
        return inputs + transformed


class MAGCResnet(layers.Sequential):

    def __init__(
        self,
        headers: int = 1,
        input_shape: Tuple[int, int, int] = (48, 160, 3),
    ) -> None:
        _layers = [
            # conv_1x
            conv_sequence(out_channels=64, activation='relu', bn=True, kernel_size=3, input_shape=input_shape),
            conv_sequence(out_channels=128, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # conv_2x
            ResnetStage(num_blocks=1, output_channels=256),
            MAGC(inplanes=256, headers=headers, att_scale=True),
            conv_sequence(out_channels=256, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # conv_3x
            ResnetStage(num_blocks=2, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 1), (2, 1)),
            # conv_4x
            ResnetStage(num_blocks=5, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
            # conv_5x
            ResnetStage(num_blocks=3, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
        ]
        super().__init__(_layers)


class MASTER(RecognitionModel):
    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official TF implementation: <https://github.com/jiangxiluning/MASTER-TF>`_.

    Args:
        vocab_size: size of the vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        headers: headers for the MAGC module
        dff: depth of the pointwise feed-forward layer
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        input_shape: size of the image inputs

    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        headers: int,
        dff: int,
        num_layers: int = 3,
        max_length: int = 50,
        input_shape: tuple = (48, 160, 3),
    ) -> None:
        super(MASTER, self).__init__(name='Master')

        self.input_shape = input_shape
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.feature_extractor = MAGCResnet(headers=headers, input_shape=input_shape)
        self.seq_embedding = layers.Embedding(vocab_size, d_model)

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=headers,
            dff=dff,
            target_vocab_size=vocab_size,
            maximum_position_encoding=max_length,
        )
        self.feature_pe = positional_encoding(input_shape[0] * input_shape[1], d_model)
        self.linear = layers.Dense(vocab_size, kernel_initializer=tf.initializers.he_uniform())

    @tf.function
    def make_mask(self, target: List[int]) -> tf.Tensor:
        look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
        target_padding_mask = create_padding_mask(target, self.vocab_size + 2)  # TODO: define padding symbol in fn
        combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)
        return combined_mask

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        image: tf.Tensor = inputs[0]
        transcript: tf.Tensor = inputs[1]

        feature = self.feature_extractor(image, **kwargs)

        B, H, W, C = tf.shape(feature)

        feature = tf.reshape(feature, shape=(B, H * W, C))
        memory = feature + self.feature_pe[:, :H * W, :]

        tgt_mask = self.make_mask(transcript[:, :-1])

        output, _ = self.decoder(transcript, memory, tgt_mask, None, training=True)
        logits = self.linear(output)

        return logits

    @tf.function
    def decode(
        self,
        image: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        feature = self.feature_extractor(image, training=False)
        B, H, W, C = tf.shape(feature)

        feature = tf.reshape(feature, shape=(B, H * W, C))
        memory = feature + self.feature_pe[:, :H * W, :]

        max_len = tf.constant(self.max_length, dtype=tf.int32)
        start_symbol = tf.constant(self.vocab_size + 1, dtype=tf.int32)  # SOS (EOS = vocab_size)
        padding_symbol = tf.constant(self.vocab_size + 2, dtype=tf.int32)  # PAD

        ys = tf.fill(dims=(B, max_len - 1), value=padding_symbol)
        start_vector = tf.fill(dims=(B, 1), value=start_symbol)
        ys = tf.concat([start_vector, ys], axis=-1)

        final_logits = tf.zeros(shape=(B, max_len - 1, self.vocab_size), dtype=tf.float32)
        # max_len = len + 2
        for i in range(max_len - 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(final_logits, tf.TensorShape([None, None, self.vocab_size]))]
            )
            ys_mask = self.make_mask(ys)
            #output, _ = self.decoder(ys, memory, False, ys_mask, None)
            output = self.decoder(self.seq_embedding(ys), memory, None, ys_mask, training=False)
            logits = self.linear(output)
            prob = tf.nn.softmax(logits, axis=-1)
            next_word = tf.argmax(prob, axis=-1, output_type=ys.dtype)

            # ys.shape = B, T
            i_mesh, j_mesh = tf.meshgrid(tf.range(B), tf.range(max_len), indexing='ij')
            indices = tf.stack([i_mesh[:, i + 1], j_mesh[:, i + 1]], axis=1)

            ys = tf.tensor_scatter_nd_update(ys, indices, next_word[:, i + 1])

            if i == (max_len - 2):
                final_logits = logits

        return ys, final_logits[:, 1:]
