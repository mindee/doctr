# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from typing import Tuple

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
        super().__init__(**kwargs)

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


class MAGCResnet(Sequential):

    """Implements the modified resnet with MAGC layers, as described in paper.

    Args:
        headers: number of header to split channels in MAGC layers
        input_shape: shape of the model input (without batch dim)
    """

    def __init__(
        self,
        headers: int = 1,
        input_shape: Tuple[int, int, int] = (48, 160, 3),
    ) -> None:
        _layers = [
            # conv_1x
            *conv_sequence(out_channels=64, activation='relu', bn=True, kernel_size=3, input_shape=input_shape),
            *conv_sequence(out_channels=128, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # conv_2x
            ResnetStage(num_blocks=1, output_channels=256),
            MAGC(inplanes=256, headers=headers, att_scale=True),
            *conv_sequence(out_channels=256, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # conv_3x
            ResnetStage(num_blocks=2, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 1), (2, 1)),
            # conv_4x
            ResnetStage(num_blocks=5, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
            # conv_5x
            ResnetStage(num_blocks=3, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
        ]
        super().__init__(_layers)


class MASTER(RecognitionModel):

    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official TF implementation: <https://github.com/jiangxiluning/MASTER-TF>`_.

    Args:
        vocab: vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        headers: headers for the MAGC module
        dff: depth of the pointwise feed-forward layer
        num_heads: number of heads for the mutli-head attention module
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        input_size: size of the image inputs
    """

    def __init__(
        self,
        vocab: str,
        d_model: int = 512,
        headers: int = 1,
        dff: int = 2048,
        num_heads: int = 8,
        num_layers: int = 3,
        max_length: int = 50,
        input_size: Tuple[int, int, int] = (48, 160, 3),
    ) -> None:
        super().__init__(vocab=vocab)

        self.input_size = input_size
        self.max_length = max_length
        self.vocab_size = len(vocab)

        self.feature_extractor = MAGCResnet(headers=headers, input_shape=input_size)
        self.seq_embedding = layers.Embedding(self.vocab_size + 1, d_model)  # One additional class for EOS

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=self.vocab_size,
            maximum_position_encoding=max_length,
        )
        self.feature_pe = positional_encoding(input_size[0] * input_size[1], d_model)
        self.linear = layers.Dense(self.vocab_size + 1, kernel_initializer=tf.initializers.he_uniform())

    @tf.function
    def make_mask(self, target: tf.Tensor) -> tf.Tensor:
        look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
        target_padding_mask = create_padding_mask(target, self.vocab_size)  # Pad with EOS
        combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)
        return combined_mask

    def call(self, inputs: tf.Tensor, labels: tf.Tensor, **kwargs) -> tf.Tensor:
        """Call function for training

        Args:
            inputs: images
            labels: tensor of labels

        Return:
            Computed logits
        """
        # Encode
        feature = self.feature_extractor(inputs, **kwargs)
        b, h, w, c = (tf.shape(feature)[i] for i in range(4))
        feature = tf.reshape(feature, shape=(b, h * w, c))
        encoded = feature + self.feature_pe[:, :h * w, :]

        tgt_mask = self.make_mask(labels)

        output = self.decoder(labels, encoded, tgt_mask, None, training=True)
        logits = self.linear(output)

        return logits

    @tf.function
    def decode(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Decode function for prediction

        Args:
            inputs: images to predict

        Return:
            A Tuple of tf.Tensor: predictions, logits
        """

        feature = self.feature_extractor(inputs, training=False)
        b, h, w, c = (tf.shape(feature)[i] for i in range(4))

        feature = tf.reshape(feature, shape=(b, h * w, c))
        encoded = feature + self.feature_pe[:, :h * w, :]

        max_len = tf.constant(self.max_length, dtype=tf.int32)
        start_symbol = tf.constant(self.vocab_size + 1, dtype=tf.int32)  # SOS (EOS = vocab_size)
        padding_symbol = tf.constant(self.vocab_size, dtype=tf.int32)

        ys = tf.fill(dims=(b, max_len - 1), value=padding_symbol)
        start_vector = tf.fill(dims=(b, 1), value=start_symbol)
        ys = tf.concat([start_vector, ys], axis=-1)

        final_logits = tf.zeros(shape=(b, max_len - 1, self.vocab_size + 1), dtype=tf.float32)  # don't fgt EOS
        # max_len = len + 2
        for i in range(self.max_length - 1):
            ys_mask = self.make_mask(ys)
            output = self.decoder(ys, encoded, ys_mask, None, training=False)
            logits = self.linear(output)
            prob = tf.nn.softmax(logits, axis=-1)
            next_word = tf.argmax(prob, axis=-1, output_type=ys.dtype)

            # ys.shape = B, T
            i_mesh, j_mesh = tf.meshgrid(tf.range(b), tf.range(max_len), indexing='ij')
            indices = tf.stack([i_mesh[:, i + 1], j_mesh[:, i + 1]], axis=1)

            ys = tf.tensor_scatter_nd_update(ys, indices, next_word[:, i + 1])

            if i == (self.max_length - 2):
                final_logits = logits

        # ys predictions of shape B x max_length, final_logits of shape B x max_length x vocab_size + 1
        return ys, final_logits
