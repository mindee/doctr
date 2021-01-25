# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from typing import Tuple

from ..vgg import VGG16BN
from .core import RecognitionModel

__all__ = ['SARResNet50']


class AttentionModule(layers.Layer):
    """Implements attention module of the SAR model

    Args:
        attention_units: number of hidden attention units

    """
    def __init__(
        self,
        attention_units: int
    ) -> None:

        self.hidden_state_projector = layers.Conv2D(
            filters=attention_units, kernel_size=1, strides=1, use_bias=False, padding='same'
        )
        self.features_projector = layers.Conv2D(
            filters=attention_units, kernel_size=3, strides=1, use_bias=True, padding='same'
        )
        self.attention_projector = layers.Conv2D(
            filters=1, kernel_size=1, strides=1, use_bias=False, padding="same"
        )
        self.flatten = layers.Flatten()

    def __call__(
        self,
        features: tf.Tensor,
        hidden_state: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        [H, W] = features.get_shape().as_list()[1:3]
        # shape (N, 1, 1, rnn_units) -> (N, 1, 1, attention_units)
        hidden_state_projection = self.hidden_state_projector(hidden_state)
        # shape (N, H, W, vgg_units) -> (N, H, W, attention_units)
        features_projection = self.feature_projector(features)
        projection = tf.math.tanh(hidden_state_projection + features_projection)
        # shape (N, H, W, attention_units) -> (N, H, W, 1)
        attention = self.attention_projector(projection)
        # shape (N, H, W, 1) -> (N, H * W)
        attention = self.flatten(attention)
        attention = tf.nn.softmax(attention)
        # shape (N, H * W) -> (N, H, W, 1)
        attention = tf.reshape(attention, [-1, H, W, 1])
        glimpse = tf.math.multiply(features, attention)
        # shape (N, H * W) -> (N, 1)
        glimpse = tf.reduce_sum(glimpse, axis=[1, 2])
        return glimpse, attention


class AttentionDecoder(layers.Layer):
    """Implements attention decoder module of the SAR model

    Args:
        num_classes: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units
        decoder: lstm decoder layer

    """
    def __init__(
        self,
        num_classes: int,
        embedding_units: int,
        attention_units: int,
        decoder: layers.Layer,
    ) -> None:

        self.num_classes = num_classes
        self.embed = layers.Dense(embedding_units, use_bias=False)
        self.decoder = decoder
        self.attention_module = AttentionModule(attention_units)
        self.output_dense = layers.Dense(num_classes + 1, use_bias=True)

    def __call__(
        self,
        symbol: tf.Tensor,
        states: tf.Tensor,
        features: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        # embed symbol: shape symbol (N,) -> (N, num_classes + 2)
        embeded_symbol = self.embed(tf.one_hot(symbol, depth=self.num_classes + 2))
        print(embeded_symbol.shape)

        output, states = self.decoder(embeded_symbol, states)
        attention_state, attention_map = self.attention_module(
            features=features, hidden_state=tf.expand_dims(tf.expand_dims(output, axis=1), axis=1)
        )
        output = tf.concat([output, attention_state], axis=-1)
        output = self.output_dense(output)

        return output, attention_map, states


class Decoder(layers.Layer):
    """Implements decoder module of the SAR model

    Args:
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        num_decoder_layers: number of LSTM layers to stack
        num_classes: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units

    """
    def __init__(
        self,
        rnn_units: int,
        max_length: int,
        num_classes: int,
        embedding_units: int,
        attention_units: int,
        num_decoder_layers: int = 2
    ) -> None:

        self.num_classes = num_classes
        self.max_length = max_length
        self.lstm_decoder = layers.StackedRNNCells(
            [layers.LSTMCell(rnn_units, implementation=1) for _ in range(num_decoder_layers)]
        ))
        self.attention_decoder = AttentionDecoder(
            num_classes, embedding_units, attention_units, decoder=self.lstm_decoder
        )

    def __call__(
        self,
        features: tf.Tensor,
        holistic: tf.Tensor,
    ) -> tf.Tensor:

        batch_size = tf.shape(features)[0]
        states = self.lstm_decoder.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=tf.float32
        )  # shape (N, rnn_units)

        # run first step of lstm
        _, states = self.lstm_decoder(holistic, states)  # shape (N, rnn_units)
        sos_symbol = self.num_classes + 1
        symbol = sos_symbol * tf.ones(shape=(batch_size,), dtype=tf.int32)

        logits_list = []
        for t in range(self.max_length + 1):
            logits, attentions, states = self.attention_decoder(symbol, states, features)
            logits_list.append(logits)
        outputs = tf.stack(logits_list, axis=1)  # shape (N, max_length + 1, )

        return outputs


class SARResNet50(RecognitionModel):
    """SAR with a ResNet-50 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        input_size (Tuple[int, int]): shape of the input (H, W) in pixels

    """
    def __init__(
        self,
        input_size: Tuple[int, int, int],
        rnn_units: int,
        embedding_units: int,
        attention_units: int,
        max_length: int,
        num_classes: int,
        num_decoder_layers: int
    ) -> None:

        super().__init__(input_size)

        self.feat_extractor = VGG16BN(input_size=input_size)

        self.encoder = Sequential(
            [
                layers.Bidirectional(layers.LSTM(units=rnn_units, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(units=rnn_units, return_sequences=False))
            ]
        )

        self.decoder = Decoder(
            rnn_units, max_length, num_classes, embedding_units, attention_units, num_decoder_layers,

        )

    def __call__(
        self,
        inputs: tf.Tensor
    ) -> tf.Tensor:

        features = self.feat_extractor(inputs)
        pooled_features = tf.reduce_max(features, axis=1)  # vertical max pooling
        encoded = self.encoder(pooled_features)
        decoded = self.decoder(features=features, holistic=encoded)

        return decoded
