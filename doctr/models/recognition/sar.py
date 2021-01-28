# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from typing import Tuple, Dict, List

from ..vgg import vgg16_bn
from .core import RecognitionModel
from .core import RecognitionPostProcessor

__all__ = ['SAR', 'SARPostProcessor']


class AttentionModule(layers.Layer):
    """Implements attention module of the SAR model

    Args:
        attention_units: number of hidden attention units

    """
    def __init__(
        self,
        attention_units: int
    ) -> None:

        super().__init__()
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

    def call(
        self,
        features: tf.Tensor,
        hidden_state: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        [H, W] = features.get_shape().as_list()[1:3]
        # shape (N, 1, 1, rnn_units) -> (N, 1, 1, attention_units)
        hidden_state_projection = self.hidden_state_projector(hidden_state)
        # shape (N, H, W, vgg_units) -> (N, H, W, attention_units)
        features_projection = self.features_projector(features)
        projection = tf.math.tanh(hidden_state_projection + features_projection)
        # shape (N, H, W, attention_units) -> (N, H, W, 1)
        attention = self.attention_projector(projection)
        # shape (N, H, W, 1) -> (N, H * W)
        attention = self.flatten(attention)
        attention = tf.nn.softmax(attention)
        # shape (N, H * W) -> (N, H, W, 1)
        attention_map = tf.reshape(attention, [-1, H, W, 1])
        glimpse = tf.math.multiply(features, attention_map)
        # shape (N, H * W) -> (N, 1)
        glimpse = tf.reduce_sum(glimpse, axis=[1, 2])
        return glimpse, attention_map


class SARDecoder(layers.Layer):
    """Implements decoder module of the SAR model

    Args:
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        num_classes: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units
        num_decoder_layers: number of LSTM layers to stack


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

        super().__init__()
        self.num_classes = num_classes
        self.embed = layers.Dense(embedding_units, use_bias=False)
        self.attention_module = AttentionModule(attention_units)
        self.output_dense = layers.Dense(num_classes + 1, use_bias=True)
        self.max_length = max_length
        self.lstm_decoder = layers.StackedRNNCells(
            [layers.LSTMCell(rnn_units, dtype=tf.float32, implementation=1) for _ in range(num_decoder_layers)]
        )

    def call(
        self,
        features: tf.Tensor,
        holistic: tf.Tensor,
    ) -> tf.Tensor:

        batch_size = tf.shape(features)[0]
        # initialize states (each of shape (N, rnn_units))
        states = self.lstm_decoder.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=tf.float32
        )
        # run first step of lstm
        # holistic: shape (N, rnn_units)
        _, states = self.lstm_decoder(holistic, states)
        sos_symbol = self.num_classes + 1
        symbol = sos_symbol * tf.ones(shape=(batch_size,), dtype=tf.int32)
        logits_list = []
        for t in range(self.max_length + 1):  # keep 1 step for <eos>
            # one-hot symbol with depth num_classes + 2
            # embeded_symbol: shape (N, embedding_units)
            embeded_symbol = self.embed(tf.one_hot(symbol, depth=self.num_classes + 2))
            logits, states = self.lstm_decoder(embeded_symbol, states)
            glimpse, attention_map = self.attention_module(
                features=features, hidden_state=tf.expand_dims(tf.expand_dims(logits, axis=1), axis=1)
            )
            # logits: shape (N, rnn_units), glimpse: shape (N, 1)
            logits = tf.concat([logits, glimpse], axis=-1)
            # shape (N, rnn_units + 1) -> (N, num_classes + 1)
            logits = self.output_dense(logits)
            logits_list.append(logits)
        outputs = tf.stack(logits_list, axis=1)  # shape (N, max_length + 1, num_classes + 1)

        return outputs


class SAR(RecognitionModel):
    """SAR with a VGG16 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        input_size: shape of the input (H, W) in pixels
        rnn_units: number of hidden units in both encoder and decoder LSTM
        embedding_units: number of embedding units
        attention_units: number of hidden units in attention module
        max_length: maximum word length handled by the model
        num_classes: size of the alphabet
        num_decoder_layers: number of LSTM to stack in decoder layer

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

        self.feat_extractor = vgg16_bn(input_size=input_size, include_top=False)

        self.encoder = Sequential(
            [
                layers.LSTM(units=rnn_units, return_sequences=True),
                layers.LSTM(units=rnn_units, return_sequences=False)
            ]
        )

        self.decoder = SARDecoder(
            rnn_units, max_length, num_classes, embedding_units, attention_units, num_decoder_layers,

        )

    def call(
        self,
        inputs: tf.Tensor
    ) -> tf.Tensor:

        features = self.feat_extractor(inputs)
        pooled_features = tf.reduce_max(features, axis=1)  # vertical max pooling
        encoded = self.encoder(pooled_features)
        decoded = self.decoder(features=features, holistic=encoded)

        return decoded


class SARPostProcessor(RecognitionPostProcessor):
    """
    Postprocess raw prediction of the model (logits) to a list of words

    Args:
        label_to_idx: dictionnary mapping alphabet labels to idx of the model classes
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters
    """
    def __init__(
        self,
        label_to_idx: Dict[str, int],
        ignore_case: bool = False,
        ignore_accents: bool = False
    ) -> None:

        self.label_to_idx = label_to_idx
        self.ignore_case = ignore_case
        self.ignore_accents = ignore_accents

    def __call__(
        self,
        logits: tf.Tensor,
    ) -> List[str]:
        # compute pred with argmax for attention models
        pred = tf.math.argmax(logits, axis=2)

        # create tf_label_to_idx mapping to decode classes
        label_mapping = self.label_to_idx.copy()
        label_mapping['<eos>'] = int(len(label_mapping))
        label, _ = zip(*sorted(label_mapping.items(), key=lambda x: x[1]))
        tf_label_to_idx = tf.constant(
            value=label, dtype=tf.string, shape=[int(len(label_mapping))], name='dic_idx_label'
        )

        # decode raw output of the model with tf_label_to_idx
        pred = tf.cast(pred, dtype='int32')
        decoded_strings_pred = tf.strings.reduce_join(inputs=tf.nn.embedding_lookup(tf_label_to_idx, pred), axis=-1)
        decoded_strings_pred = tf.strings.split(decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(decoded_strings_pred.to_sparse(), default_value='not valid')[:, 0]
        words_list = [word.decode() for word in list(decoded_strings_pred.numpy())]

        if self.ignore_case:
            words_list = [word.lower() for word in words_list]

        if self.ignore_accents:
            raise NotImplementedError

        return words_list
