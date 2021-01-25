# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Tuple

from ..vgg import VGG16BN
from .core import RecognitionModel

__all__ = ['CRNN']


class CRNN(RecognitionModel):
    """CRNN with a VGG-16 backbone as described in `"Convolutional RNN: an Enhanced Model for Extracting Features
    from Sequential Data" <https://arxiv.org/pdf/1602.05875.pdf>`_.

    Args:
        num_classes: number of output classes
        input_shape: shape of the image inputs
        rnn_units: number of units in the LSTM layers
    """
    def __init__(
        self,
        num_classes: int,
        input_size: Tuple[int, int, int] = (640, 640, 3),
        rnn_units: int = 128
    ) -> None:
        super().__init__(input_size)
        self.feat_extractor = VGG16BN(input_size=input_size)
        self.decoder = Sequential(
            [
                layers.Bidirectional(layers.LSTM(units=rnn_units, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(units=rnn_units, return_sequences=True)),
                layers.Dense(units=num_classes + 1)
            ]
        )

    def __call__(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:

        features = self.feat_extractor(inputs)
        # B x H x W x C --> B x W x H x C
        transposed_feat = tf.transpose(features, perm=[0, 2, 1, 3])
        w, h, c = transposed_feat.get_shape().as_list()[1:]
        # B x W x H x C --> B x W x H * C
        features_seq = tf.reshape(transposed_feat, shape=(-1, w, h * c))
        decoded_features = self.decoder(features_seq)

        return decoded_features
