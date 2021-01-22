# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Tuple, List, Union

from doctr.models.recognition.model import RecognitionModel

__all__ = ['CRNN']


class VGG16(Sequential):
    """Visual Geometry Group (Oxford, 2014) network

    Args:
        input_shape: shapes of the images

    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (640, 640, 3)
    ) -> None:
        _layers = [
            layers.Conv2D(64, 3, padding='same', input_shape=input_size),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 1)),
            layers.Conv2D(512, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 1)),
            layers.Conv2D(512, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 1)),
        ]
        super().__init__(_layers)


class CRNN(RecognitionModel):
    """Convolutional recurrent neural network (CRNN) class as described in paper
    Feature Extractor: VGG16

    Args:
        input_shape: shape of the image inputs

    """
    def __init__(
        self,
        num_classes: int,
        input_size: Tuple[int, int, int] = (640, 640, 3),
        rnn_units: int = 128
    ) -> None:
        super().__init__(input_size)
        self.vgg16 = VGG16(input_size=input_size)
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
        training: bool = False,
    ) -> tf.Tensor:

        features = self.vgg16(inputs)
        print(features.shape)
        transposed_feat = tf.transpose(features, perm=[0, 2, 1, 3])
        num_columns, num_lines, num_features = transposed_feat.get_shape().as_list()[1:]
        features_seq = tf.reshape(transposed_feat, shape=(-1, num_columns, num_lines * num_features))
        decoded_features = self.decoder(features_seq)

        return decoded_features
