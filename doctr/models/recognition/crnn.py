# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from keras import layers

class VGG(layers.Layer):
    """Visual Geometry Group (Oxford, 2014) network

    Args:
        filters: number of filters
        k_size: kernel size used for convolutions

    """
    def __init__(
        self,
        filters: int,
        k_size: int
    ) -> None:

        self.conv_maxpool_1 = conv_maxpool(num_filters=filters, k_size=k_size, p_size=2)
        self.conv_maxpool_2 = conv_maxpool(num_filters=2*filters, k_size=k_size, p_size=2)
        self.conv_1 = conv(num_filters=4*filters, k_size=k_size)
        self.conv_maxpool_3 = conv_maxpool(num_filters=4*filters, k_size=k_size, p_size=[2,1])
        self.conv_2 = conv(num_filters=8*filters, k_size=k_size)
        self.conv_maxpool_4 = self.conv_maxpool(num_filters=8*filters, k_size=k_size, p_size=[2,1])
        self.conv_3 = conv(num_filters=8*filters, k_size=k_size)

    @staticmethod
    def conv_maxpool(
        num_filters: int,
        k_size: int, 
        p_size: Union[int, List[int]]
        ) -> layers.Layer:

        module = keras.Sequential(
            [
                layers.Conv2D(filters=num_filters, kernel_size=k_size, strides=1, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPool2D(pool_size=p_size, strides=p_size, padding="same"),
            ]
        )
        return module

    @staticmethod
    def conv(
        num_filters: int,
        k_size: int
        ) -> layers.Layer:

        module = keras.Sequential(
            [
                layers.Conv2D(filters=num_filters, kernel_size=k_size, strides=1, padding="same", use_bias=False)
                layers.BatchNormalization()
                layers.Activation('relu')
            ]
        )
        return module

    def __call__(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:

        x = self.conv_maxpool_1(inputs)
        x = self.conv_maxpool_2(x)
        x = self.conv_1(x)
        x = self.conv_maxpool_3(x)
        x = self.conv_2(x)
        x = self.conv_maxpool_4(x)
        x = self.conv_3(x)
        return x

class CRNN(RecognitionModel):
    """Convolutional recurrent neural network (CRNN) class as described in paper

    Args:

    """

    def __init__(
        self,
    ) -> :

    def __call__(
        self
    ) -> :

def build_crnn(img_h,
            img_w,
            filters,
            k_size,
            rnn_units,
            num_classes):

    #vgg_model = build_vgg(img_h, img_w, filters, k_size)
    vgg_model = build_resnet(img_h, img_w)

    vgg_input = keras.Input(shape=(img_h, img_w, 3,), name="img")
    vgg_output = vgg_model(vgg_input)

    # RESHAPING THE FEATURE MAP
    features = layers.Permute(dims=(2,1,3))(vgg_output)
    num_columns, num_lines, num_features = features.get_shape().as_list()[1:]
    features_seq = layers.Reshape(target_shape=(num_columns, num_lines * num_features))(features)

    x = layers.Bidirectional(layers.LSTM(units=rnn_units,
                                    return_sequences=True))(features_seq)

    x = layers.Bidirectional(layers.LSTM(units=rnn_units,
                                    return_sequences=True))(x)

    crnn_output =  layers.Dense(units=num_classes + 1)(x)

    crnn = keras.Model(vgg_input, crnn_output, name="crnn")

    return crnn
