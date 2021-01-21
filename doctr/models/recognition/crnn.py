# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from keras import layers

class VGG(layers.Layer):
    """Visual Geometry Group (Oxford, 2014) network

    Args:

    """

    def __init__(
        self,
    ) -> :

    def __call__(
        self
    ) -> :

def conv2d_batchnorm_relu_maxpool2d(x,
                                num_filters,
                                k_size, 
                                p_size):

    module = keras.Sequential(
        [
            layers.Conv2D(filters=num_filters, kernel_size=k_size, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=p_size, strides=p_size, padding="same"),
        ]
    )
    return module

def conv2d_batchnorm_relu(x, num_filters, k_size):

    module = keras.Sequential(
        [
            layers.Conv2D(filters=num_filters, kernel_size=k_size, strides=1, padding="same", use_bias=False)
            layers.BatchNormalization()
            layers.Activation('relu')
        ]
    )
    return module


def build_vgg(img_h, img_w, filters, k_size):

    vgg_input = keras.Input(shape=(img_h, img_w, 3,), name="img")

    x = conv2d_batchnorm_relu_maxpool2d(vgg_input, num_filters=filters, k_size=k_size, p_size=2)
    x = conv2d_batchnorm_relu_maxpool2d(x, num_filters=2*filters, k_size=k_size, p_size=2)

    x = conv2d_batchnorm_relu(x, num_filters=4*filters, k_size=k_size)
    x = conv2d_batchnorm_relu_maxpool2d(x, num_filters=4*filters, k_size=k_size, p_size=[2,1])

    x = conv2d_batchnorm_relu(x, num_filters=8*filters, k_size=k_size)
    x = conv2d_batchnorm_relu_maxpool2d(x, num_filters=8*filters, k_size=k_size, p_size=[2,1])

    vgg_output = conv2d_batchnorm_relu(x, num_filters=8*filters, k_size=k_size)

    vgg = keras.Model(vgg_input, vgg_output, name="vgg")

    return vgg
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
