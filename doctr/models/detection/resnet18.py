# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List, Union, Optional, Dict


class ResBlock(layers.Layer):
    
    def __init__(
        self,
        filters: int,
        k_size: int = 3,
        strides: int,
    ) -> None:
        self.filters = filters
        self.k_size = k_size
        self.strides = strides

    def __call__(
        self,
        x: tf.Tensor,
    ) -> tf.Tensor:

        if strides > 1:
            shortcut = layers.Conv2D(filters=self.filters, kernel_size=(1, 1), strides=self.strides)(x)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            shortcut = layers.Lambda(lambda x: x)(x)

        x = layers.Conv2D(filters=n, kernel_size=self.k_size, strides=self.strides, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=n, kernel_size=self.k_size, strides=1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x + shortcut)
        return x


class Resnet18(layers.Layer):

    def __init__(
        self,
        shape: Tuple[int, int] = (600, 600),
    ) -> None:
        self.shape = shape

    def __call__(
        self,
    ) -> layers.Layer:
        
        
        

    def resnet_bottleneck_layer(x, n, k_size, strides, force_shortcut=None):
        if strides > 1 or force_shortcut:
            shortcut = layers.Conv2D(filters=n,
                                    kernel_size=(1, 1),
                                    strides=strides)(x)
        else:
            shortcut = layers.Lambda(lambda x: x)(x)

        x = layers.Conv2D(filters=n//4,
                        kernel_size=1,
                        strides=strides,
                        padding="same",
                        use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=n//4,
                        kernel_size=k_size,
                        strides=1,
                        padding="same",
                        use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=n,
                        kernel_size=1,
                        strides=1,
                        padding="same",
                        use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x + shortcut)
        return x



    def resnet18_block(x, n):
        x = resnet_layer(x, n, k_size=3, strides=2)
        x = resnet_layer(x, n, k_size=3, strides=1)
        return x

    def first_block_resnet18(x, n):
        x = resnet_layer(x, n, k_size=3, strides=1)
        x = resnet_layer(x, n, k_size=3, strides=1)
        return x

    def resnet50_block(x, n, n_layers):
        x = resnet_bottleneck_layer(x, n, k_size=3, strides=2)
        for _ in range(1, n_layers):
            x = resnet_bottleneck_layer(x, n, k_size=3, strides=1)
        return x

    def first_block_resnet50(x, n):
        x = resnet_bottleneck_layer(x, n, k_size=3, strides=1, force_shortcut=True)
        x = resnet_bottleneck_layer(x, n, k_size=3, strides=1)
        x = resnet_bottleneck_layer(x, n, k_size=3, strides=1)
        return x


    def build_resnet18(img_h, img_w):
        """

        resnet18 backbone as described in https://arxiv.org/pdf/1512.03385.pdf 
        outputs 4 features maps (conv_2, conv_3, conv_4, conv_5) for segmentation pyramidal module

        """
        resnet_input = keras.Input(shape=(img_h, img_w, 3,), name="img")

        x = layers.Conv2D(filters=64,
                        kernel_size=7,
                        strides=2,
                        padding="same",
                        use_bias=False)(resnet_input)

        x = layers.MaxPool2D(pool_size=(2,2),
                            strides=(2,2))(x)

        x1 = first_block_resnet18(x, 64)
        x2 = resnet18_block(x1, 128)
        x3 = resnet18_block(x2, 256)
        x4 = resnet18_block(x3, 512)

        resnet18 = keras.Model(resnet_input, [x1, x2, x3, x4], name="resnet18")

        return resnet18


    def build_resnet50(img_h, img_w):
        """

        resnet50 backbone as described in https://arxiv.org/pdf/1512.03385.pdf 
        outputs 4 features maps (conv_2, conv_3, conv_4, conv_5) for segmentation pyramidal module

        """
        resnet_input = keras.Input(shape=(img_h, img_w, 3,), name="img")
        
        x = layers.Conv2D(filters=64,
                        kernel_size=7,
                        strides=2,
                        padding="same",
                        use_bias=False)(resnet_input)

        x = layers.MaxPool2D(pool_size=(2,2),
                            strides=(2,2))(x)

        x1 = first_block_resnet50(x, 256)
        x2 = resnet50_block(x1, 512, n_layers=4)
        x3 = resnet50_block(x2, 1024, n_layers=6)
        x4 = resnet50_block(x3, 2048, n_layers=3)

        resnet50 = keras.Model(resnet_input, [x1, x2, x3, x4], name="resnet50")

        return resnet50

