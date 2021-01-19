# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from typing import List, Union, Optional, Tuple, Dict


class DBModel(DetectionModel, keras.Model):

    def __init__(
        self,
        backbone: str = "resnet18",
        shape: Tuple[int, int] = (600, 600),
        channels: int = 128,
    ) -> None:
        """
        Backbone : choice between a light (resnet18) backbone and a heavy (resnet30) backbone
        """

        self.backbone = backbone
        self.shape = shape
        self.channels = channels

    def upsampling_addition(
        self,
        x_small: tf.Tensor,
        x_big: tf.Tensor
    ) -> tf.Tensor:
        """
        Performs Upsampling x2 on x_small and element-wise addition x_small + x_big
        """

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x_small)
        x = layers.Add()([x, x_big])
        return x

    def conv_upsampling(
        self,
        up: int = 0,
    ) -> layers.Layer:
        """
        Module which performs a 3x3 convolution followed by up-sampling
        up: dilatation factor to scale the convolution output before concatenation
        """

        model = keras.Sequential(
            [
                layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=(1,1), padding='same'),
                layers.BatchNormalization(),
                layers.Activation('relu'),
            ]
        )
        if up > 0:
            model.add(layers.UpSampling2D(size=(up, up), interpolation='nearest'))

        return model

    def reduce_channel(
        self,
        feat_maps: List[tf.Tensor],
    ) -> List[tf.Tensor]:
        """
        Set channels for all tensors of the feat_maps list to n, performing a 1x1 conv
        """

        new_feat_maps = [0, 0, 0, 0]
        for i in range(len(feat_maps)):
            new_feat_maps[i] = layers.Conv2D(filters=self.channels, kernel_size=(1, 1), strides=1)(feat_maps[i])

        return new_feat_maps


    def pyramid_module(
        self,
        x: List[tf.Tensor],
    ) -> tf.Tensor:
        """
        Implements Pyramidal module as described in paper, 
        input: List of features maps (from resnet backbone)
        output: concatenated features
        """

        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        y1 = self.upsampling_addition(x4, x3)
        y2 = self.upsampling_addition(y1, x2)
        y3 = self.upsampling_addition(y2, x1)

        z1 = self.conv_upsampling(self.channels, up=0)(y3)
        z2 = self.conv_upsampling(self.channels, up=2)(y2)
        z3 = self.conv_upsampling(self.channels, up=4)(y1)
        z4 = self.conv_upsampling(self.channels, up=8)(x4)

        features_concat = layers.Concatenate()([z1, z2, z3, z4])

        return features_concat


    def get_p_map(
        self,
    ) -> layers.Layer:
        """
        get probability or treshold map function from features
        """

        model = keras.Sequential(
            [
                layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False, name="p_map1"),
                layers.BatchNormalization(name="p_map2"),
                layers.Activation('relu'),
                layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="p_map3"),
                layers.BatchNormalization(name="p_map4"),
                layers.Activation('relu'),
                layers.Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(2, 2), name="p_map5"),
                layers.Activation('sigmoid'),
            ]
        )

        return model
    
    def get_t_map(
        self,
    ) -> layers.Layer:
        """
        get treshold map function from features
        """

        model = keras.Sequential(
            [
                layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False, name="t_map1"),
                layers.BatchNormalization(name="t_map2"),
                layers.Activation('relu'),
                layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="t_map3"),
                layers.BatchNormalization(name="t_map4"),
                layers.Activation('relu'),
                layers.Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(2, 2), name="t_map5"),
                layers.Activation('sigmoid'),
            ]
        )
        return model


    def get_approximate_binary_map(
        self,
        p: tf.Tensor,
        t: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute approximate binary map as described in paper, from threshold map t and probability map p
        """

        b_hat = layers.Lambda(lambda x: 1 / (1 + tf.exp(-50. * (x[0] - x[1]))), name="approx_bin_map")([p, t])
        return b_hat

    def __call__(
        self,
    ) -> Tuple[keras.Model, keras.Model]:
        """
        Returns a tuple of keras DB models : training and inference models
        At inference time, we remove the threshold branch to fasten computation
        """

        db_input = keras.Input(shape=(self.shape[0], self.shape[1], 3,), name="img")

        if self.backbone == 'resnet18':
            resnet = build_resnet18(img_h, img_w)
        
        if self.backbone == 'resnet50':
            resnet = build_resnet50(img_h, img_w)

        features = resnet(dbnet_input)
        reduced_channel_feat = self.reduce_channel(features)
        concat_features = self.pyramid_module(reduce_channel_feat)

        probability_map = self.get_p_map()(concat_features)
        treshold_map = self.get_t_map()(concat_features)

        approx_binary_map = self.get_approximate_binary_map(probability_map, treshold_map)

        dbnet_training = keras.Model(
            inputs=dbnet_input, outputs=[probability_map, treshold_map, approx_binary_map], name="dbnet_training"
            )
        dbnet_inference = keras.Model(inputs=dbnet_input, outputs=probability_map, name="dbnet_inference")

        return dbnet_training, dbnet_inference

