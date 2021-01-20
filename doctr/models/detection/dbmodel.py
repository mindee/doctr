# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from typing import List, Union, Optional, Tuple, Dict


class DBModel(DetectionModel, keras.Model):
    """Implements DB keras model

    Args:
        shape (Tuple[int, int]): shape of the input (h, w) in pixels
        channels (int): number of channels too keep during after extracting features map

    """

    def __init__(
        self,
        shape: Tuple[int, int] = (600, 600),
        channels: int = 128,
    ) -> None:
        self.shape = shape
        self.channels = channels

    def build_resnet(
        self,
    ) -> tf.keras.Model:
        """Import and build ResNet50V2 from the keras.applications lib

        Args:

        Returns:
            a resnet model (instance of tf.keras.Model)

        """
        resnet_input = keras.Input(shape=(self.shape[0], self.shape[1], 3,), name="input")
        
        resnet = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_tensor=db_input,
            input_shape=(self.shape[0], self.shape[1], 3,),
            pooling=None,
        )

        return resnet
    
    @staticmethod
    def upsampling_addition(
        x_small: tf.Tensor,
        x_big: tf.Tensor
    ) -> tf.Tensor:
        """Performs Upsampling x2 on x_small and element-wise addition x_small + x_big

        Args:
            x_small (tf.Tensor): small tensor to upscale before addition
            x_big (tf.Tensor): big tensor to sum with the up-scaled x_small

        Returns:
            a tf.Tensor

        """
        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x_small)
        x = layers.Add()([x, x_big])
        return x

    def conv_upsampling(
        self,
        up: int = 0,
    ) -> layers.Layer:
        """Module which performs a 3x3 convolution followed by up-sampling
        
        Args:
            up (int): dilatation factor to scale the convolution output before concatenation

        Returns:
            a  keras.layers.Layer object, wrapiing these operations in a sequential module

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
        """Set channels for all tensors of the feat_maps list to self.channels, performing a 1x1 conv

        Args:
            feat_maps (List[tf.Tensor]): list of features maps

        Returns:
            a List[tf.Tensor], the feature_maps with self.channels channels

        """
        new_feat_maps = [0, 0, 0, 0]
        for i in range(len(feat_maps)):
            new_feat_maps[i] = layers.Conv2D(filters=self.channels, kernel_size=(1, 1), strides=1)(feat_maps[i])

        return new_feat_maps


    def pyramid_module(
        self,
        x: List[tf.Tensor],
    ) -> tf.Tensor:
        """Implements Pyramidal module as described in paper, 
        
        Args: 
            x (List[tf.Tensor]): List of features maps (from resnet backbone)

        Returns: 
            concatenated features (tf.Tensor)

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

    @staticmethod
    def get_p_map() -> layers.Layer:
        """Get probability map module, wrapped in a sequential model

        Args:

        Returns:
            a tf.keras.layers.Layer

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
    
    @staticmethod
    def get_t_map() -> layers.Layer:
        """Get threshold map module, wrapped in a sequential model

        Args:

        Returns:
            a tf.keras.layers.Layer

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

    @staticmethod
    def get_approximate_binary_map(
        p: tf.Tensor,
        t: tf.Tensor
    ) -> tf.Tensor:
        """Compute approximate binary map as described in paper, 
        from threshold map t and probability map p

        Args:
            p (tf.Tensor): probability map
            t (tf.Tensor): threshold map

        returns:
            a tf.Tensor

        """
        b_hat = layers.Lambda(lambda x: 1 / (1 + tf.exp(-50. * (x[0] - x[1]))), name="approx_bin_map")([p, t])
        return b_hat

    def __call__(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> Tuple[keras.Model, keras.Model]:

        resnet = self.build_resnet()
        features_maps = [resnet(inputs).get_layer('conv'+i+'_block3_out').output for i in range(2, 6)]

        reduced_channel_feat = self.reduce_channel(features_map)
        concat_features = self.pyramid_module(reduce_channel_feat)

        probability_map = self.get_p_map()(concat_features)
        treshold_map = self.get_t_map()(concat_features)

        approx_binary_map = self.get_approximate_binary_map(probability_map, treshold_map)

        if training:
            return [probability_map, treshold_map, approx_binary_map]
        else:
            return probability_map
