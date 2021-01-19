# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def up_scale_addition(x_small, x_big):
    x = layers.UpSampling2D(size=(2, 2),
                        interpolation='nearest')(x_small)
    x = layers.Add()([x, x_big])
    return x
 

def conv_up(x, n:int, up:int):
    x = layers.Conv2D(filters=n, 
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if up > 0:
        x = layers.UpSampling2D(size=(up, up),
                        interpolation='nearest')(x)
    return x 


def reduce_channel(feat_maps, n):
    thin_feat_maps = [0, 0, 0, 0]
    for i in range(len(feat_maps)):
        thin_feat_maps[i] = layers.Conv2D(filters=n,
                                kernel_size=(1, 1),
                                strides=1)(feat_maps[i])
    return thin_feat_maps


def pyramid_module(x, n:int):

    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

    y1 = up_scale_addition(x4, x3)
    y2 = up_scale_addition(y1, x2)
    y3 = up_scale_addition(y2, x1)

    z1 = conv_up(y3, n, up=0)
    z2 = conv_up(y2, n, up=2)
    z3 = conv_up(y1, n, up=4)
    z4 = conv_up(x4, n, up=8)

    features_concat = layers.Concatenate()([z1, z2, z3, z4])

    return features_concat


def get_p_map(features):
    """

    get probability or treshold map function from features

    """
    p = layers.Conv2D(filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=False, name="p_map1")(features)
    p = layers.BatchNormalization(name="p_map2")(p)
    p = layers.Activation('relu')(p)

    p = layers.Conv2DTranspose(filters=64,
                            kernel_size=(2, 2),
                            strides=(2, 2),
                            use_bias=False, name="p_map3")(p)
    p = layers.BatchNormalization(name="p_map4")(p)
    p = layers.Activation('relu')(p)

    p = layers.Conv2DTranspose(filters=1,
                            kernel_size=(2, 2), 
                            strides=(2, 2), name="p_map5")(p)
    p = layers.Activation('sigmoid')(p)

    return p
 
def get_t_map(features):
    """

    get probability or treshold map function from features

    """
    t = layers.Conv2D(filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=False, name="t_map1")(features)
    t = layers.BatchNormalization(name="t_map2")(t)
    t = layers.Activation('relu')(t)

    t = layers.Conv2DTranspose(filters=64,
                            kernel_size=(2, 2),
                            strides=(2, 2),
                            use_bias=False, name="t_map3")(t)
    t = layers.BatchNormalization(name="t_map4")(t)
    t = layers.Activation('relu')(t)

    t = layers.Conv2DTranspose(filters=1,
                            kernel_size=(2, 2), 
                            strides=(2, 2), name="t_map5")(t)
    t = layers.Activation('sigmoid')(t)

    return t


def get_approximate_binary_map(p, t):
    b_hat = layers.Lambda(lambda x: 1 / (1 + tf.exp(-50. * (x[0] - x[1]))), name="approx_bin_map")([p, t])
    return b_hat

   
def build_dbnet(img_h, img_w, n:int, backbone:str):
    """

    build db net as described in https://arxiv.org/pdf/1911.08947.pdf 
    backbone: choose between 'resnet18' (light) or resnet50 (heavy)

    """
    dbnet_input = keras.Input(shape=(img_h, img_w, 3,), name="img")

    if backbone == 'resnet18':
        resnet = build_resnet18(img_h, img_w)
    
    if backbone == 'resnet50':
        resnet = build_resnet50(img_h, img_w)

    features = resnet(dbnet_input)
    thin_features = reduce_channel(features, n)
    concat_features = pyramid_module(thin_features, n)

    probability_map = get_p_map(concat_features)
    treshold_map = get_t_map(concat_features)

    approx_binary_map = get_approximate_binary_map(probability_map, treshold_map)

    dbnet_training = keras.Model(inputs=dbnet_input, outputs=[probability_map, treshold_map, approx_binary_map], name="dbnet_training")
    dbnet_inference = keras.Model(inputs=dbnet_input, outputs=probability_map, name="dbnet_inference")

    return dbnet_training, dbnet_inference

