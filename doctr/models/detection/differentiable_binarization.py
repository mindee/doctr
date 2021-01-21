# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv2
import json
import os
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Union, List, Tuple, Optional, Any, Dict

from .postprocessor import PostProcessor
from .model import DetectionModel

__all__ = ['DBPostProcessor', 'DBModel']


class DBPostProcessor(PostProcessor):
    """Class to postprocess Differentiable binarization model outputs
    Inherits from Postprocessor

    Args:
        unclip ratio (Union[float, int]): ratio used to unshrink polygons
        min_size_box (int): minimal length (pix) to keep a box
        max_candidates (int): maximum boxes to consider in a single page
        box_thresh (float): minimal objectness score to consider a box
        bin_thresh (float): threshold used to binzarized p_map at inference time

    """
    def __init__(
        self,
        unclip_ratio: Union[float, int] = 1.5,
        min_size_box: int = 5,
        max_candidates: int = 100,
        box_thresh: float = 0.5,
        bin_thresh: float = 0.3,
    ) -> None:

        super().__init__(
            min_size_box,
            max_candidates,
            box_thresh
        )
        self.unclip_ratio = unclip_ratio
        self.bin_thresh = bin_thresh

    @staticmethod
    def box_score(
        pred: np.ndarray,
        points: np.ndarray
    ) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            score (float): Polygon objectness

        """
        h, w = pred.shape[:2]
        xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int), 0, h - 1)

        return pred[ymin:ymax + 1, xmin:xmax + 1].mean()

    def polygon_to_box(
        self,
        points: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a 4-points box

        Args:
            points (np.ndarray): The first parameter.

        Returns:
            box (Tuple[int, int, int, int]): an absolute box (x, y, w, h)

        """
        poly = Polygon(points)
        distance = poly.area * self.unclip_ratio / poly.length  # compute distance to expand polygon
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded_points = np.array(offset.Execute(distance))  # expand polygon
        x, y, w, h = cv2.boundingRect(expanded_points)  # compute a 4-points box from expanded polygon
        return x, y, w, h

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> List[List[float]]:
        """Compute boxes from a bitmap/pred_map

        Args:
            pred (np.ndarray): Pred map from differentiable binarization output
            bitmap (np.ndarray): Bitmap map computed from pred (binarized)

        Returns:
            boxes (List[List[float]]): list of boxes for the bitmap, each box is a 5-element list
                containing x, y, w, h, score for the box

        """
        height, width = bitmap.shape[:2]
        boxes = []
        # get contours from connected components on the bitmap
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)  # approximate contour by a polygon
            points = approx.reshape((-1, 2))  # get polygon points
            if points.shape[0] < 4:  # remove polygons with 3 points or less
                continue
            score = self.box_score(pred, points.reshape(-1, 2))
            if self.box_thresh > score:   # remove polygons with a weak objectness
                continue
            x, y, w, h = self.polygon_to_box(points)
            if h < self.min_size_box or w < self.min_size_box:  # remove to small boxes
                continue
            x = x / width  # compute relative polygon to get rid of img shape
            y = y / height
            w = w / width
            h = h / height
            boxes.append([x, y, w, h, score])
        return boxes

    def __call__(
        self,
        raw_pred: List[tf.Tensor],
    ) -> List[List[np.ndarray]]:
        """Performs postprocessing for a list of model outputs

        Args:
            raw_pred (List[tf.Tensor]): list of raw output from the model,
                each tensor has a shape (batch_size x H x W x 1)

        returns:
            bounding_boxes (List[List[np.ndarray]]): list of batches, each batches is a list of tensor.
                Each tensor (= 1 image) has a shape(num_boxes, 5).

        """
        bounding_boxes = []
        for raw_batch in raw_pred:
            p = tf.squeeze(raw_batch, axis=-1)  # remove last dim
            bitmap = tf.cast(p > self.bin_thresh, tf.float32)

            p = tf.unstack(p, axis=0)
            bitmap = tf.unstack(bitmap, axis=0)

            boxes_batch = []

            for p_, bitmap_ in zip(p, bitmap):
                p_ = p_.numpy()
                bitmap_ = bitmap_.numpy()
                boxes = self.bitmap_to_boxes(pred=p_, bitmap=bitmap_)
                boxes_batch.append(np.array(boxes))

            bounding_boxes.append(boxes_batch)
        return bounding_boxes


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
        super().__init__(shape)
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

        resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=resnet_input,
            input_shape=(self.shape[0], self.shape[1], 3,),
            pooling=None,
        )

        return resnet

    def build_feat_extractor(
        self
    ) -> tf.keras.Model:
        """Build a tf.keras.Model feature extractor = returning the 4 feature maps

        Args:

        Returns:
            a tf.keras.Model feat_extractor, returning a list of feature maps

        """
        resnet = self.build_resnet()
        res_layers = [
            resnet.get_layer("conv2_block3_out"),
            resnet.get_layer("conv3_block4_out"),
            resnet.get_layer("conv4_block6_out"),
            resnet.get_layer("conv5_block3_out"),
        ]
        res_outputs = [res_layer.output for res_layer in res_layers]
        feat_extractor = keras.Model(resnet.input, outputs=res_outputs)

        return feat_extractor

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
                layers.Conv2D(filters=self.channels, kernel_size=(3, 3), strides=(1, 1), padding='same'),
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

        z1 = self.conv_upsampling(up=0)(y3)
        z2 = self.conv_upsampling(up=2)(y2)
        z3 = self.conv_upsampling(up=4)(y1)
        z4 = self.conv_upsampling(up=8)(x4)

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
        training: bool = False
    ) -> Tuple[keras.Model, keras.Model]:

        feat_extractor = self.build_feat_extractor()
        features_maps = feat_extractor(inputs)

        reduced_channel_feat = self.reduce_channel(features_maps)
        concat_features = self.pyramid_module(reduced_channel_feat)

        probability_map = self.get_p_map()(concat_features)
        treshold_map = self.get_t_map()(concat_features)

        approx_binary_map = self.get_approximate_binary_map(probability_map, treshold_map)

        if training:
            return [probability_map, treshold_map, approx_binary_map]
        else:
            return probability_map
