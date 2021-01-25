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

from .core import DetectionModel, PostProcessor

__all__ = ['DBPostProcessor', 'DBResNet50']


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


class FeaturePyramidNetwork(layers.Layer):
    """Feature Pyramid Network as described in `"Feature Pyramid Networks for Object Detection"
    <https://arxiv.org/pdf/1612.03144.pdf>`_.

    Args:
        channels: number of channel to output
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.inner_blocks = [layers.Conv2D(filters=channels, kernel_size=1, strides=1) for _ in range(4)]
        self.layer_blocks = [self.build_upsampling(channels, dilation_factor=2 ** idx) for idx in range(4)]

    @staticmethod
    def build_upsampling(
        channels: int,
        dilation_factor: int = 1,
    ) -> layers.Layer:
        """Module which performs a 3x3 convolution followed by up-sampling

        Args:
            channels: number of output channels
            dilation_factor (int): dilation factor to scale the convolution output before concatenation

        Returns:
            a keras.layers.Layer object, wrapping these operations in a sequential module

        """
        _layers = [
            layers.Conv2D(filters=channels, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ]

        if dilation_factor > 1:
            _layers.append(layers.UpSampling2D(size=(dilation_factor, dilation_factor), interpolation='nearest'))

        module = keras.Sequential(_layers)

        return module

    def __call__(
        self,
        x: List[tf.Tensor]
    ) -> tf.Tensor:

        # Channel mapping
        results = [block(fmap) for block, fmap in zip(self.inner_blocks, x)]
        # Upsample & sum
        for idx in range(len(results) - 1, -1):
            results[idx] += self.upsample(results[idx + 1])
        # Conv & upsample
        results = [block(fmap) for block, fmap in zip(self.layer_blocks, results)]

        return layers.concatenate(results)


class IntermediateLayerGetter(keras.Model):
    """Implements an intermediate layer getter

    Args:
        model: the model to extract feature maps from
        layer_names: the list of layers to retrieve the feature map from
    """
    def __init__(
        self,
        model: tf.keras.Model,
        layer_names: List[str]
    ) -> None:
        intermediate_fmaps = [model.get_layer(layer_name).output for layer_name in layer_names]
        super().__init__(model.input, outputs=intermediate_fmaps)


class DBResNet50(DetectionModel):
    """DBNet with a ResNet-50 backbone as described in `"Real-time Scene Text Detection with Differentiable
    Binarization" <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
        input_size (Tuple[int, int]): shape of the input (H, W) in pixels
        channels (int): number of channels too keep during after extracting features map
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (600, 600),
        channels: int = 128,
    ) -> None:

        super().__init__(input_size)

        resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(*input_size, 3),
            pooling=None,
        )

        self.feat_extractor = IntermediateLayerGetter(
            resnet,
            ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
        )

        self.fpn = FeaturePyramidNetwork(channels=channels)

        self.probability_head = keras.Sequential(
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
        self.threshold_head = keras.Sequential(
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

    @staticmethod
    def compute_approx_binmap(
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
        return 1 / (1 + tf.exp(-50. * (p - t)))

    def __call__(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:

        feat_maps = self.feat_extractor(inputs)
        feat_concat = self.fpn(feat_maps)
        prob_map = self.probability_head(feat_concat)

        if training:
            thresh_map = self.threshold_head(feat_concat)
            approx_binmap = self.compute_approx_binmap(prob_map, thresh_map)
            return prob_map, thresh_map, approx_binmap

        else:
            return prob_map
