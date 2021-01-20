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
from typing import Union, List, Tuple, Optional, Any, Dict

from doctr.models.detection.postprocessor import Postprocessor

__all__ = ['DBPostprocessor']


class DBPostprocessor(Postprocessor):
    """Class to postprocess Differentiable binzarization model outputs
    Unherits from Postprocessor

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
