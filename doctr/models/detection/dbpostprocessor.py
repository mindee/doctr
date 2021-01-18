# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
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

    def box_score(
        self,
        pred: np.ndarray,
        _box: np.ndarray
    ) -> float:
        """
        Compute the confidence score for a box : mean between p_map values on the box
        :param pred: p_map (output of the model)
        :param _box: box
        """

        h, w = pred.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

        return cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def polygon_to_box(
        self,
        points: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """
        Expand polygon (box) by a factor unclip_ratio
        :param poly: polygon to unclip
        :param unclip_ratio: dilatation ratio
        returns absolutes boxes
        """

        poly = Polygon(points)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded_points = np.array(offset.Execute(distance))
        x, y, w, h = cv2.boundingRect(expanded_points)
        return x, y, w, h

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> List[List[float]]:
        """
        predict scores and boxes from p_map and bin_map
        :param pred : probability map (np array)
        :param bitmap: bin_map (generated from p_map with a constant threshold at inference time), np array
        :param max candidates: max boxes to look for in a document page
        :param box_thresh: min score to consider a box
        """

        height, width = bitmap.shape[:2]
        boxes = []
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            x, y, w, h = self.polygon_to_box(points)
            if h < self.min_size_box or w < self.min_size_box:
                continue
            x = x / width
            y = y / height
            w = w / width
            h = h / height
            boxes.append([x, y, w, h, score])
        return boxes

    def __call__(
        self,
        raw_pred: List[tf.Tensor],
    ) -> List[List[np.ndarray]]:
        """
        postprocessing function which convert the model output to boxes and scores
        :param raw_pred: raw outputs of the differentiable binarization model, list of batches
        output : list of batches, 1 batch = list of np tensors, len = batch_size,
        each tensor of size num_boxes X 5 (x, y, w, h, score)
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
                boxes = self.bitmap_to_boxes(p_, bitmap_)
                boxes_batch.append(np.array(boxes))

            bounding_boxes.append(boxes_batch)
        return bounding_boxes
