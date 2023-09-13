# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from typing import List, Union

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from ..core import DetectionPostProcessor

__all__ = ["FastPostProcessor"]


class FastPostProcessor(DetectionPostProcessor):
    """Implements a post processor for LinkNet model.

    Args:
        bin_thresh: threshold used to binzarized p_map at inference time
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: whether the inputs were expected to have horizontal text elements
    """

    def __init__(
        self,
        bin_thresh: float = 0.1,
        box_thresh: float = 0.1,
        assume_straight_pages: bool = True,
    ) -> None:
        super().__init__(box_thresh, bin_thresh, assume_straight_pages)
        self.unclip_ratio = 1.2

    def polygon_to_box(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
            points: The first parameter.

        Returns:
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (4, 2) array (quadrangle)
        """
        if not self.assume_straight_pages:
            # Compute the rectangle polygon enclosing the raw polygon
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            # Add 1 pixel to correct cv2 approx
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            poly = Polygon(points)
            area = poly.area
            length = poly.length
        distance = area * self.unclip_ratio / length  # compute distance to expand polygon
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        # Take biggest stack of points
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            # We ensure that _points can be correctly casted to a ndarray
            _points = [_points[idx]]
        expanded_points: np.ndarray = np.asarray(_points)  # expand polygon
        if len(expanded_points) < 1:
            return None  # type: ignore[return-value]
        return (
            cv2.boundingRect(expanded_points)
            if self.assume_straight_pages
            else np.roll(cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0)
        )

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> np.ndarray:
        """Compute boxes from a bitmap/pred_map: find connected components then filter boxes

        Args:
            pred: Pred map from differentiable linknet output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
            np tensor boxes for the bitmap, each box is a 6-element list
                containing x, y, w, h, alpha, score for the box
        """
        height, width = bitmap.shape[:2]
        boxes: List[Union[np.ndarray, List[float]]] = []
        # get contours from connected components on the bitmap
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Check whether smallest enclosing bounding box is not too small
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < 2):
                continue
            # Compute objectness
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points: np.ndarray = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)

            if score < self.box_thresh:  # remove polygons with a weak objectness
                continue

            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
            else:
                _box = self.polygon_to_box(np.squeeze(contour))

            if self.assume_straight_pages:
                # compute relative polygon to get rid of img shape
                x, y, w, h = _box
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                # compute relative box to get rid of img shape
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(_box)

        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 4, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)
