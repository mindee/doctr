# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper
from typing import Union, List, Tuple, Optional

from doctr.utils.geometry import fit_rbbox, rbbox_to_polygon
from doctr.utils.common_types import RotatedBbox
from ..core import DetectionPostProcessor

__all__ = ['DBPostProcessor']


class DBPostProcessor(DetectionPostProcessor):
    """Implements a post processor for DBNet adapted from the implementation of `xuannianz
    <https://github.com/xuannianz/DifferentiableBinarization>`_.

    Args:
        unclip ratio: ratio used to unshrink polygons
        min_size_box: minimal length (pix) to keep a box
        max_candidates: maximum boxes to consider in a single page
        box_thresh: minimal objectness score to consider a box
        bin_thresh: threshold used to binzarized p_map at inference time

    """
    def __init__(
        self,
        box_thresh: float = 0.1,
        bin_thresh: float = 0.3,
        rotated_bbox: bool = False,
    ) -> None:

        super().__init__(
            box_thresh,
            bin_thresh,
            rotated_bbox
        )
        self.unclip_ratio = 2.2 if self.rotated_bbox else 1.5

    def polygon_to_box(
        self,
        points: np.ndarray,
    ) -> Optional[Union[RotatedBbox, Tuple[float, float, float, float]]]:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a rotated box: x, y, w, h, alpha

        Args:
            points: The first parameter.

        Returns:
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (x, y, w, h, alpha)
        """
        poly = Polygon(points)
        distance = poly.area * self.unclip_ratio / poly.length  # compute distance to expand polygon
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
        expanded_points = np.asarray(_points)  # expand polygon
        if len(expanded_points) < 1:
            return None
        return fit_rbbox(expanded_points) if self.rotated_bbox else cv2.boundingRect(expanded_points)

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> np.ndarray:
        """Compute boxes from a bitmap/pred_map

        Args:
            pred: Pred map from differentiable binarization output
            bitmap: Bitmap map computed from pred (binarized)

        Returns:
            np tensor boxes for the bitmap, each box is a 5-element list
                containing x, y, w, h, score for the box
        """
        height, width = bitmap.shape[:2]
        min_size_box = 1 + int(height / 512)
        boxes = []
        # get contours from connected components on the bitmap
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Check whether smallest enclosing bounding box is not too small
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < min_size_box):
                continue
            # Compute objectness
            if self.rotated_bbox:
                score = self.box_score(pred, contour, rotated_bbox=True)
            else:
                x, y, w, h = cv2.boundingRect(contour)
                points = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, rotated_bbox=False)

            if self.box_thresh > score:   # remove polygons with a weak objectness
                continue

            _box = self.polygon_to_box(np.squeeze(contour)) if self.rotated_bbox else self.polygon_to_box(points)

            if _box is None or _box[2] < min_size_box or _box[3] < min_size_box:  # remove to small boxes
                continue

            if self.rotated_bbox:
                x, y, w, h, alpha = _box  # type: ignore[misc]
                # compute relative box to get rid of img shape
                x, y, w, h = x / width, y / height, w / width, h / height
                boxes.append([x, y, w, h, alpha, score])
            else:
                x, y, w, h = _box  # type: ignore[misc]
                # compute relative polygon to get rid of img shape
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])

        if self.rotated_bbox:
            if len(boxes) == 0:
                return np.zeros((0, 6), dtype=pred.dtype)
            coord = np.clip(np.asarray(boxes)[:, :4], 0, 1)  # clip boxes coordinates
            boxes = np.concatenate((coord, np.asarray(boxes)[:, 4:]), axis=1)
            return boxes
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class _DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        fpn_channels: number of channels each extracted feature maps is mapped to
    """

    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7
    min_size_box = 3
    rotated_bbox: bool = False

    @staticmethod
    def compute_distance(
        xs: np.array,
        ys: np.array,
        a: np.array,
        b: np.array,
        eps: float = 1e-7,
    ) -> float:
        """Compute the distance for each point of the map (xs, ys) to the (a, b) segment

        Args:
            xs : map of x coordinates (height, width)
            ys : map of y coordinates (height, width)
            a: first point defining the [ab] segment
            b: second point defining the [ab] segment

        Returns:
            The computed distance

        """
        square_dist_1 = np.square(xs - a[0]) + np.square(ys - a[1])
        square_dist_2 = np.square(xs - b[0]) + np.square(ys - b[1])
        square_dist = np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        cosin = (square_dist - square_dist_1 - square_dist_2) / (2 * np.sqrt(square_dist_1 * square_dist_2) + eps)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_dist_1 * square_dist_2 * square_sin / square_dist)
        result[cosin < 0] = np.sqrt(np.fmin(square_dist_1, square_dist_2))[cosin < 0]
        return result

    def draw_thresh_map(
        self,
        polygon: np.array,
        canvas: np.array,
        mask: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a polygon treshold map on a canvas, as described in the DB paper

        Args:
            polygon : array of coord., to draw the boundary of the polygon
            canvas : threshold map to fill with polygons
            mask : mask for training on threshold polygons
        """
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise AttributeError("polygon should be a 2 dimensional array of coords")

        # Augment polygon by shrink_ratio
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(coor) for coor in polygon]  # Get coord as list of tuples
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])

        # Fill the mask with 1 on the new padded polygon
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        # Get min/max to recover polygon after distance computation
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        # Get absolute polygon for distance computation
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        # Get absolute padded polygon
        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        # Compute distance map to fill the padded polygon
        distance_map = np.zeros((polygon.shape[0], height, width), dtype=polygon.dtype)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)

        # Clip the padded polygon inside the canvas
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        # Fill the canvas with the distances computed inside the valid padded polygon
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymin + 1,
                xmin_valid - xmin:xmax_valid - xmin + 1
            ],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1]
        )

        return polygon, canvas, mask

    def compute_target(
        self,
        target: List[np.ndarray],
        output_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if any(t.dtype not in (np.float32, np.float16) for t in target):
            raise AssertionError("the expected dtype of target 'boxes' entry is either 'np.float32' or 'np.float16'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for t in target):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")

        input_dtype = target[0].dtype if len(target) > 0 else np.float32

        seg_target = np.zeros(output_shape, dtype=np.uint8)
        seg_mask = np.ones(output_shape, dtype=bool)
        thresh_target = np.zeros(output_shape, dtype=np.uint8)
        thresh_mask = np.ones(output_shape, dtype=np.uint8)

        for idx, _target in enumerate(target):
            # Draw each polygon on gt
            if _target.shape[0] == 0:
                # Empty image, full masked
                seg_mask[idx] = False

            # Absolute bounding boxes
            abs_boxes = _target.copy()
            abs_boxes[:, [0, 2]] *= output_shape[-1]
            abs_boxes[:, [1, 3]] *= output_shape[-2]
            abs_boxes = abs_boxes.round().astype(np.int32)

            if abs_boxes.shape[1] == 5:
                boxes_size = np.minimum(abs_boxes[:, 2], abs_boxes[:, 3])
                polys = np.stack([
                    rbbox_to_polygon(tuple(rbbox)) for rbbox in abs_boxes  # type: ignore[arg-type]
                ], axis=1)
            else:
                boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])
                polys = np.stack([
                    abs_boxes[:, [0, 1]],
                    abs_boxes[:, [0, 3]],
                    abs_boxes[:, [2, 3]],
                    abs_boxes[:, [2, 1]],
                ], axis=1)

            for box, box_size, poly in zip(abs_boxes, boxes_size, polys):
                # Mask boxes that are too small
                if box_size < self.min_size_box:
                    seg_mask[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue

                # Negative shrink for gt, as described in paper
                polygon = Polygon(poly)
                distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(coor) for coor in poly]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                # Draw polygon on gt if it is valid
                if len(shrinked) == 0:
                    seg_mask[box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                if shrinked.shape[0] <= 2 or not Polygon(shrinked).is_valid:
                    seg_mask[box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue
                cv2.fillPoly(seg_target[idx], [shrinked.astype(np.int32)], 1)

                # Draw on both thresh map and thresh mask
                poly, thresh_target[idx], thresh_mask[idx] = self.draw_thresh_map(poly, thresh_target[idx],
                                                                                  thresh_mask[idx])

        thresh_target = thresh_target.astype(input_dtype) * (self.thresh_max - self.thresh_min) + self.thresh_min

        seg_target = seg_target.astype(input_dtype)
        seg_mask = seg_mask.astype(bool)
        thresh_target = thresh_target.astype(input_dtype)
        thresh_mask = thresh_mask.astype(bool)

        return seg_target, seg_mask, thresh_target, thresh_mask
