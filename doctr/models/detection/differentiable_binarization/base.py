# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from ..core import DetectionPostProcessor

__all__ = ["DBPostProcessor"]


class DBPostProcessor(DetectionPostProcessor):
    """Implements a post processor for DBNet adapted from the implementation of `xuannianz
    <https://github.com/xuannianz/DifferentiableBinarization>`_.

    Args:
    ----
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
        assume_straight_pages: bool = True,
    ) -> None:
        super().__init__(box_thresh, bin_thresh, assume_straight_pages)
        self.unclip_ratio = 1.5 if assume_straight_pages else 2.2

    def polygon_to_box(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
        ----
            points: The first parameter.

        Returns:
        -------
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
        """Compute boxes from a bitmap/pred_map

        Args:
        ----
            pred: Pred map from differentiable binarization output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
        -------
            np tensor boxes for the bitmap, each box is a 5-element list
                containing x, y, w, h, score for the box
        """
        height, width = bitmap.shape[:2]
        min_size_box = 1 + int(height / 512)
        boxes: List[Union[np.ndarray, List[float]]] = []
        # get contours from connected components on the bitmap
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Check whether smallest enclosing bounding box is not too small
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < min_size_box):
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

            # Remove too small boxes
            if self.assume_straight_pages:
                if _box is None or _box[2] < min_size_box or _box[3] < min_size_box:
                    continue
            elif np.linalg.norm(_box[2, :] - _box[0, :], axis=-1) < min_size_box:
                continue

            if self.assume_straight_pages:
                x, y, w, h = _box
                # compute relative polygon to get rid of img shape
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                # compute relative box to get rid of img shape, in that case _box is a 4pt polygon
                if not isinstance(_box, np.ndarray) and _box.shape == (4, 2):
                    raise AssertionError("When assume straight pages is false a box is a (4, 2) array (polygon)")
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(_box)

        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 4, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class _DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
    ----
        feature extractor: the backbone serving as feature extractor
        fpn_channels: number of channels each extracted feature maps is mapped to
    """

    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7
    min_size_box = 3
    assume_straight_pages: bool = True

    @staticmethod
    def compute_distance(
        xs: np.ndarray,
        ys: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        eps: float = 1e-7,
    ) -> float:
        """Compute the distance for each point of the map (xs, ys) to the (a, b) segment

        Args:
        ----
            xs : map of x coordinates (height, width)
            ys : map of y coordinates (height, width)
            a: first point defining the [ab] segment
            b: second point defining the [ab] segment
            eps: epsilon to avoid division by zero

        Returns:
        -------
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
        polygon: np.ndarray,
        canvas: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a polygon treshold map on a canvas, as described in the DB paper

        Args:
        ----
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
        padded_polygon: np.ndarray = np.array(padding.Execute(distance)[0])

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
        xs: np.ndarray = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys: np.ndarray = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

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
        canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1] = np.fmax(
            1 - distance_map[ymin_valid - ymin : ymax_valid - ymin + 1, xmin_valid - xmin : xmax_valid - xmin + 1],
            canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1],
        )

        return polygon, canvas, mask

    def build_target(
        self,
        target: List[Dict[str, np.ndarray]],
        output_shape: Tuple[int, int, int, int],
        channels_last: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")

        input_dtype = next(iter(target[0].values())).dtype if len(target) > 0 else np.float32

        if channels_last:
            h, w = output_shape[1:-1]
            target_shape = (output_shape[0], output_shape[-1], h, w)  # (Batch_size, num_classes, h, w)
        else:
            h, w = output_shape[-2:]
            target_shape = output_shape  # (Batch_size, num_classes, h, w)
        seg_target: np.ndarray = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: np.ndarray = np.ones(target_shape, dtype=bool)
        thresh_target: np.ndarray = np.zeros(target_shape, dtype=np.float32)
        thresh_mask: np.ndarray = np.ones(target_shape, dtype=np.uint8)

        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                # Draw each polygon on gt
                if _tgt.shape[0] == 0:
                    # Empty image, full masked
                    # seg_mask[idx, :, :, class_idx] = False
                    seg_mask[idx, class_idx] = False

                # Absolute bounding boxes
                abs_boxes = _tgt.copy()
                if abs_boxes.ndim == 3:
                    abs_boxes[:, :, 0] *= w
                    abs_boxes[:, :, 1] *= h
                    polys = abs_boxes
                    boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                    abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
                else:
                    abs_boxes[:, [0, 2]] *= w
                    abs_boxes[:, [1, 3]] *= h
                    abs_boxes = abs_boxes.round().astype(np.int32)
                    polys = np.stack(
                        [
                            abs_boxes[:, [0, 1]],
                            abs_boxes[:, [0, 3]],
                            abs_boxes[:, [2, 3]],
                            abs_boxes[:, [2, 1]],
                        ],
                        axis=1,
                    )
                    boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])

                for box, box_size, poly in zip(abs_boxes, boxes_size, polys):
                    # Mask boxes that are too small
                    if box_size < self.min_size_box:
                        # seg_mask[idx, box[1] : box[3] + 1, box[0] : box[2] + 1, class_idx] = False
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
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
                        # seg_mask[idx, box[1] : box[3] + 1, box[0] : box[2] + 1, class_idx] = False
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
                        continue
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] <= 2 or not Polygon(shrinked).is_valid:
                        # seg_mask[idx, box[1] : box[3] + 1, box[0] : box[2] + 1, class_idx] = False
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
                        continue
                    cv2.fillPoly(seg_target[idx, class_idx], [shrinked.astype(np.int32)], 1)

                    # Draw on both thresh map and thresh mask
                    poly, thresh_target[idx, class_idx], thresh_mask[idx, class_idx] = self.draw_thresh_map(
                        poly, thresh_target[idx, class_idx], thresh_mask[idx, class_idx]
                    )
        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))
            thresh_target = thresh_target.transpose((0, 2, 3, 1))
            thresh_mask = thresh_mask.transpose((0, 2, 3, 1))

        thresh_target = thresh_target.astype(input_dtype) * (self.thresh_max - self.thresh_min) + self.thresh_min

        seg_target = seg_target.astype(input_dtype)
        seg_mask = seg_mask.astype(bool)
        thresh_target = thresh_target.astype(input_dtype)
        thresh_mask = thresh_mask.astype(bool)

        return seg_target, seg_mask, thresh_target, thresh_mask
