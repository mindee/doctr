# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from doctr.models.core import BaseModel

from ..core import DetectionPostProcessor

__all__ = ["_LinkNet", "LinkNetPostProcessor"]


class LinkNetPostProcessor(DetectionPostProcessor):
    """Implements a post processor for LinkNet model.

    Args:
    ----
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
        """Compute boxes from a bitmap/pred_map: find connected components then filter boxes

        Args:
        ----
            pred: Pred map from differentiable linknet output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
        -------
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


class _LinkNet(BaseModel):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
    ----
        out_chan: number of channels for the output
    """

    min_size_box: int = 3
    assume_straight_pages: bool = True
    shrink_ratio = 0.5

    def build_target(
        self,
        target: List[Dict[str, np.ndarray]],
        output_shape: Tuple[int, int, int],
        channels_last: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the target, and it's mask to be used from loss computation.

        Args:
        ----
            target: target coming from dataset
            output_shape: shape of the output of the model without batch_size
            channels_last: whether channels are last or not

        Returns:
        -------
            the new formatted target and the mask
        """
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")

        h: int
        w: int
        if channels_last:
            h, w, num_classes = output_shape
        else:
            num_classes, h, w = output_shape
        target_shape = (len(target), num_classes, h, w)

        seg_target: np.ndarray = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: np.ndarray = np.ones(target_shape, dtype=bool)

        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                # Draw each polygon on gt
                if _tgt.shape[0] == 0:
                    # Empty image, full masked
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

                for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                    # Mask boxes that are too small
                    if box_size < self.min_size_box:
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
                        continue

                    # Negative shrink for gt, as described in paper
                    polygon = Polygon(poly)
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                    subject = [tuple(coor) for coor in poly]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrunken = padding.Execute(-distance)

                    # Draw polygon on gt if it is valid
                    if len(shrunken) == 0:
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
                        continue
                    shrunken = np.array(shrunken[0]).reshape(-1, 2)
                    if shrunken.shape[0] <= 2 or not Polygon(shrunken).is_valid:
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
                        continue
                    cv2.fillPoly(seg_target[idx, class_idx], [shrunken.astype(np.int32)], 1)

        # Don't forget to switch back to channel last if Tensorflow is used
        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))

        return seg_target, seg_mask
