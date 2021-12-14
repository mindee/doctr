# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from typing import List, Tuple

import cv2
import numpy as np

from doctr.file_utils import is_tf_available
from doctr.models.core import BaseModel
from doctr.utils.geometry import fit_rbbox, rbbox_to_polygon

from ..core import DetectionPostProcessor

__all__ = ['_LinkNet', 'LinkNetPostProcessor']


class LinkNetPostProcessor(DetectionPostProcessor):
    """Implements a post processor for LinkNet model.

    Args:
        min_size_box: minimal length (pix) to keep a box
        box_thresh: minimal objectness score to consider a box
        bin_thresh: threshold used to binzarized p_map at inference time

    """
    def __init__(
        self,
        bin_thresh: float = 0.15,
        box_thresh: float = 0.1,
        assume_straight_pages: bool = True,
    ) -> None:
        super().__init__(
            box_thresh,
            bin_thresh,
            assume_straight_pages
        )

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
        angle_tol: float = 5.,
        ratio_tol: float = 2.,
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
        min_size_box = 1 + int(height / 512)
        boxes = []
        # get contours from connected components on the bitmap
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Check whether smallest enclosing bounding box is not too small
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < min_size_box):
                continue
            # Compute objectness
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)

            if score < self.box_thresh:   # remove polygons with a weak objectness
                continue

            if self.assume_straight_pages:
                # compute relative polygon to get rid of img shape
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                x, y, w, h, alpha = fit_rbbox(contour)
                # compute relative box to get rid of img shape
                x, y, w, h = x / width, y / height, w / width, h / height
                boxes.append([x, y, w, h, alpha, score])

        if not self.assume_straight_pages:
            # Compute median angle and mean aspect ratio to resolve quadrants
            np_boxes = np.asarray(boxes, dtype=np.float32)
            median_angle = np.median(np_boxes[:, -2])
            median_w, median_h = np.median(np_boxes[:, 2]), np.median(np_boxes[:, 3])

            # Rectify angles
            new_boxes = []
            for x, y, w, h, alpha, score in boxes:
                if median_h >= median_w:
                    # We are in the upper quadrant
                    if 1 / ratio_tol < h / w < ratio_tol:
                        # If a box has an aspect ratio close to 1 (cubic), we set the angle to the median angle
                        _rbox = [x, y, h, w, 90 + median_angle, score]
                    elif abs(90 - abs(alpha) - abs(median_angle)) <= angle_tol:
                        # We jumped to the next quadrant, rectify
                        _rbox = [x, y, w, h, alpha, score]
                    else:
                        _rbox = [x, y, h, w, 90 + alpha, score]
                else:
                    # We are in the lower quadrant.
                    if 1 / ratio_tol < h / w < ratio_tol:
                        # If a box has an aspect ratio close to 1 (cubic), we set the angle to the median angle
                        _rbox = [x, y, w, h, median_angle, score]
                    elif abs(90 - abs(alpha) - abs(median_angle)) <= angle_tol:
                        # We jumped to the next quadrant, rectify
                        _rbox = [x, y, h, w, 90 + alpha, score]
                    else:
                        _rbox = [x, y, w, h, alpha, score]
                new_boxes.append(_rbox)
            boxes = new_boxes

        if not self.assume_straight_pages:
            if len(boxes) == 0:
                return np.zeros((0, 6), dtype=pred.dtype)
            coord = np.clip(np.asarray(boxes)[:, :4], 0, 1)  # clip boxes coordinates
            boxes = np.concatenate((coord, np.asarray(boxes)[:, 4:]), axis=1)
            return boxes
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class _LinkNet(BaseModel):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
        out_chan: number of channels for the output
    """

    min_size_box: int = 3
    assume_straight_pages: bool = True

    def build_target(
        self,
        target: List[np.ndarray],
        output_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if any(t.dtype != np.float32 for t in target):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for t in target):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")

        h, w = output_shape
        target_shape = (len(target), h, w, 1)

        if self.assume_straight_pages:
            seg_target = np.zeros(target_shape, dtype=bool)
            edge_mask = np.zeros(target_shape, dtype=bool)
        else:
            seg_target = np.zeros(target_shape, dtype=np.uint8)

        seg_mask = np.ones(target_shape, dtype=bool)

        for idx, _target in enumerate(target):
            # Draw each polygon on gt
            if _target.shape[0] == 0:
                # Empty image, full masked
                seg_mask[idx] = False

            # Absolute bounding boxes
            abs_boxes = _target.copy()
            abs_boxes[:, [0, 2]] *= w
            abs_boxes[:, [1, 3]] *= h
            abs_boxes = abs_boxes.round().astype(np.int32)

            # Convert to polygons --> (N, 4, 2)
            if abs_boxes.shape[1] == 5:
                boxes_size = np.minimum(abs_boxes[:, 2], abs_boxes[:, 3])
                polys = np.stack([
                    rbbox_to_polygon(tuple(rbbox)) for rbbox in abs_boxes  # type: ignore[arg-type]
                ], axis=0)
            else:
                boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])
                polys = [None] * abs_boxes.shape[0]  # Unused

            for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                # Mask boxes that are too small
                if box_size < self.min_size_box:
                    seg_mask[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue
                # Fill polygon with 1
                if not self.assume_straight_pages:
                    cv2.fillPoly(seg_target[idx], [poly.astype(np.int32)], 1)
                else:
                    seg_target[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = True
                    # top edge
                    edge_mask[idx, box[1], box[0]: min(box[2] + 1, w)] = True
                    # bot edge
                    edge_mask[idx, min(box[3], h - 1), box[0]: min(box[2] + 1, w)] = True
                    # left edge
                    edge_mask[idx, box[1]: min(box[3] + 1, h), box[0]] = True
                    # right edge
                    edge_mask[idx, box[1]: min(box[3] + 1, h), min(box[2], w - 1)] = True

        # Don't forget to switch back to channel first if PyTorch is used
        if not is_tf_available():
            seg_target = seg_target.transpose(0, 3, 1, 2)
            seg_mask = seg_mask.transpose(0, 3, 1, 2)
            edge_mask = edge_mask.transpose(0, 3, 1, 2)

        return seg_target, seg_mask, edge_mask
