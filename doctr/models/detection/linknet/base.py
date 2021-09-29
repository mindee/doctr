# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

import numpy as np
import cv2
from typing import Tuple, List

from doctr.utils.geometry import fit_rbbox, rbbox_to_polygon
from doctr.models.core import BaseModel
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
        rotated_bbox: bool = False,
    ) -> None:
        super().__init__(
            box_thresh,
            bin_thresh,
            rotated_bbox
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
            if self.rotated_bbox:
                score = self.box_score(pred, contour, rotated_bbox=True)
            else:
                x, y, w, h = cv2.boundingRect(contour)
                points = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, rotated_bbox=False)

            if self.box_thresh > score:   # remove polygons with a weak objectness
                continue

            if self.rotated_bbox:
                x, y, w, h, alpha = fit_rbbox(contour)
                # compute relative box to get rid of img shape
                x, y, w, h = x / width, y / height, w / width, h / height
                boxes.append([x, y, w, h, alpha, score])
            else:
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


class _LinkNet(BaseModel):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
        out_chan: number of channels for the output
    """

    min_size_box: int = 3
    rotated_bbox: bool = False

    def compute_target(
        self,
        target: List[np.ndarray],
        output_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if any(t.dtype not in (np.float32, np.float16) for t in target):
            raise AssertionError("the expected dtype of target 'boxes' entry is either 'np.float32' or 'np.float16'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for t in target):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")

        if self.rotated_bbox:
            seg_target = np.zeros(output_shape, dtype=np.uint8)
        else:
            seg_target = np.zeros(output_shape, dtype=bool)
            edge_mask = np.zeros(output_shape, dtype=bool)
        seg_mask = np.ones(output_shape, dtype=bool)

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
                polys = [None] * abs_boxes.shape[0]  # Unused

            for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                # Mask boxes that are too small
                if box_size < self.min_size_box:
                    seg_mask[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue
                # Fill polygon with 1
                if self.rotated_bbox:
                    cv2.fillPoly(seg_target[idx], [poly.astype(np.int32)], 1)
                else:
                    seg_target[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = True
                    # fill the 2 vertical edges
                    edge_mask[idx, max(0, box[1] - 1): min(box[1] + 1, box[3]), box[0]: box[2] + 1] = True
                    edge_mask[idx, max(box[1] + 1, box[3]): min(output_shape[1], box[3] + 2), box[0]: box[2] + 1] = True
                    # fill the 2 horizontal edges
                    edge_mask[idx, box[1]: box[3] + 1, max(0, box[0] - 1): min(box[0] + 1, box[2])] = True
                    edge_mask[idx, box[1]: box[3] + 1, max(box[0] + 1, box[2]): min(output_shape[2], box[2] + 2)] = True

        return seg_target, seg_mask, edge_mask
