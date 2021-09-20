# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
from typing import List, Tuple

from doctr.utils.repr import NestedObject
from doctr.utils.geometry import rotate_image
from .._utils import get_bitmap_angle


__all__ = ['DetectionPostProcessor']


class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        min_size_box (int): minimal length (pix) to keep a box
        max_candidates (int): maximum boxes to consider in a single page
        box_thresh (float): minimal objectness score to consider a box
    """

    def __init__(
        self,
        box_thresh: float = 0.5,
        bin_thresh: float = 0.5,
        rotated_bbox: bool = False
    ) -> None:

        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.rotated_bbox = rotated_bbox

    def extra_repr(self) -> str:
        return f"box_thresh={self.box_thresh}"

    @staticmethod
    def box_score(
        pred: np.ndarray,
        points: np.ndarray,
        rotated_bbox: bool = False
    ) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]

        if not rotated_bbox:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin:ymax + 1, xmin:xmax + 1].mean()

        else:
            mask = np.zeros((h, w), np.int32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)
            product = pred * mask
            return np.sum(product) / np.count_nonzero(product)

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        proba_map: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Performs postprocessing for a list of model outputs

        Args:
            proba_map: probability map of shape (N, H, W)

        returns:
            list of N tensors (for each input sample), with each tensor of shape (*, 5) or (*, 6),
            and a list of N angles (page orientations).
        """

        bitmap = (proba_map > self.bin_thresh).astype(proba_map.dtype)

        boxes_batch, angles_batch = [], []
        # Kernel for opening, empirical law for ksize
        k_size = 1 + int(proba_map[0].shape[0] / 512)
        kernel = np.ones((k_size, k_size), np.uint8)

        for p_, bitmap_ in zip(proba_map, bitmap):
            # Perform opening (erosion + dilatation)
            bitmap_ = cv2.morphologyEx(bitmap_, cv2.MORPH_OPEN, kernel)
            # Rotate bitmap and proba_map
            angle = get_bitmap_angle(bitmap_)
            angles_batch.append(angle)
            bitmap_, p_ = rotate_image(bitmap_, -angle, False), rotate_image(p_, -angle, False)
            boxes = self.bitmap_to_boxes(pred=p_, bitmap=bitmap_)
            boxes_batch.append(boxes)

        return boxes_batch, angles_batch
