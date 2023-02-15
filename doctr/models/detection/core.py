# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

import cv2
import numpy as np

from doctr.utils.repr import NestedObject

__all__ = ["DetectionPostProcessor"]


class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        box_thresh (float): minimal objectness score to consider a box
        bin_thresh (float): threshold to apply to segmentation raw heatmap
        assume straight_pages (bool): if True, fit straight boxes only
    """

    def __init__(self, box_thresh: float = 0.5, bin_thresh: float = 0.5, assume_straight_pages: bool = True) -> None:
        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.assume_straight_pages = assume_straight_pages
        self._opening_kernel: np.ndarray = np.ones((3, 3), dtype=np.uint8)

    def extra_repr(self) -> str:
        return f"bin_thresh={self.bin_thresh}, box_thresh={self.box_thresh}"

    @staticmethod
    def box_score(pred: np.ndarray, points: np.ndarray, assume_straight_pages: bool = True) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]

        if assume_straight_pages:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin : ymax + 1, xmin : xmax + 1].mean()

        else:
            mask: np.ndarray = np.zeros((h, w), np.int32)
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
        proba_map,
    ) -> List[List[np.ndarray]]:
        """Performs postprocessing for a list of model outputs

        Args:
            proba_map: probability map of shape (N, H, W, C)

        Returns:
            list of N class predictions (for each input sample), where each class predictions is a list of C tensors
        of shape (*, 5) or (*, 6)
        """

        if proba_map.ndim != 4:
            raise AssertionError(f"arg `proba_map` is expected to be 4-dimensional, got {proba_map.ndim}.")

        # Erosion + dilation on the binary map
        bin_map = [
            [
                cv2.morphologyEx(bmap[..., idx], cv2.MORPH_OPEN, self._opening_kernel)
                for idx in range(proba_map.shape[-1])
            ]
            for bmap in (proba_map >= self.bin_thresh).astype(np.uint8)
        ]

        return [
            [self.bitmap_to_boxes(pmaps[..., idx], bmaps[idx]) for idx in range(proba_map.shape[-1])]
            for pmaps, bmaps in zip(proba_map, bin_map)
        ]
