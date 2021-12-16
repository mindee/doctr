# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List

import cv2
import numpy as np

from doctr.file_utils import is_tf_available
from doctr.utils.repr import NestedObject

from ._utils import generate_bin_map

__all__ = ['DetectionPostProcessor']


class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        box_thresh (float): minimal objectness score to consider a box
        bin_thresh (float): threshold to apply to segmentation raw heatmap
        assume straight_pages (bool): if True, fit straight boxes only
    """

    def __init__(
        self,
        box_thresh: float = 0.5,
        bin_thresh: float = 0.5,
        assume_straight_pages: bool = True
    ) -> None:

        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.assume_straight_pages = assume_straight_pages

    def extra_repr(self) -> str:
        return f"box_thresh={self.box_thresh}"

    @staticmethod
    def box_score(
        pred: np.ndarray,
        points: np.ndarray,
        assume_straight_pages: bool = True
    ) -> float:
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

    def locate_boxes(self, prob_map: np.ndarray, bin_map: np.ndarray) -> np.ndarray:
        """Locate bounding boxes in predicted maps

        Args:
            prob_map: prob map of shape (C, H, W)
            bin_map: bin map of shape (C, H, W)

        Returns:
            box tensors of shape (*, 5) or (*, 6)
        """

        # Assume there is only 1 output class
        return self.bitmap_to_boxes(np.squeeze(prob_map, 0), np.squeeze(bin_map, 0))

    def __call__(
        self,
        proba_map,
    ) -> List[np.ndarray]:
        """Performs postprocessing for a list of model outputs

        Args:
            proba_map: probability map of shape (N, C, H, W) for PyTorch or (N, H, W, C) for TensorFlow

        returns:
            list of N tensors (for each input sample), with each tensor of shape (*, 5) or (*, 6)
        """

        # Erosion + dilation on the binary map
        bin_map = generate_bin_map(proba_map, self.bin_thresh)
        # (N, C, H, W)
        bin_map = bin_map.numpy().transpose(0, 3, 1, 2) if is_tf_available() else bin_map.cpu().numpy()
        proba_map = proba_map.numpy().transpose(0, 3, 1, 2) if is_tf_available() else proba_map.cpu().numpy()

        return [self.locate_boxes(pmap, bmap) for pmap, bmap in zip(proba_map, bin_map)]
