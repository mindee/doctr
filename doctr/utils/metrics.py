# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from typing import List, Tuple

__all__ = ['ExactMatch']


class ExactMatch:
    """Implements exact match metric (word-level accuracy) for recognition task

    Args:
        ignore_case: if true, ignore letter case when computing metric
        ignore_accents: if true, ignore accents errors when computing metrics"""

    def __init__(
        self,
        ignore_case: bool = False,
        ignore_accents: bool = False,
    ) -> None:

        self.matches = 0
        self.total = 0
        self.ignore_case = ignore_case
        self.ignore_accents = ignore_accents

    @staticmethod
    def remove_accent(input_string: str) -> str:
        """Removes all accents (¨^çéè...) from input_string

        Args:
            input_string: character sequence with accents

        Returns:
            character sequence without accents"""

        raise NotImplementedError

    def update(
        self,
        gt: List[str],
        pred: List[str],
    ) -> None:
        """Update the state of the metric with new predictions

        Args:
            gt: list of groung-truth character sequences
            pred: list of predicted character sequences"""

        if len(gt) != len(pred):
            raise AssertionError("prediction size does not match with ground-truth labels size")

        for pred_word, gt_word in zip(pred, gt):
            if self.ignore_accents:
                gt_word = remove_accent(gt_word)
                pred_word = remove_accent(pred_word)

            if self.ignore_case:
                gt_word = gt_word.lower()
                pred_word = pred_word.lower()

            if pred_word == gt_word:
                self.matches += 1

        self.total += len(gt)

    def summary(self) -> float:
        """Computes the aggregated evaluation

        Returns:
            metric result"""

        return self.matches / self.total


def compute_iou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Compute the IoU between two sets of bounding boxes

    Args:
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)

    Returns:
        the IoU matrix of shape (N, M)
    """

    return 0.


def assign_iou(iou_mat: np.ndarray, iou_threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    """Assigns boxes by IoU"""
    gt_kept = iou_mat.max(axis=1) >= iou_threshold
    _idxs = iou_mat.argmax(axis=1)
    assign_unique = np.unique(_idxs[gt_kept])
    # Filter
    if _idxs[gt_kept].shape[0] == assign_unique.shape[0]:
        return np.arange(iou_mat.shape[0])[gt_kept], _idxs[gt_kept]  # type: ignore[return-value]
    else:
        gt_indices, pred_indices = [], []
        for pred_idx in assign_unique:
            selection = iou.values[gt_kept][_idxs[gt_kept] == pred_idx].argmax()
            gt_indices.append(np.arange(iou_mat.shape[0])[gt_kept][selection].item())
            pred_indices.append(_idxs[gt_kept][selection].item())
        return gt_indices, pred_indices  # type: ignore[return-value]


class LocalizationConfusion:
    """Implements common confusion metrics and mean IoU for localization evaluation

    Args:
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
    """

    def __init__(self, iou_thresh: float = 0.5) -> None:

        self.iou_thresh = iou_thresh
        self.num_gts = 0
        self.num_preds = 0
        self.num_matches = 0
        self.tot_iou = 0.

    def update(self, gts: np.ndarray, preds: np.ndarray) -> None:

        # Compute IoU
        iou_mat = compute_iou(gts, preds)
        self.tot_iou += iou_mat.max(axis=1).sum()

        # Assign pairs
        gt_indices, pred_indices = assign_iou(iou_mat, self.iou_thresh)
        self.num_matches += len(gt_indices)

        # Update counts
        self.num_gts += gts.shape[0]
        self.num_preds += preds.shape[0]


    def result(self) -> Tuple[float, float]:

        # Recall
        recall = self.num_matches / self.num_gts

        # Precision
        precision = self.num_matches / self.num_preds

        # mean IoU
        mean_iou = self.tot_iou / self.num_preds

        return recall, precision, mean_iou