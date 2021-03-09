# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from textdistance import levenshtein
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment

__all__ = ['ExactMatch', 'box_iou', 'assign_pairs', 'LocalizationConfusion', 'OCRMetric']


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
                gt_word = self.remove_accent(gt_word)
                pred_word = self.remove_accent(pred_word)

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
        if self.total == 0:
            raise AssertionError("you need to update the metric before getting the summary")
        return self.matches / self.total


def box_iou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Compute the IoU between two sets of bounding boxes

    Args:
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)

    Returns:
        the IoU matrix of shape (N, M)
    """

    l1, t1, r1, b1 = np.split(boxes_1, 4, axis=1)
    l2, t2, r2, b2 = np.split(boxes_2, 4, axis=1)

    left = np.maximum(l1, l2.T)
    top = np.maximum(t1, t2.T)
    right = np.minimum(r1, r2.T)
    bot = np.minimum(b1, b2.T)

    intersection = np.abs(right - left) * np.abs(bot - top)

    union = (r1 - l1) * (b1 - t1) + ((r2 - l2) * (b2 - t2)).T - intersection

    return intersection / union


def assign_pairs(score_mat: np.ndarray, score_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Assigns candidates by maximizing the score of all pairs

    Args:
        score_mat: score matrix
        score_threshold: minimum score to validate an assignment
    Returns:
        a tuple of two lists: the list of assigned row candidates indices, and the list of their column counterparts
    """

    row_ind, col_ind = linear_sum_assignment(-score_mat)
    is_kept = score_mat[row_ind, col_ind] >= score_threshold
    return row_ind[is_kept], col_ind[is_kept]


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
        iou_mat = box_iou(gts, preds)
        self.tot_iou += float(iou_mat.max(axis=1).sum())

        # Assign pairs
        gt_indices, _ = assign_pairs(iou_mat, self.iou_thresh)
        self.num_matches += len(gt_indices)

        # Update counts
        self.num_gts += gts.shape[0]
        self.num_preds += preds.shape[0]

    def summary(self) -> Tuple[float, float, float]:

        # Recall
        recall = self.num_matches / self.num_gts

        # Precision
        precision = self.num_matches / self.num_preds

        # mean IoU
        mean_iou = self.tot_iou / self.num_preds

        return recall, precision, mean_iou


class OCRMetric:
    """Implements end-to-end OCR metric

    Args:
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
        max_dist: maximum Levenshtein distance between 2 sequence to consider a match
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        max_dist: int = 0
    ) -> None:

        self.iou_thresh = iou_thresh
        self.max_dist = max_dist
        self.num_gts = 0
        self.num_preds = 0
        self.num_det_matches = 0
        self.num_reco_matches = 0
        self.tot_iou = 0.
        self.tot_dist = 0

    def update(
        self,
        gts_vertices: np.ndarray,
        preds_vertices: np.ndarray,
        gts_texts: List[str],
        preds_texts: List[str],
    ) -> None:

        # Compute IoU
        iou_mat = box_iou(gts_vertices, preds_vertices)
        self.tot_iou += float(iou_mat.max(axis=1).sum())

        # Assign pairs
        gt_indices, preds_indices = assign_pairs(iou_mat, self.iou_thresh)

        # Compare sequences
        for gt_idx, pred_idx in zip(gt_indices, preds_indices):
            dist = levenshtein.distance(gts_texts[gt_idx], preds_texts[pred_idx])
            self.tot_dist += dist
            if dist <= self.max_dist:
                self.num_reco_matches += 1

        # Update counts
        self.num_det_matches = len(gt_indices)
        self.num_gts += gts_vertices.shape[0]
        self.num_preds += preds_vertices.shape[0]

    def summary(self) -> Tuple[float, float, float, float]:

        # Recall
        recall = self.num_reco_matches / self.num_gts

        # Precision
        precision = self.num_reco_matches / self.num_preds

        # mean IoU (overall detected boxes)
        mean_iou = self.tot_iou / self.num_preds

        # mean distance (overall detection-matching boxes)
        mean_distance = self.tot_dist / self.num_det_matches

        return recall, precision, mean_iou, mean_distance
