# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from rapidfuzz.string_metric import levenshtein
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment

__all__ = ['ExactMatch', 'box_iou', 'assign_pairs', 'LocalizationConfusion', 'OCRMetric']


class ExactMatch:
    """Implements exact match metric (word-level accuracy) for recognition task.

    The aggregated metric is computed as follows:

    .. math::
        \\forall X, Y \\in \\mathcal{W}^N,
        ExactMatch(X, Y) = \\frac{1}{N} \\sum\\limits_{i=1}^N f_{Y_i}(X_i)

    with the indicator function :math:`f_{a}` defined as:

    .. math::
        \\forall a, x \\in \\mathcal{W},
        f_a(x) = \\left\\{
            \\begin{array}{ll}
                1 & \\mbox{if } x = a \\\\
                0 & \\mbox{otherwise.}
            \\end{array}
        \\right.

    where :math:`\\mathcal{W}` is the set of all possible character sequences,
    :math:`N` is a strictly positive integer.

    Example::
        >>> from doctr.utils import ExactMatch
        >>> metric = ExactMatch()
        >>> metric.update(['Hello', 'world'], ['hello', 'world'])
        >>> metric.summary()

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

    intersection = np.clip(right - left, 0, np.Inf) * np.clip(bot - top, 0, np.Inf)
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
    """Implements common confusion metrics and mean IoU for localization evaluation.

    The aggregated metrics are computed as follows:

    .. math::
        \\forall Y \\in \\mathcal{B}^N, \\forall X \\in \\mathcal{B}^M, \\\\
        Recall(X, Y) = \\frac{1}{N} \\sum\\limits_{i=1}^N g_{X}(Y_i) \\\\
        Precision(X, Y) = \\frac{1}{M} \\sum\\limits_{i=1}^N g_{X}(Y_i) \\\\
        meanIoU(X, Y) = \\frac{1}{M} \\sum\\limits_{i=1}^M \\max\\limits_{j \\in [1, N]}  IoU(X_i, Y_j)

    with the function :math:`IoU(x, y)` being the Intersection over Union between bounding boxes :math:`x` and
    :math:`y`, and the function :math:`g_{X}` defined as:

    .. math::
        \\forall y \\in \\mathcal{B},
        g_X(y) = \\left\\{
            \\begin{array}{ll}
                1 & \\mbox{if } y\\mbox{ has been assigned to any }(X_i)_i\\mbox{ with an }IoU \\geq 0.5 \\\\
                0 & \\mbox{otherwise.}
            \\end{array}
        \\right.

    where :math:`\\mathcal{B}` is the set of possible bounding boxes,
    :math:`N` (number of ground truths) and :math:`M` (number of predictions) are strictly positive integers.

    Example::
        >>> import numpy as np
        >>> from doctr.utils import LocalizationConfusion
        >>> metric = LocalizationConfusion(iou_thresh=0.5)
        >>> metric.update(np.asarray([[0, 0, 100, 100]]), np.asarray([[0, 0, 70, 70], [110, 95, 200, 150]]))
        >>> metric.summary()

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
    """Implements end-to-end OCR metric.

    The aggregated metrics are computed as follows:

    .. math::
        \\forall (B, L) \\in \\mathcal{B}^N \\times \\mathcal{L}^N,
        \\forall (\\hat{B}, \\hat{L}) \\in \\mathcal{B}^M \\times \\mathcal{L}^M, \\\\
        Recall(B, \\hat{B}, L, \\hat{L}) = \\frac{1}{N} \\sum\\limits_{i=1}^N h_{B,L}(\\hat{B}_i, \\hat{L}_i) \\\\
        Precision(B, \\hat{B}, L, \\hat{L}) = \\frac{1}{M} \\sum\\limits_{i=1}^N h_{B,L}(\\hat{B}_i, \\hat{L}_i) \\\\
        meanIoU(B, \\hat{B}) = \\frac{1}{M} \\sum\\limits_{i=1}^M \\max\\limits_{j \\in [1, N]}  IoU(\\hat{B}_i, B_j)

    with the function :math:`IoU(x, y)` being the Intersection over Union between bounding boxes :math:`x` and
    :math:`y`, and the function :math:`h_{B, L}` defined as:

    .. math::
        \\forall (b, l) \\in \\mathcal{B} \\times \\mathcal{L},
        h_{B,L}(b, l) = \\left\\{
            \\begin{array}{ll}
                1 & \\mbox{if } b\\mbox{ has been assigned to a given }B_j\\mbox{ with an } \\\\
                & IoU \\geq 0.5 \\mbox{ and that for this assignment, } l = L_j\\\\
                0 & \\mbox{otherwise.}
            \\end{array}
        \\right.

    where :math:`\\mathcal{B}` is the set of possible bounding boxes,
    :math:`\\mathcal{L}` is the set of possible character sequences,
    :math:`N` (number of ground truths) and :math:`M` (number of predictions) are strictly positive integers.

    Example::
        >>> import numpy as np
        >>> from doctr.utils import OCRMetric
        >>> metric = OCRMetric(iou_thresh=0.5)
        >>> metric.update(np.asarray([[0, 0, 100, 100]]), np.asarray([[0, 0, 70, 70], [110, 95, 200, 150]]),
        ['hello'], ['hello', 'world'])
        >>> metric.summary()

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
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        gt_labels: List[str],
        pred_labels: List[str],
    ) -> None:

        # Compute IoU
        iou_mat = box_iou(gt_boxes, pred_boxes)
        self.tot_iou += float(iou_mat.max(axis=1).sum())

        # Assign pairs
        gt_indices, preds_indices = assign_pairs(iou_mat, self.iou_thresh)

        # Compare sequences
        for gt_idx, pred_idx in zip(gt_indices, preds_indices):
            dist = levenshtein(gt_labels[gt_idx], pred_labels[pred_idx])
            self.tot_dist += dist
            if dist <= self.max_dist:
                self.num_reco_matches += 1

        # Update counts
        self.num_det_matches = len(gt_indices)
        self.num_gts += gt_boxes.shape[0]
        self.num_preds += pred_boxes.shape[0]

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
