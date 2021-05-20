# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from typing import List, Tuple, Dict
from unidecode import unidecode
from scipy.optimize import linear_sum_assignment

__all__ = ['TextMatch', 'box_iou', 'LocalizationConfusion', 'OCRMetric']


def string_match(word1: str, word2: str) -> Tuple[bool, bool, bool, bool]:
    """Perform string comparison with multiple levels of tolerance

    Args:
        word1: a string
        word2: another string

    Returns:
        a tuple with booleans specifying respectively whether the raw strings, their lower-case counterparts, their
            unidecode counterparts and their lower-case unidecode counterparts match
    """
    raw_match = (word1 == word2)
    caseless_match = (word1.lower() == word2.lower())
    unidecode_match = (unidecode(word1) == unidecode(word2))

    # Warning: the order is important here otherwise the pair ("EUR", "€") cannot be matched
    unicase_match = (unidecode(word1).lower() == unidecode(word2).lower())

    return raw_match, caseless_match, unidecode_match, unicase_match


class TextMatch:
    """Implements text match metric (word-level accuracy) for recognition task.

    The raw aggregated metric is computed as follows:

    .. math::
        \\forall X, Y \\in \\mathcal{W}^N,
        TextMatch(X, Y) = \\frac{1}{N} \\sum\\limits_{i=1}^N f_{Y_i}(X_i)

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
        >>> from doctr.utils import TextMatch
        >>> metric = TextMatch()
        >>> metric.update(['Hello', 'world'], ['hello', 'world'])
        >>> metric.summary()
    """

    def __init__(self) -> None:
        self.reset()

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

        for gt_word, pred_word in zip(gt, pred):
            _raw, _caseless, _unidecode, _unicase = string_match(gt_word, pred_word)
            self.raw += int(_raw)
            self.caseless += int(_caseless)
            self.unidecode += int(_unidecode)
            self.unicase += int(_unicase)

        self.total += len(gt)

    def summary(self) -> Dict[str, float]:
        """Computes the aggregated metrics

        Returns:
            a dictionary with the exact match score for the raw data, its lower-case counterpart, its unidecode
            counterpart and its lower-case unidecode counterpart
        """
        if self.total == 0:
            raise AssertionError("you need to update the metric before getting the summary")

        return dict(
            raw=self.raw / self.total,
            caseless=self.caseless / self.total,
            unidecode=self.unidecode / self.total,
            unicase=self.unicase / self.total,
        )

    def reset(self) -> None:
        self.raw = 0
        self.caseless = 0
        self.unidecode = 0
        self.unicase = 0
        self.total = 0


def box_iou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Compute the IoU between two sets of bounding boxes

    Args:
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)

    Returns:
        the IoU matrix of shape (N, M)
    """

    iou_mat = np.zeros((boxes_1.shape[0], boxes_2.shape[0]), dtype=np.float32)

    if boxes_1.shape[0] > 0 and boxes_2.shape[0] > 0:
        l1, t1, r1, b1 = np.split(boxes_1, 4, axis=1)
        l2, t2, r2, b2 = np.split(boxes_2, 4, axis=1)

        left = np.maximum(l1, l2.T)
        top = np.maximum(t1, t2.T)
        right = np.minimum(r1, r2.T)
        bot = np.minimum(b1, b2.T)

        intersection = np.clip(right - left, 0, np.Inf) * np.clip(bot - top, 0, np.Inf)
        union = (r1 - l1) * (b1 - t1) + ((r2 - l2) * (b2 - t2)).T - intersection
        iou_mat = intersection / union

    return iou_mat


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
        self.reset()

    def update(self, gts: np.ndarray, preds: np.ndarray) -> None:

        if preds.shape[0] > 0:
            # Compute IoU
            iou_mat = box_iou(gts, preds)
            self.tot_iou += float(iou_mat.max(axis=1).sum())

            # Assign pairs
            gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
            self.matches += int((iou_mat[gt_indices, pred_indices] >= self.iou_thresh).sum())

        # Update counts
        self.num_gts += gts.shape[0]
        self.num_preds += preds.shape[0]

    def summary(self) -> Tuple[float, float, float]:
        """Computes the aggregated metrics

        Returns:
            a tuple with the recall, precision and meanIoU scores
        """

        # Recall
        recall = self.matches / self.num_gts if self.num_gts > 0 else None

        # Precision
        precision = self.matches / self.num_preds if self.num_preds > 0 else None

        # mean IoU
        mean_iou = self.tot_iou / self.num_preds if self.num_preds > 0 else None

        return recall, precision, mean_iou

    def reset(self) -> None:
        self.num_gts = 0
        self.num_preds = 0
        self.matches = 0
        self.tot_iou = 0.


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
    """

    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = iou_thresh
        self.reset()

    def update(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        gt_labels: List[str],
        pred_labels: List[str],
    ) -> None:

        if gt_boxes.shape[0] != len(gt_labels) or pred_boxes.shape[0] != len(pred_labels):
            raise AssertionError("there should be the same number of boxes and string both for the ground truth "
                                 "and the predictions")

        # Compute IoU
        if pred_boxes.shape[0] > 0:
            iou_mat = box_iou(gt_boxes, pred_boxes)
            self.tot_iou += float(iou_mat.max(axis=1).sum())

            # Assign pairs
            gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
            is_kept = iou_mat[gt_indices, pred_indices] >= self.iou_thresh
            # String comparison
            for gt_idx, pred_idx in zip(gt_indices[is_kept], pred_indices[is_kept]):
                _raw, _caseless, _unidecode, _unicase = string_match(gt_labels[gt_idx], pred_labels[pred_idx])
                self.raw_matches += int(_raw)
                self.caseless_matches += int(_caseless)
                self.unidecode_matches += int(_unidecode)
                self.unicase_matches += int(_unicase)

        self.num_gts += gt_boxes.shape[0]
        self.num_preds += pred_boxes.shape[0]

    def summary(self) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """Computes the aggregated metrics

        Returns:
            a tuple with the recall & precision for each string comparison flexibility and the mean IoU
        """

        # Recall
        recall = dict(
            raw=self.raw_matches / self.num_gts if self.num_gts > 0 else None,
            caseless=self.caseless_matches / self.num_gts if self.num_gts > 0 else None,
            unidecode=self.unidecode_matches / self.num_gts if self.num_gts > 0 else None,
            unicase=self.unicase_matches / self.num_gts if self.num_gts > 0 else None,
        )

        # Precision
        precision = dict(
            raw=self.raw_matches / self.num_preds if self.num_preds > 0 else None,
            caseless=self.caseless_matches / self.num_preds if self.num_preds > 0 else None,
            unidecode=self.unidecode_matches / self.num_preds if self.num_preds > 0 else None,
            unicase=self.unicase_matches / self.num_preds if self.num_preds > 0 else None,
        )

        # mean IoU (overall detected boxes)
        mean_iou = self.tot_iou / self.num_preds if self.num_preds > 0 else None

        return recall, precision, mean_iou

    def reset(self) -> None:
        self.num_gts = 0
        self.num_preds = 0
        self.tot_iou = 0.
        self.raw_matches = 0
        self.caseless_matches = 0
        self.unidecode_matches = 0
        self.unicase_matches = 0
