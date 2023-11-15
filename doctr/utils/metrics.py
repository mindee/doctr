# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from unidecode import unidecode

__all__ = [
    "TextMatch",
    "box_iou",
    "box_ioa",
    "mask_iou",
    "polygon_iou",
    "nms",
    "LocalizationConfusion",
    "OCRMetric",
    "DetectionMetric",
]


def string_match(word1: str, word2: str) -> Tuple[bool, bool, bool, bool]:
    """Performs string comparison with multiple levels of tolerance

    Args:
    ----
        word1: a string
        word2: another string

    Returns:
    -------
        a tuple with booleans specifying respectively whether the raw strings, their lower-case counterparts, their
            unidecode counterparts and their lower-case unidecode counterparts match
    """
    raw_match = word1 == word2
    caseless_match = word1.lower() == word2.lower()
    unidecode_match = unidecode(word1) == unidecode(word2)

    # Warning: the order is important here otherwise the pair ("EUR", "€") cannot be matched
    unicase_match = unidecode(word1).lower() == unidecode(word2).lower()

    return raw_match, caseless_match, unidecode_match, unicase_match


class TextMatch:
    r"""Implements text match metric (word-level accuracy) for recognition task.

    The raw aggregated metric is computed as follows:

    .. math::
        \forall X, Y \in \mathcal{W}^N,
        TextMatch(X, Y) = \frac{1}{N} \sum\limits_{i=1}^N f_{Y_i}(X_i)

    with the indicator function :math:`f_{a}` defined as:

    .. math::
        \forall a, x \in \mathcal{W},
        f_a(x) = \left\{
            \begin{array}{ll}
                1 & \mbox{if } x = a \\
                0 & \mbox{otherwise.}
            \end{array}
        \right.

    where :math:`\mathcal{W}` is the set of all possible character sequences,
    :math:`N` is a strictly positive integer.

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
        ----
            gt: list of groung-truth character sequences
            pred: list of predicted character sequences
        """
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

        Returns
        -------
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
    """Computes the IoU between two sets of bounding boxes

    Args:
    ----
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)

    Returns:
    -------
        the IoU matrix of shape (N, M)
    """
    iou_mat: np.ndarray = np.zeros((boxes_1.shape[0], boxes_2.shape[0]), dtype=np.float32)

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


def box_ioa(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Computes the IoA (intersection over area) between two sets of bounding boxes:
    ioa(i, j) = inter(i, j) / area(i)

    Args:
    ----
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)

    Returns:
    -------
        the IoA matrix of shape (N, M)
    """
    ioa_mat: np.ndarray = np.zeros((boxes_1.shape[0], boxes_2.shape[0]), dtype=np.float32)

    if boxes_1.shape[0] > 0 and boxes_2.shape[0] > 0:
        l1, t1, r1, b1 = np.split(boxes_1, 4, axis=1)
        l2, t2, r2, b2 = np.split(boxes_2, 4, axis=1)

        left = np.maximum(l1, l2.T)
        top = np.maximum(t1, t2.T)
        right = np.minimum(r1, r2.T)
        bot = np.minimum(b1, b2.T)

        intersection = np.clip(right - left, 0, np.Inf) * np.clip(bot - top, 0, np.Inf)
        area = (r1 - l1) * (b1 - t1)
        ioa_mat = intersection / area

    return ioa_mat


def mask_iou(masks_1: np.ndarray, masks_2: np.ndarray) -> np.ndarray:
    """Computes the IoU between two sets of boolean masks

    Args:
    ----
        masks_1: boolean masks of shape (N, H, W)
        masks_2: boolean masks of shape (M, H, W)

    Returns:
    -------
        the IoU matrix of shape (N, M)
    """
    if masks_1.shape[1:] != masks_2.shape[1:]:
        raise AssertionError("both boolean masks should have the same spatial shape")

    iou_mat: np.ndarray = np.zeros((masks_1.shape[0], masks_2.shape[0]), dtype=np.float32)

    if masks_1.shape[0] > 0 and masks_2.shape[0] > 0:
        axes = tuple(range(2, masks_1.ndim + 1))
        intersection = np.logical_and(masks_1[:, None, ...], masks_2[None, ...]).sum(axis=axes)
        union = np.logical_or(masks_1[:, None, ...], masks_2[None, ...]).sum(axis=axes)
        iou_mat = intersection / union

    return iou_mat


def polygon_iou(
    polys_1: np.ndarray, polys_2: np.ndarray, mask_shape: Tuple[int, int], use_broadcasting: bool = False
) -> np.ndarray:
    """Computes the IoU between two sets of rotated bounding boxes

    Args:
    ----
        polys_1: rotated bounding boxes of shape (N, 4, 2)
        polys_2: rotated bounding boxes of shape (M, 4, 2)
        mask_shape: spatial shape of the intermediate masks
        use_broadcasting: if set to True, leverage broadcasting speedup by consuming more memory

    Returns:
    -------
        the IoU matrix of shape (N, M)
    """
    if polys_1.ndim != 3 or polys_2.ndim != 3:
        raise AssertionError("expects boxes to be in format (N, 4, 2)")

    iou_mat: np.ndarray = np.zeros((polys_1.shape[0], polys_2.shape[0]), dtype=np.float32)

    if polys_1.shape[0] > 0 and polys_2.shape[0] > 0:
        if use_broadcasting:
            masks_1 = rbox_to_mask(polys_1, shape=mask_shape)
            masks_2 = rbox_to_mask(polys_2, shape=mask_shape)
            iou_mat = mask_iou(masks_1, masks_2)
        else:
            # Save memory by doing the computation for each pair
            for idx, b1 in enumerate(polys_1):
                m1 = _rbox_to_mask(b1, mask_shape)
                for _idx, b2 in enumerate(polys_2):
                    m2 = _rbox_to_mask(b2, mask_shape)
                    iou_mat[idx, _idx] = np.logical_and(m1, m2).sum() / np.logical_or(m1, m2).sum()

    return iou_mat


def _rbox_to_mask(box: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Converts a rotated bounding box to a boolean mask

    Args:
    ----
        box: rotated bounding box of shape (4, 2)
        shape: spatial shapes of the output masks

    Returns:
    -------
        the boolean mask of the specified shape
    """
    mask: np.ndarray = np.zeros(shape, dtype=np.uint8)
    # Get absolute coords
    if not np.issubdtype(box.dtype, np.integer):
        abs_box = box.copy()
        abs_box[:, 0] = abs_box[:, 0] * shape[1]
        abs_box[:, 1] = abs_box[:, 1] * shape[0]
        abs_box = abs_box.round().astype(int)
    else:
        abs_box = box
        abs_box[2:] = abs_box[2:] + 1
    cv2.fillPoly(mask, [abs_box - 1], 1)

    return mask.astype(bool)


def rbox_to_mask(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Converts rotated bounding boxes to boolean masks

    Args:
    ----
        boxes: rotated bounding boxes of shape (N, 4, 2)
        shape: spatial shapes of the output masks

    Returns:
    -------
        the boolean masks of shape (N, H, W)
    """
    masks: np.ndarray = np.zeros((boxes.shape[0], *shape), dtype=np.uint8)

    if boxes.shape[0] > 0:
        # Get absolute coordinates
        if not np.issubdtype(boxes.dtype, np.integer):
            abs_boxes = boxes.copy()
            abs_boxes[:, :, 0] = abs_boxes[:, :, 0] * shape[1]
            abs_boxes[:, :, 1] = abs_boxes[:, :, 1] * shape[0]
            abs_boxes = abs_boxes.round().astype(int)
        else:
            abs_boxes = boxes
            abs_boxes[:, 2:] = abs_boxes[:, 2:] + 1

        # TODO: optimize slicing to improve vectorization
        for idx, _box in enumerate(abs_boxes):
            cv2.fillPoly(masks[idx], [_box - 1], 1)
    return masks.astype(bool)


def nms(boxes: np.ndarray, thresh: float = 0.5) -> List[int]:
    """Perform non-max suppression, borrowed from <https://github.com/rbgirshick/fast-rcnn>`_.

    Args:
    ----
        boxes: np array of straight boxes: (*, 5), (xmin, ymin, xmax, ymax, score)
        thresh: iou threshold to perform box suppression.

    Returns:
    -------
        A list of box indexes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class LocalizationConfusion:
    r"""Implements common confusion metrics and mean IoU for localization evaluation.

    The aggregated metrics are computed as follows:

    .. math::
        \forall Y \in \mathcal{B}^N, \forall X \in \mathcal{B}^M, \\
        Recall(X, Y) = \frac{1}{N} \sum\limits_{i=1}^N g_{X}(Y_i) \\
        Precision(X, Y) = \frac{1}{M} \sum\limits_{i=1}^M g_{X}(Y_i) \\
        meanIoU(X, Y) = \frac{1}{M} \sum\limits_{i=1}^M \max\limits_{j \in [1, N]}  IoU(X_i, Y_j)

    with the function :math:`IoU(x, y)` being the Intersection over Union between bounding boxes :math:`x` and
    :math:`y`, and the function :math:`g_{X}` defined as:

    .. math::
        \forall y \in \mathcal{B},
        g_X(y) = \left\{
            \begin{array}{ll}
                1 & \mbox{if } y\mbox{ has been assigned to any }(X_i)_i\mbox{ with an }IoU \geq 0.5 \\
                0 & \mbox{otherwise.}
            \end{array}
        \right.

    where :math:`\mathcal{B}` is the set of possible bounding boxes,
    :math:`N` (number of ground truths) and :math:`M` (number of predictions) are strictly positive integers.

    >>> import numpy as np
    >>> from doctr.utils import LocalizationConfusion
    >>> metric = LocalizationConfusion(iou_thresh=0.5)
    >>> metric.update(np.asarray([[0, 0, 100, 100]]), np.asarray([[0, 0, 70, 70], [110, 95, 200, 150]]))
    >>> metric.summary()

    Args:
    ----
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
        use_polygons: if set to True, predictions and targets will be expected to have rotated format
        mask_shape: if use_polygons is True, describes the spatial shape of the image used
        use_broadcasting: if use_polygons is True, use broadcasting for IoU computation by consuming more memory
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_polygons: bool = False,
        mask_shape: Tuple[int, int] = (1024, 1024),
        use_broadcasting: bool = True,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.use_polygons = use_polygons
        self.mask_shape = mask_shape
        self.use_broadcasting = use_broadcasting
        self.reset()

    def update(self, gts: np.ndarray, preds: np.ndarray) -> None:
        """Updates the metric

        Args:
        ----
            gts: a set of relative bounding boxes either of shape (N, 4) or (N, 5) if they are rotated ones
            preds: a set of relative bounding boxes either of shape (M, 4) or (M, 5) if they are rotated ones
        """
        if preds.shape[0] > 0:
            # Compute IoU
            if self.use_polygons:
                iou_mat = polygon_iou(gts, preds, self.mask_shape, self.use_broadcasting)
            else:
                iou_mat = box_iou(gts, preds)
            self.tot_iou += float(iou_mat.max(axis=0).sum())

            # Assign pairs
            gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
            self.matches += int((iou_mat[gt_indices, pred_indices] >= self.iou_thresh).sum())

        # Update counts
        self.num_gts += gts.shape[0]
        self.num_preds += preds.shape[0]

    def summary(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Computes the aggregated metrics

        Returns
        -------
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
        self.tot_iou = 0.0


class OCRMetric:
    r"""Implements an end-to-end OCR metric.

    The aggregated metrics are computed as follows:

    .. math::
        \forall (B, L) \in \mathcal{B}^N \times \mathcal{L}^N,
        \forall (\hat{B}, \hat{L}) \in \mathcal{B}^M \times \mathcal{L}^M, \\
        Recall(B, \hat{B}, L, \hat{L}) = \frac{1}{N} \sum\limits_{i=1}^N h_{B,L}(\hat{B}_i, \hat{L}_i) \\
        Precision(B, \hat{B}, L, \hat{L}) = \frac{1}{M} \sum\limits_{i=1}^M h_{B,L}(\hat{B}_i, \hat{L}_i) \\
        meanIoU(B, \hat{B}) = \frac{1}{M} \sum\limits_{i=1}^M \max\limits_{j \in [1, N]}  IoU(\hat{B}_i, B_j)

    with the function :math:`IoU(x, y)` being the Intersection over Union between bounding boxes :math:`x` and
    :math:`y`, and the function :math:`h_{B, L}` defined as:

    .. math::
        \forall (b, l) \in \mathcal{B} \times \mathcal{L},
        h_{B,L}(b, l) = \left\{
            \begin{array}{ll}
                1 & \mbox{if } b\mbox{ has been assigned to a given }B_j\mbox{ with an } \\
                & IoU \geq 0.5 \mbox{ and that for this assignment, } l = L_j\\
                0 & \mbox{otherwise.}
            \end{array}
        \right.

    where :math:`\mathcal{B}` is the set of possible bounding boxes,
    :math:`\mathcal{L}` is the set of possible character sequences,
    :math:`N` (number of ground truths) and :math:`M` (number of predictions) are strictly positive integers.

    >>> import numpy as np
    >>> from doctr.utils import OCRMetric
    >>> metric = OCRMetric(iou_thresh=0.5)
    >>> metric.update(np.asarray([[0, 0, 100, 100]]), np.asarray([[0, 0, 70, 70], [110, 95, 200, 150]]),
    >>>               ['hello'], ['hello', 'world'])
    >>> metric.summary()

    Args:
    ----
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
        use_polygons: if set to True, predictions and targets will be expected to have rotated format
        mask_shape: if use_polygons is True, describes the spatial shape of the image used
        use_broadcasting: if use_polygons is True, use broadcasting for IoU computation by consuming more memory
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_polygons: bool = False,
        mask_shape: Tuple[int, int] = (1024, 1024),
        use_broadcasting: bool = True,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.use_polygons = use_polygons
        self.mask_shape = mask_shape
        self.use_broadcasting = use_broadcasting
        self.reset()

    def update(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        gt_labels: List[str],
        pred_labels: List[str],
    ) -> None:
        """Updates the metric

        Args:
        ----
            gt_boxes: a set of relative bounding boxes either of shape (N, 4) or (N, 5) if they are rotated ones
            pred_boxes: a set of relative bounding boxes either of shape (M, 4) or (M, 5) if they are rotated ones
            gt_labels: a list of N string labels
            pred_labels: a list of M string labels
        """
        if gt_boxes.shape[0] != len(gt_labels) or pred_boxes.shape[0] != len(pred_labels):
            raise AssertionError(
                "there should be the same number of boxes and string both for the ground truth " "and the predictions"
            )

        # Compute IoU
        if pred_boxes.shape[0] > 0:
            if self.use_polygons:
                iou_mat = polygon_iou(gt_boxes, pred_boxes, self.mask_shape, self.use_broadcasting)
            else:
                iou_mat = box_iou(gt_boxes, pred_boxes)

            self.tot_iou += float(iou_mat.max(axis=0).sum())

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

    def summary(self) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], Optional[float]]:
        """Computes the aggregated metrics

        Returns
        -------
            a tuple with the recall & precision for each string comparison and the mean IoU
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
        self.tot_iou = 0.0
        self.raw_matches = 0
        self.caseless_matches = 0
        self.unidecode_matches = 0
        self.unicase_matches = 0


class DetectionMetric:
    r"""Implements an object detection metric.

    The aggregated metrics are computed as follows:

    .. math::
        \forall (B, C) \in \mathcal{B}^N \times \mathcal{C}^N,
        \forall (\hat{B}, \hat{C}) \in \mathcal{B}^M \times \mathcal{C}^M, \\
        Recall(B, \hat{B}, C, \hat{C}) = \frac{1}{N} \sum\limits_{i=1}^N h_{B,C}(\hat{B}_i, \hat{C}_i) \\
        Precision(B, \hat{B}, C, \hat{C}) = \frac{1}{M} \sum\limits_{i=1}^M h_{B,C}(\hat{B}_i, \hat{C}_i) \\
        meanIoU(B, \hat{B}) = \frac{1}{M} \sum\limits_{i=1}^M \max\limits_{j \in [1, N]}  IoU(\hat{B}_i, B_j)

    with the function :math:`IoU(x, y)` being the Intersection over Union between bounding boxes :math:`x` and
    :math:`y`, and the function :math:`h_{B, C}` defined as:

    .. math::
        \forall (b, c) \in \mathcal{B} \times \mathcal{C},
        h_{B,C}(b, c) = \left\{
            \begin{array}{ll}
                1 & \mbox{if } b\mbox{ has been assigned to a given }B_j\mbox{ with an } \\
                & IoU \geq 0.5 \mbox{ and that for this assignment, } c = C_j\\
                0 & \mbox{otherwise.}
            \end{array}
        \right.

    where :math:`\mathcal{B}` is the set of possible bounding boxes,
    :math:`\mathcal{C}` is the set of possible class indices,
    :math:`N` (number of ground truths) and :math:`M` (number of predictions) are strictly positive integers.

    >>> import numpy as np
    >>> from doctr.utils import DetectionMetric
    >>> metric = DetectionMetric(iou_thresh=0.5)
    >>> metric.update(np.asarray([[0, 0, 100, 100]]), np.asarray([[0, 0, 70, 70], [110, 95, 200, 150]]),
    >>>               np.zeros(1, dtype=np.int64), np.array([0, 1], dtype=np.int64))
    >>> metric.summary()

    Args:
    ----
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
        use_polygons: if set to True, predictions and targets will be expected to have rotated format
        mask_shape: if use_polygons is True, describes the spatial shape of the image used
        use_broadcasting: if use_polygons is True, use broadcasting for IoU computation by consuming more memory
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_polygons: bool = False,
        mask_shape: Tuple[int, int] = (1024, 1024),
        use_broadcasting: bool = True,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.use_polygons = use_polygons
        self.mask_shape = mask_shape
        self.use_broadcasting = use_broadcasting
        self.reset()

    def update(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> None:
        """Updates the metric

        Args:
        ----
            gt_boxes: a set of relative bounding boxes either of shape (N, 4) or (N, 5) if they are rotated ones
            pred_boxes: a set of relative bounding boxes either of shape (M, 4) or (M, 5) if they are rotated ones
            gt_labels: an array of class indices of shape (N,)
            pred_labels: an array of class indices of shape (M,)
        """
        if gt_boxes.shape[0] != gt_labels.shape[0] or pred_boxes.shape[0] != pred_labels.shape[0]:
            raise AssertionError(
                "there should be the same number of boxes and string both for the ground truth " "and the predictions"
            )

        # Compute IoU
        if pred_boxes.shape[0] > 0:
            if self.use_polygons:
                iou_mat = polygon_iou(gt_boxes, pred_boxes, self.mask_shape, self.use_broadcasting)
            else:
                iou_mat = box_iou(gt_boxes, pred_boxes)

            self.tot_iou += float(iou_mat.max(axis=0).sum())

            # Assign pairs
            gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
            is_kept = iou_mat[gt_indices, pred_indices] >= self.iou_thresh
            # Category comparison
            self.num_matches += int((gt_labels[gt_indices[is_kept]] == pred_labels[pred_indices[is_kept]]).sum())

        self.num_gts += gt_boxes.shape[0]
        self.num_preds += pred_boxes.shape[0]

    def summary(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Computes the aggregated metrics

        Returns
        -------
            a tuple with the recall & precision for each class prediction and the mean IoU
        """
        # Recall
        recall = self.num_matches / self.num_gts if self.num_gts > 0 else None

        # Precision
        precision = self.num_matches / self.num_preds if self.num_preds > 0 else None

        # mean IoU (overall detected boxes)
        mean_iou = self.tot_iou / self.num_preds if self.num_preds > 0 else None

        return recall, precision, mean_iou

    def reset(self) -> None:
        self.num_gts = 0
        self.num_preds = 0
        self.tot_iou = 0.0
        self.num_matches = 0
