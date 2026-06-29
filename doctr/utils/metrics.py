# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import numpy as np
from anyascii import anyascii
from scipy.optimize import linear_sum_assignment
from shapely import area, intersection, polygons

__all__ = [
    "TextMatch",
    "box_iou",
    "polygon_iou",
    "nms",
    "LocalizationConfusion",
    "OCRMetric",
    "DetectionMetric",
    "ObjectDetectionMetric",
]


def string_match(word1: str, word2: str) -> tuple[bool, bool, bool, bool]:
    """Performs string comparison with multiple levels of tolerance

    Args:
        word1: a string
        word2: another string

    Returns:
        a tuple with booleans specifying respectively whether the raw strings, their lower-case counterparts, their
            anyascii counterparts and their lower-case anyascii counterparts match
    """
    raw_match = word1 == word2
    caseless_match = word1.lower() == word2.lower()
    anyascii_match = anyascii(word1) == anyascii(word2)

    # Warning: the order is important here otherwise the pair ("EUR", "€") cannot be matched
    unicase_match = anyascii(word1).lower() == anyascii(word2).lower()

    return raw_match, caseless_match, anyascii_match, unicase_match


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
        gt: list[str],
        pred: list[str],
    ) -> None:
        """Update the state of the metric with new predictions

        Args:
            gt: list of groung-truth character sequences
            pred: list of predicted character sequences
        """
        if len(gt) != len(pred):
            raise AssertionError("prediction size does not match with ground-truth labels size")

        for gt_word, pred_word in zip(gt, pred):
            _raw, _caseless, _anyascii, _unicase = string_match(gt_word, pred_word)
            self.raw += int(_raw)
            self.caseless += int(_caseless)
            self.anyascii += int(_anyascii)
            self.unicase += int(_unicase)

        self.total += len(gt)

    def summary(self) -> dict[str, float]:
        """Computes the aggregated metrics

        Returns:
            a dictionary with the exact match score for the raw data, its lower-case counterpart, its anyascii
            counterpart and its lower-case anyascii counterpart
        """
        if self.total == 0:
            raise AssertionError("you need to update the metric before getting the summary")

        return dict(
            raw=self.raw / self.total,
            caseless=self.caseless / self.total,
            anyascii=self.anyascii / self.total,
            unicase=self.unicase / self.total,
        )

    def reset(self) -> None:
        self.raw = 0
        self.caseless = 0
        self.anyascii = 0
        self.unicase = 0
        self.total = 0


def box_iou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Computes the IoU between two sets of bounding boxes

    Args:
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)

    Returns:
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

        intersection = np.clip(right - left, 0, np.inf) * np.clip(bot - top, 0, np.inf)
        union = (r1 - l1) * (b1 - t1) + ((r2 - l2) * (b2 - t2)).T - intersection
        iou_mat = intersection / union

    return iou_mat


def polygon_iou(polys_1: np.ndarray, polys_2: np.ndarray) -> np.ndarray:
    """Computes the IoU between two sets of rotated bounding boxes

    Args:
        polys_1: rotated bounding boxes of shape (N, 4, 2)
        polys_2: rotated bounding boxes of shape (M, 4, 2)

    Returns:
        the IoU matrix of shape (N, M)
    """
    if polys_1.ndim != 3 or polys_2.ndim != 3:
        raise AssertionError("expects boxes to be in format (N, 4, 2)")

    n, m = polys_1.shape[0], polys_2.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    geoms_1 = polygons(polys_1)
    geoms_2 = polygons(polys_2)
    grid_1 = np.repeat(geoms_1, m)
    grid_2 = np.tile(geoms_2, n)

    # Compute intersections and areas
    intersections = area(intersection(grid_1, grid_2))
    areas_1 = area(grid_1)
    areas_2 = area(grid_2)

    # Compute IoU
    unions = areas_1 + areas_2 - intersections
    iou_flat = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    return iou_flat.reshape(n, m).astype(np.float32)


def nms(boxes: np.ndarray, thresh: float = 0.5) -> list[int]:
    """Perform non-max suppression, borrowed from <https://github.com/rbgirshick/fast-rcnn>`_.

    Args:
        boxes: np array of straight boxes: (*, 5), (xmin, ymin, xmax, ymax, score)
        thresh: iou threshold to perform box suppression.

    Returns:
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
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
        use_polygons: if set to True, predictions and targets will be expected to have rotated format
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_polygons: bool = False,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.use_polygons = use_polygons
        self.reset()

    def update(self, gts: np.ndarray, preds: np.ndarray) -> None:
        """Updates the metric

        Args:
            gts: a set of relative bounding boxes either of shape (N, 4) or (N, 5) if they are rotated ones
            preds: a set of relative bounding boxes either of shape (M, 4) or (M, 5) if they are rotated ones
        """
        if preds.shape[0] > 0:
            # Compute IoU
            if self.use_polygons:
                iou_mat = polygon_iou(gts, preds)
            else:
                iou_mat = box_iou(gts, preds)
            self.tot_iou += float(iou_mat.max(axis=0).sum())

            # Assign pairs
            gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
            self.matches += int((iou_mat[gt_indices, pred_indices] >= self.iou_thresh).sum())

        # Update counts
        self.num_gts += gts.shape[0]
        self.num_preds += preds.shape[0]

    def summary(self) -> tuple[float | None, float | None, float | None]:
        """Computes the aggregated metrics

        Returns:
            a tuple with the recall, precision and meanIoU scores
        """
        # Recall
        recall = self.matches / self.num_gts if self.num_gts > 0 else None

        # Precision
        precision = self.matches / self.num_preds if self.num_preds > 0 else None

        # mean IoU
        mean_iou = round(self.tot_iou / self.num_preds, 2) if self.num_preds > 0 else None

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
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
        use_polygons: if set to True, predictions and targets will be expected to have rotated format
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_polygons: bool = False,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.use_polygons = use_polygons
        self.reset()

    def update(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        gt_labels: list[str],
        pred_labels: list[str],
    ) -> None:
        """Updates the metric

        Args:
            gt_boxes: a set of relative bounding boxes either of shape (N, 4) or (N, 5) if they are rotated ones
            pred_boxes: a set of relative bounding boxes either of shape (M, 4) or (M, 5) if they are rotated ones
            gt_labels: a list of N string labels
            pred_labels: a list of M string labels
        """
        if gt_boxes.shape[0] != len(gt_labels) or pred_boxes.shape[0] != len(pred_labels):
            raise AssertionError(
                "there should be the same number of boxes and string both for the ground truth and the predictions"
            )

        # Compute IoU
        if pred_boxes.shape[0] > 0:
            if self.use_polygons:
                iou_mat = polygon_iou(gt_boxes, pred_boxes)
            else:
                iou_mat = box_iou(gt_boxes, pred_boxes)

            self.tot_iou += float(iou_mat.max(axis=0).sum())

            # Assign pairs
            gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
            is_kept = iou_mat[gt_indices, pred_indices] >= self.iou_thresh
            # String comparison
            for gt_idx, pred_idx in zip(gt_indices[is_kept], pred_indices[is_kept]):
                _raw, _caseless, _anyascii, _unicase = string_match(gt_labels[gt_idx], pred_labels[pred_idx])
                self.raw_matches += int(_raw)
                self.caseless_matches += int(_caseless)
                self.anyascii_matches += int(_anyascii)
                self.unicase_matches += int(_unicase)

        self.num_gts += gt_boxes.shape[0]
        self.num_preds += pred_boxes.shape[0]

    def summary(self) -> tuple[dict[str, float | None], dict[str, float | None], float | None]:
        """Computes the aggregated metrics

        Returns:
            a tuple with the recall & precision for each string comparison and the mean IoU
        """
        # Recall
        recall = dict(
            raw=self.raw_matches / self.num_gts if self.num_gts > 0 else None,
            caseless=self.caseless_matches / self.num_gts if self.num_gts > 0 else None,
            anyascii=self.anyascii_matches / self.num_gts if self.num_gts > 0 else None,
            unicase=self.unicase_matches / self.num_gts if self.num_gts > 0 else None,
        )

        # Precision
        precision = dict(
            raw=self.raw_matches / self.num_preds if self.num_preds > 0 else None,
            caseless=self.caseless_matches / self.num_preds if self.num_preds > 0 else None,
            anyascii=self.anyascii_matches / self.num_preds if self.num_preds > 0 else None,
            unicase=self.unicase_matches / self.num_preds if self.num_preds > 0 else None,
        )

        # mean IoU (overall detected boxes)
        mean_iou = round(self.tot_iou / self.num_preds, 2) if self.num_preds > 0 else None

        return recall, precision, mean_iou

    def reset(self) -> None:
        self.num_gts = 0
        self.num_preds = 0
        self.tot_iou = 0.0
        self.raw_matches = 0
        self.caseless_matches = 0
        self.anyascii_matches = 0
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
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
        use_polygons: if set to True, predictions and targets will be expected to have rotated format
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_polygons: bool = False,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.use_polygons = use_polygons
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
            gt_boxes: a set of relative bounding boxes either of shape (N, 4) or (N, 5) if they are rotated ones
            pred_boxes: a set of relative bounding boxes either of shape (M, 4) or (M, 5) if they are rotated ones
            gt_labels: an array of class indices of shape (N,)
            pred_labels: an array of class indices of shape (M,)
        """
        if gt_boxes.shape[0] != gt_labels.shape[0] or pred_boxes.shape[0] != pred_labels.shape[0]:
            raise AssertionError(
                "there should be the same number of boxes and string both for the ground truth and the predictions"
            )

        # Compute IoU
        if pred_boxes.shape[0] > 0:
            if self.use_polygons:
                iou_mat = polygon_iou(gt_boxes, pred_boxes)
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

    def summary(self) -> tuple[float | None, float | None, float | None]:
        """Computes the aggregated metrics

        Returns:
            a tuple with the recall & precision for each class prediction and the mean IoU
        """
        # Recall
        recall = self.num_matches / self.num_gts if self.num_gts > 0 else None

        # Precision
        precision = self.num_matches / self.num_preds if self.num_preds > 0 else None

        # mean IoU (overall detected boxes)
        mean_iou = round(self.tot_iou / self.num_preds, 2) if self.num_preds > 0 else None

        return recall, precision, mean_iou

    def reset(self) -> None:
        self.num_gts = 0
        self.num_preds = 0
        self.tot_iou = 0.0
        self.num_matches = 0


class ObjectDetectionMetric:
    r"""Implements a COCO-style object detection metric (mAP@[.5:.95]) inspired by the COCO evaluation protocol.
    The aggregated metrics are computed as follows:

    .. math::

        \forall (B, C) \in \mathcal{B}^N \times \mathcal{C}^N,
        \forall (\hat{B}, \hat{C}, S) \in \mathcal{B}^M \times \mathcal{C}^M \times \mathbb{R}^M, \\

        AP_t(C) =
        \frac{1}{101}
        \sum\limits_{r \in \{0, 0.01, \dots, 1.0\}}
        \max_{\tilde{r} \geq r} Precision_t(\tilde{r}, C) \\

        mAP@[.5:.95] =
        \frac{1}{|\mathcal{T}|}
        \sum\limits_{t \in \mathcal{T}}
        \frac{1}{|\mathcal{C}|}
        \sum\limits_{c \in \mathcal{C}} AP_t(c)

    where:
        - :math:`\mathcal{B}` is the set of possible bounding boxes,
        - :math:`\mathcal{C}` is the set of possible class indices,
        - :math:`S` are confidence scores associated to predictions,
        - :math:`\mathcal{T} = \{0.5, 0.55, \dots, 0.95\}` is the set of IoU thresholds,
        - :math:`AP_t(c)` is the Average Precision for class :math:`c`
        at IoU threshold :math:`t`.

    For a given class and IoU threshold, predictions from all images are
    aggregated and sorted globally by decreasing confidence score.

    Each prediction is greedily matched to the unmatched ground-truth box
    with the highest IoU, provided that:
        - the IoU is greater than or equal to the threshold,
        - the ground-truth box has not already been matched.

    True positives and false positives are accumulated to build a
    precision-recall curve.

    Average Precision is computed using the COCO 101-point interpolated
    precision-recall curve.

    >>> import numpy as np
    >>> from doctr.utils import ObjectDetectionMetric
    >>> metric = ObjectDetectionMetric()
    >>> metric.update(
    ...     np.asarray([[0, 0, 100, 100]]),
    ...     np.asarray([[0, 0, 80, 80], [120, 120, 200, 200]]),
    ...     np.asarray([0]),
    ...     np.asarray([0, 1]),
    ...     np.asarray([0.9, 0.3])
    ... )
    >>> metric.summary()

    Args:
        iou_thresholds: sequence of IoU thresholds used to compute the metric
            (defaults to np.arange(0.5, 1.0, 0.05))
        num_classes: total number of classes. If None, inferred from data
        use_polygons: if set to True, predictions and targets will be expected
            to have rotated format
    """

    def __init__(
        self,
        iou_thresholds: np.ndarray | None = None,
        num_classes: int | None = None,
        use_polygons: bool = False,
    ) -> None:
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else np.round(np.arange(0.5, 1.0, 0.05), 2)
        self.num_classes = num_classes
        self.use_polygons = use_polygons
        self.reset()

    def update(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
    ) -> None:
        if (
            gt_boxes.shape[0] != gt_labels.shape[0]
            or pred_boxes.shape[0] != pred_labels.shape[0]
            or pred_boxes.shape[0] != pred_scores.shape[0]
        ):
            raise AssertionError("Mismatch between boxes, labels, scores")

        self._gts.append({"boxes": gt_boxes, "labels": gt_labels})
        self._preds.append({"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores})

    def summary(self) -> dict[str, float | dict[float, float]]:
        """Computes the aggregated metrics

        Returns:
            a dictionary with the mAP@[.5:.95], AP@[.5], AP@[.75] and AP per IoU threshold
        """
        if len(self._gts) == 0:
            raise AssertionError("No samples added")

        classes = self._get_classes()
        ap_per_iou = {}

        for iou_thresh in self.iou_thresholds:
            class_aps = []
            for c in classes:
                ap = self._evaluate_class(c, iou_thresh)
                if ap is not None:
                    class_aps.append(ap)
            ap_per_iou[float(iou_thresh)] = float(np.mean(class_aps)) if class_aps else 0.0

        map_value = float(np.mean(list(ap_per_iou.values())))

        return {
            "mAP@[.5:.95]": map_value,
            "AP@[.5]": ap_per_iou.get(0.5, 0.0),
            "AP@[.75]": ap_per_iou.get(0.75, 0.0),
            "AP_per_IoU": ap_per_iou,
        }

    def _get_classes(self) -> np.ndarray:
        if self.num_classes is None:
            labels = []
            for g in self._gts:
                labels.extend(g["labels"].tolist())
            for p in self._preds:
                labels.extend(p["labels"].tolist())
            return np.unique(labels)
        return np.arange(self.num_classes)

    def _collect_gt_by_image(self, class_id) -> tuple[dict, int]:
        gt_by_image = {}
        total_gt = 0
        for img_idx, gt in enumerate(self._gts):
            mask = gt["labels"] == class_id
            gt_boxes = gt["boxes"][mask]
            gt_by_image[img_idx] = {
                "boxes": gt_boxes,
                "matched": np.zeros(len(gt_boxes), dtype=bool),
            }
            total_gt += len(gt_boxes)
        return gt_by_image, total_gt

    def _collect_detections(self, class_id) -> list[dict]:
        detections = []
        for img_idx, pred in enumerate(self._preds):
            mask = pred["labels"] == class_id
            pred_boxes = pred["boxes"][mask]
            pred_scores = pred["scores"][mask]
            for box, score in zip(pred_boxes, pred_scores):
                detections.append({
                    "image_id": img_idx,
                    "box": box,
                    "score": float(score),
                })
        return detections

    def _match_detections(self, detections, gt_by_image, iou_thresh) -> tuple[np.ndarray, np.ndarray]:
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        for det_idx, det in enumerate(detections):
            img_idx = det["image_id"]
            pred_box = det["box"]
            gt_data = gt_by_image[img_idx]
            gt_boxes = gt_data["boxes"]
            if len(gt_boxes) == 0:
                fp[det_idx] = 1
                continue
            if self.use_polygons:
                iou_mat = polygon_iou(gt_boxes, np.expand_dims(pred_box, axis=0))
            else:
                iou_mat = box_iou(gt_boxes, np.expand_dims(pred_box, axis=0))
            ious = iou_mat[:, 0]
            best_gt = np.argmax(ious)
            best_iou = ious[best_gt]
            if best_iou >= iou_thresh and not gt_data["matched"][best_gt]:
                tp[det_idx] = 1
                gt_data["matched"][best_gt] = True
            else:
                fp[det_idx] = 1
        return tp, fp

    def _evaluate_class(self, class_id, iou_thresh) -> float | None:
        gt_by_image, total_gt = self._collect_gt_by_image(class_id)
        if total_gt == 0:
            return None

        detections = self._collect_detections(class_id)
        if len(detections) == 0:
            return 0.0

        detections.sort(key=lambda x: -x["score"])
        tp, fp = self._match_detections(detections, gt_by_image, iou_thresh)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / total_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

        return self._compute_ap(recall, precision)

    def _compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """Computes the Average Precision using the 101-point interpolation method from COCO

        Args:
            recall: array of recall values
            precision: array of precision values

        Returns:
            the Average Precision score
        """
        # 101-point interpolation as per COCO
        precision = np.maximum.accumulate(precision[::-1])[::-1]

        recall_levels = np.linspace(0, 1, 101)
        precisions = np.zeros_like(recall_levels)

        for i, r in enumerate(recall_levels):
            p = precision[recall >= r]
            precisions[i] = np.max(p) if p.size > 0 else 0.0

        return float(np.mean(precisions))

    def reset(self):
        self._gts = []
        self._preds = []
