# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
from copy import deepcopy
from typing import List, Any, Optional, Dict, Tuple

from doctr.utils.repr import NestedObject
from .._utils import rotate_page, get_bitmap_angle
from ...utils.metrics import box_iou
from .. import PreProcessor


__all__ = ['DetectionModel', 'DetectionPostProcessor', 'DetectionPredictor']


class DetectionModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg


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

    @staticmethod
    def nms(boxes: np.ndarray, thresh: float = .5) -> List[int]:
        """Perform non-max suppression, borrowed from <https://github.com/rbgirshick/fast-rcnn>`_.

        Args:
            boxes: np array of straight boxes: (*, 5)
            thresh: iou threshold to perform box suppression.

        Returns:
            A list of box indexes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def filter_boxes(
        self,
        boxes: np.ndarray,
        nms_thresh: float = .8,
        merging_thresh: float = .2,
    ) -> np.ndarray:
        """Filter an array of boxes: boxes with an iou > merging threshold will be merged, and
        nms is performed with a nms threshold

        Args:
            boxes: array of boxes, shape (N, 5) or (N, 6)
            nms_thresh: iou threshold to perform nms
            merging_thresh: threshold used to merge boxes

        Returns:
            A np array of filtered boxes, containing at most N boxes.
        """
        # Array of rotated boxes: not supported for now
        # TODO: support for rotated boxes (both NMS & merging algorithm)
        if boxes.shape[1] == 6:
            return boxes

        if not np.any(boxes):
            # If array of boxes is empty, return it
            return boxes

        # First perform NMS on array of boxes
        keep = self.nms(boxes, thresh=nms_thresh)

        # Update box list
        box_list = [list(box) for i, box in enumerate(list(boxes)) if i in keep]

        # Compute iou beween boxes
        boxes = np.asarray(box_list)
        iou_mat = box_iou(boxes[:, :4], boxes[:, :4])
        tri_mat = np.tril(iou_mat, -1)

        def enclosing(box_a: List[float], box_b: List[float]) -> List[float]:
            """Compute enclosing bbox from 2 list of 5 floats [xmin, ymin, xmax, ymax, score].
            Return a list of [xmin, ymin, xmax, ymax, score].
            """
            x1, y1, x2, y2 = box_a[:4]
            x3, y3, x4, y4 = box_b[:4]
            score = (box_a[4] + box_b[4]) / 2
            enclosing = [min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4), score]
            return enclosing

        # Find box indexes with a iou > merging_thresh, and merge them
        to_merge = np.argwhere(tri_mat > merging_thresh)
        merged = {k: v for k, v in enumerate(box_list)}
        for [i, j] in list(to_merge):
            # Resolve enclosing box, update dictionnary
            enclosing_box = enclosing(merged[i], merged[j])
            merged[i] = enclosing_box
            merged[j] = enclosing_box

        # Update box list: remove duplicates from merged dictionnary
        box_list = []
        for box in merged.values():
            if box not in box_list:
                box_list.append(box)

        return np.asarray(box_list)

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

        bitmap = (proba_map > self.bin_thresh).astype(np.float32)

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
            bitmap_, p_ = rotate_page(bitmap_, -angle), rotate_page(p_, -angle)
            boxes = self.bitmap_to_boxes(pred=p_, bitmap=bitmap_)
            # Filter boxes
            boxes = self.filter_boxes(boxes)
            boxes_batch.append(boxes)

        return boxes_batch, angles_batch


class DetectionPredictor(NestedObject):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    _children_names: List[str] = ['pre_processor', 'model']

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: DetectionModel,
    ) -> None:

        self.pre_processor = pre_processor
        self.model = model

    def __call__(
        self,
        pages: List[np.ndarray],
        **kwargs: Any,
    ) -> List[np.ndarray]:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        predicted_batches = [
            self.model(batch, return_boxes=True, **kwargs)['preds']  # type:ignore[operator]
            for batch in processed_batches
        ]
        return [pred for batch in predicted_batches for pred in zip(*batch)]
