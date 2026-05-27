# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

import cv2
import numpy as np

from doctr.models.core import BaseModel

__all__ = ["_LWDETR", "LWDETRPostProcessor"]


class LWDETRPostProcessor:
    """Implements a post processor for LW-DETR model

    Args:
        num_classes: number of classes
        score_thresh: confidence threshold for filtering predictions
        iou_thresh: IoU threshold for NMS
        topk: number of top predictions to keep before NMS
        assume_straight_pages: whether the pages are assumed to be straight (i.e., no rotation)
    """

    def __init__(
        self,
        num_classes: int,
        score_thresh: float = 0.3,
        iou_thresh: float = 0.5,
        topk: int = 300,
        assume_straight_pages: bool = True,
    ):
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.topk = topk
        self.assume_straight_pages = assume_straight_pages

    def _decode_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """
        Decode cxcywh -> polygons (axis-aligned rectangles)
        """
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        polys = []

        for i in range(len(boxes)):
            x1 = cx[i] - w[i] / 2
            y1 = cy[i] - h[i] / 2
            x2 = cx[i] + w[i] / 2
            y2 = cy[i] + h[i] / 2

            poly = np.array(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                dtype=np.float32,
            )

            polys.append(poly)

        return np.asarray(polys, dtype=np.float32)

    def _iou(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """Compute the IoU between two polygons

        Args:
            poly1: first polygon (4, 2)
            poly2: second polygon (4, 2)

        Returns:
            IoU between the two polygons
        """
        inter = cv2.intersectConvexConvex(
            poly1.astype(np.float32),
            poly2.astype(np.float32),
        )[0]

        if inter <= 0:
            return 0.0

        area1 = cv2.contourArea(poly1)
        area2 = cv2.contourArea(poly2)

        return inter / (area1 + area2 - inter + 1e-6)

    def _nms(self, polys: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> list[int]:
        """
        Class-wise greedy NMS for rotated polygons.

        Args:
            polys: (N, 4, 2)
            scores: (N,)
            labels: (N,)

        Returns:
            indices kept after NMS (global indices)
        """
        if len(polys) == 0:
            return []

        keep: list[int] = []

        # Process each class independently
        for cls in np.unique(labels):
            cls_idxs = np.where(labels == cls)[0]
            if len(cls_idxs) == 0:
                continue

            cls_scores = scores[cls_idxs]
            cls_polys = polys[cls_idxs]

            # sort by confidence
            order = np.argsort(cls_scores)[::-1]
            cls_idxs = cls_idxs[order]
            cls_polys = cls_polys[order]
            cls_scores = cls_scores[order]

            suppressed = np.zeros(len(cls_idxs), dtype=bool)

            for i in range(len(cls_idxs)):
                if suppressed[i]:
                    continue

                keep.append(cls_idxs[i])

                # compare current box with the rest
                for j in range(i + 1, len(cls_idxs)):
                    if suppressed[j]:
                        continue

                    iou = self._iou(cls_polys[i], cls_polys[j])
                    if iou >= self.iou_thresh:
                        suppressed[j] = True
        return keep

    def __call__(self, logits: np.ndarray, boxes: np.ndarray):

        logits = np.asarray(logits)
        boxes = np.asarray(boxes)

        results = []

        for b in range(boxes.shape[0]):
            exp = np.exp(logits[b] - logits[b].max(axis=-1, keepdims=True))
            prob = exp / exp.sum(axis=-1, keepdims=True)

            prob_fg = prob[:, :-1]
            scores = prob_fg.max(axis=-1)
            labels = prob_fg.argmax(axis=-1)

            if self.topk is not None and len(scores) > self.topk:
                idxs = np.argsort(scores)[::-1][: self.topk]
            else:
                idxs = np.arange(len(scores))

            scores_b = scores[idxs]
            labels_b = labels[idxs]
            bboxes = boxes[b][idxs]

            mask = scores_b > self.score_thresh

            bboxes = bboxes[mask]
            scores_b = scores_b[mask]
            labels_b = labels_b[mask]

            polys = self._decode_boxes(bboxes) if len(bboxes) > 0 else np.zeros((0, 4, 2), dtype=np.float32)

            keep = self._nms(polys, scores_b, labels_b) if len(polys) > 0 else []

            final_boxes = []
            final_labels = []
            final_scores = []

            for idx in keep:
                poly = polys[idx]

                if self.assume_straight_pages:
                    # 👉 COCO-style axis aligned box from polygon
                    xmin = float(np.min(poly[:, 0]))
                    xmax = float(np.max(poly[:, 0]))
                    ymin = float(np.min(poly[:, 1]))
                    ymax = float(np.max(poly[:, 1]))

                    final_boxes.append([xmin, ymin, xmax, ymax])
                else:
                    final_boxes.append(poly.reshape(-1).tolist())

                final_labels.append(int(labels_b[idx]))
                final_scores.append(float(scores_b[idx]))

            final_boxes_arr = np.asarray(final_boxes, dtype=np.float32)

            results.append((
                final_labels,
                final_boxes_arr,  # <- NOW ALWAYS CLEAN FORMAT
                final_scores,
            ))

        return results


class _LWDETR(BaseModel):
    """LW-DETR as described in `"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
    <https://arxiv.org/pdf/2406.03459v1>`_.
    """

    def build_target(
        self,
        target: list[dict[str, np.ndarray]],
        class_names: list[str],
    ) -> list[dict[str, Any]]:
        """
        Build targets in COCO format: [xmin, ymin, w, h]
        """
        targets = []
        class_to_id = {name: i for i, name in enumerate(class_names)}

        def to_quad(box: np.ndarray):
            box = np.asarray(box, dtype=np.float32)
            if box.shape == (4,):
                x1, y1, x2, y2 = box
                return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            if box.shape == (8,):
                return box.reshape(4, 2)
            if box.shape == (4, 2):
                return box.astype(np.float32)
            raise ValueError(f"Unsupported box shape: {box.shape}")

        def quad_to_coco(poly: np.ndarray) -> np.ndarray:
            xmin = float(np.min(poly[:, 0]))
            xmax = float(np.max(poly[:, 0]))
            ymin = float(np.min(poly[:, 1]))
            ymax = float(np.max(poly[:, 1]))

            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w / 2.0
            cy = ymin + h / 2.0

            return np.array([cx, cy, w, h], dtype=np.float32)

        for sample in target:
            boxes_all = []
            labels_all = []

            for class_name, boxes in sample.items():
                if class_name not in class_to_id:
                    raise ValueError(f"Unknown class name: {class_name}")

                cls_id = class_to_id[class_name]
                boxes = np.asarray(boxes)

                if boxes.ndim == 1:
                    boxes = boxes[None, :]

                # sanity check normalized coords
                flat = boxes.ravel()
                coord_vals = flat[flat > 0]
                if len(coord_vals) > 0 and coord_vals.max() > 1.5:
                    raise ValueError("build_target expects normalized [0,1] coordinates.")

                for box in boxes:
                    poly = to_quad(box)
                    coco_box = quad_to_coco(poly)

                    if coco_box[2] <= 1e-5 or coco_box[3] <= 1e-5:
                        continue

                    boxes_all.append(coco_box)
                    labels_all.append(cls_id)

            if len(boxes_all) == 0:
                boxes_all = np.zeros((0, 4), dtype=np.float32)
                labels_all = np.zeros((0,), dtype=np.int64)

            targets.append({
                "boxes": np.asarray(boxes_all, dtype=np.float32),  # (N, 4)
                "class_labels": np.asarray(labels_all, dtype=np.int64),
            })

        return targets
