# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

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

    def _decode_boxes(self, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode the predicted boxes from OBB format to polygon format

        Args:
            boxes: array of predicted boxes in OBB format (N, 6) (cx, cy, w, h, sin(theta), cos(theta))

        Returns:
            tuple of (polys, angles) where polys is an array of decoded polygons (N, 4, 2)
                and angles is an array of angles in radians (N,)
        """
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        sin, cos = boxes[:, 4], boxes[:, 5]

        angles = np.arctan2(sin, cos)

        polys = []
        for i in range(len(boxes)):
            rect = ((float(cx[i]), float(cy[i])), (float(w[i]), float(h[i])), float(np.degrees(angles[i])))

            poly = cv2.boxPoints(rect)
            polys.append(poly)

        return np.asarray(polys, dtype=np.float32), angles

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

    def _nms(self, polys: np.ndarray, scores: np.ndarray) -> list[int]:
        """Perform NMS on the predicted polygons

        Args:
            polys: array of predicted polygons (N, 4, 2)
            scores: array of predicted scores (N,)

        Returns:
            list of indices of the polygons to keep after NMS
        """
        idxs = np.argsort(scores)[::-1]
        keep = []

        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)

            if idxs.size == 1:
                break

            rest = idxs[1:]

            ious = np.array([self._iou(polys[i], polys[j]) for j in rest])

            idxs = rest[ious < self.iou_thresh]

        return keep

    def __call__(self, logits: np.ndarray, boxes: np.ndarray) -> list[tuple[list[int], np.ndarray, list[float]]]:

        logits = np.asarray(logits)
        boxes = np.asarray(boxes)

        results: list[tuple[list[int], np.ndarray, list[float]]] = []

        for b in range(boxes.shape[0]):
            # Convert logits to probabilities and get scores and labels
            exp = np.exp(logits[b] - logits[b].max(axis=-1, keepdims=True))
            prob = exp / exp.sum(axis=-1, keepdims=True)

            scores = prob[:, 1:].max(axis=-1)
            labels = prob[:, 1:].argmax(axis=-1) + 1

            # Keep only topk predictions before NMS
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

            polys, _ = (
                self._decode_boxes(bboxes)
                if len(bboxes) > 0
                else (
                    np.zeros((0, 4, 2), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )
            )

            keep = self._nms(polys, scores_b) if len(polys) > 0 else []

            final_labels = []
            final_boxes = []
            final_scores = []

            for idx in keep:
                poly = polys[idx].reshape(-1).tolist()
                if self.assume_straight_pages:
                    x_coords = poly[0::2]
                    y_coords = poly[1::2]
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    final_boxes.append([xmin, ymin, xmax, ymax])
                else:
                    final_boxes.append(poly)

                final_labels.append(int(labels_b[idx]))
                final_scores.append(float(scores_b[idx]))

            final_boxes_arr = (
                np.asarray(final_boxes, dtype=np.float32).reshape(-1, 4, 2)
                if not self.assume_straight_pages
                else np.asarray(final_boxes, dtype=np.float32).reshape(-1, 4)
            )

            results.append((
                final_labels,
                final_boxes_arr,
                final_scores,
            ))

        return results


class _LWDETR(BaseModel):
    """LW-DETR as described in `"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
    <https://arxiv.org/pdf/2406.03459v1>`_.
    """

    def build_target(
        self,
        target: list[tuple[list[int], np.ndarray]],
    ):

        targets = []

        def _quad_to_obb(poly: np.ndarray):
            p1, p2, p3, p4 = poly

            cx, cy = np.mean(poly, axis=0)

            w = (np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p4)) / 2
            h = (np.linalg.norm(p3 - p2) + np.linalg.norm(p4 - p1)) / 2

            theta = np.arctan2(*(p2 - p1)[::-1])

            return np.array(
                [cx, cy, w, h, np.sin(theta), np.cos(theta)],
                dtype=np.float32,
            )

        for class_ids, boxes in target:
            boxes_all = []
            labels_all = []

            if len(boxes) == 0:
                targets.append({
                    "boxes": np.zeros((0, 6), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64),
                })
                continue

            for cls_id, box in zip(np.asarray(class_ids), np.asarray(boxes)):
                poly = box.reshape(4, 2)
                obb = _quad_to_obb(poly)

                if obb[2] <= 1e-3 or obb[3] <= 1e-3:
                    continue

                boxes_all.append(obb)
                labels_all.append(cls_id + 1)  # background = 0

            targets.append({
                "boxes": np.asarray(boxes_all, dtype=np.float32),
                "labels": np.asarray(labels_all, dtype=np.int64),
            })

        return targets
