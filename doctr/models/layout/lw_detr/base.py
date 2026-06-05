# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

import cv2
import numpy as np

from doctr.models.core import BaseModel
from doctr.utils import order_points

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

            poly = order_points(cv2.boxPoints(rect))
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

    def __call__(self, logits: np.ndarray, boxes: np.ndarray) -> list[tuple[list[int], np.ndarray, list[float]]]:
        logits = np.asarray(logits)
        boxes = np.asarray(boxes)

        results: list[tuple[list[int], np.ndarray, list[float]]] = []

        for b in range(boxes.shape[0]):
            # Convert logits to probabilities and get scores and labels
            prob = 1.0 / (1.0 + np.exp(-logits[b]))

            # Remove background class
            prob_fg = prob[:, :-1]

            scores = prob_fg.max(axis=-1)
            labels = prob_fg.argmax(axis=-1)

            # Keep only topk predictions before NMS
            if self.topk is not None and len(scores) > self.topk:
                idxs = np.argpartition(-scores, self.topk)[: self.topk]
                idxs = idxs[np.argsort(-scores[idxs])]
            else:
                idxs = np.arange(len(scores))

            scores_b = scores[idxs]
            labels_b = labels[idxs]
            bboxes = boxes[b][idxs]

            # Filter by score threshold
            thresh_mask = scores_b >= self.score_thresh
            scores_b = scores_b[thresh_mask]
            labels_b = labels_b[thresh_mask]
            bboxes = bboxes[thresh_mask]

            polys, _ = (
                self._decode_boxes(bboxes)
                if len(bboxes) > 0
                else (
                    np.zeros((0, 4, 2), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )
            )

            keep = self._nms(polys, scores_b, labels_b) if len(polys) > 0 else []

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
        target: list[dict[str, np.ndarray]],
        class_names: list[str],
    ) -> list[dict[str, Any]]:
        """Build the target for LW-DETR training

        Args:
            target: list of dictionaries where each dictionary corresponds to a sample and has keys corresponding
                to class names and values corresponding to lists of boxes in either polygon format (4, 2)
                or bounding box format (4,) (xmin, ymin, xmax, ymax)
            class_names: list of class names

        Returns:
            list of dictionaries with keys "boxes" and "labels" where "boxes" is an array of shape (num_boxes, 6)
                containing the box parameters in OBB format (cx, cy, w, h, sin(theta), cos(theta))
                and "labels" is an array of shape (num_boxes,) containing the class labels
        """
        targets = []
        class_to_id = {name: i for i, name in enumerate(class_names)}

        def _quad_to_obb(poly: np.ndarray):
            poly = np.asarray(poly, dtype=np.float32)

            # Center point is simply the average of the relative vertices
            cx, cy = np.mean(poly, axis=0)

            edges = np.stack([
                poly[1] - poly[0],
                poly[2] - poly[1],
                poly[3] - poly[2],
                poly[0] - poly[3],
            ])

            lengths = np.linalg.norm(edges, axis=1)
            i = np.argmax(lengths)
            dx, dy = edges[i]

            theta = np.arctan2(dy, dx)

            # Width and height remain cleanly in relative coordinate space [0, 1]
            w = np.mean([lengths[i], lengths[(i + 2) % 4]])
            h = np.mean([lengths[(i + 1) % 4], lengths[(i + 3) % 4]])

            # Enforce strict unit-length normal vectors for rotation
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            norm = np.sqrt(sin_t**2 + cos_t**2) + 1e-8

            return np.array(
                [cx, cy, w, h, sin_t / norm, cos_t / norm],
                dtype=np.float32,
            )

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

                for box in boxes:
                    poly = to_quad(box)
                    obb = _quad_to_obb(poly)

                    # filter out degenerate boxes
                    if obb[2] <= 1e-5 or obb[3] <= 1e-5:
                        continue

                    boxes_all.append(obb)
                    labels_all.append(cls_id)

            targets.append({
                "boxes": np.asarray(boxes_all, dtype=np.float32),
                "labels": np.asarray(labels_all, dtype=np.int64),
            })

        return targets
