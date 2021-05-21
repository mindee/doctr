# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List
import numpy as np
import cv2
from .common_types import BoundingBox, Polygon4P

__all__ = ['bbox_to_polygon', 'polygon_to_bbox', 'resolve_enclosing_bbox']


def bbox_to_polygon(bbox: BoundingBox) -> Polygon4P:
    (x, y, w, h, alpha) = bbox
    pt1 = [x + (w / 2 * np.cos(alpha)) + (h / 2 * np.sin(alpha)), y + (w / 2 * np.sin(alpha)) - (h / 2 * np.cos(alpha))]
    pt2 = [x - (w / 2 * np.cos(alpha)) + (h / 2 * np.sin(alpha)), y - (w / 2 * np.sin(alpha)) - (h / 2 * np.cos(alpha))]
    pt3 = [x - (w / 2 * np.cos(alpha)) - (h / 2 * np.sin(alpha)), y - (w / 2 * np.sin(alpha)) + (h / 2 * np.cos(alpha))]
    pt4 = [x + (w / 2 * np.cos(alpha)) - (h / 2 * np.sin(alpha)), y + (w / 2 * np.sin(alpha)) + (h / 2 * np.cos(alpha))]
    return [pt1, pt2, pt3, pt4]


def polygon_to_bbox(polygon: Polygon4P) -> BoundingBox:
    cnt = np.array(polygon).reshape((-1, 1, 2)).astype(np.float32)
    ((x, y), (w, h), alpha) = cv2.minAreaRect(cnt)
    return (x, y, w, h, alpha)


def resolve_enclosing_bbox(bboxes: List[BoundingBox]) -> BoundingBox:
    pts = np.asarray([pt for bbox in bboxes for pt in bbox_to_polygon(bbox)], np.float32)
    ((x, y), (w, h), alpha) = cv2.minAreaRect(pts)
    return (x, y, w, h, alpha)
