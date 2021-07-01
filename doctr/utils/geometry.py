# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Union
import numpy as np
import cv2
from .common_types import BoundingBox, Polygon4P, RotatedBbox

__all__ = ['rbbox_to_polygon', 'bbox_to_polygon', 'polygon_to_bbox', 'polygon_to_rbbox',
           'resolve_enclosing_bbox', 'resolve_enclosing_bbox', 'fit_rbbox']


def bbox_to_polygon(bbox: BoundingBox) -> Polygon4P:
    return bbox[0], (bbox[1][0], bbox[0][1]), (bbox[0][0], bbox[1][1]), bbox[1]


def rbbox_to_polygon(rbbox: RotatedBbox) -> Polygon4P:
    (x, y, w, h, alpha) = rbbox
    return cv2.boxPoints(((float(x), float(y)), (float(w), float(h)), float(alpha)))


def fit_rbbox(pts: np.ndarray) -> RotatedBbox:
    ((x, y), (w, h), alpha) = cv2.minAreaRect(pts)
    return x, y, w, h, alpha


def polygon_to_bbox(polygon: Polygon4P) -> BoundingBox:
    x, y = zip(*polygon)
    return (min(x), min(y)), (max(x), max(y))


def polygon_to_rbbox(polygon: Polygon4P) -> RotatedBbox:
    cnt = np.array(polygon).reshape((-1, 1, 2)).astype(np.float32)
    return fit_rbbox(cnt)


def resolve_enclosing_bbox(bboxes: Union[List[BoundingBox], np.ndarray]) -> Union[BoundingBox, np.ndarray]:
    """Compute enclosing bbox either from:

    - an array of boxes: (*, 5), where boxes have this shape:
    (xmin, ymin, xmax, ymax, score)

    - a list of BoundingBox

    Return a (1, 5) array (enclosing boxarray), or a BoundingBox
    """
    if isinstance(bboxes, np.ndarray):
        xmin, ymin, xmax, ymax, score = np.split(bboxes, 5, axis=1)
        return np.array([xmin.min(), ymin.min(), xmax.max(), ymax.max(), score.mean()])
    else:
        x, y = zip(*[point for box in bboxes for point in box])
        return (min(x), min(y)), (max(x), max(y))


def resolve_enclosing_rbbox(rbboxes: List[RotatedBbox]) -> RotatedBbox:
    pts = np.asarray([pt for rbbox in rbboxes for pt in rbbox_to_polygon(rbbox)], np.float32)
    return fit_rbbox(pts)
