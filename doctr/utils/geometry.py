# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Union
import numpy as np
import cv2
from .common_types import BoundingBox, Polygon4P, RotatedBbox

__all__ = ['rbbox_to_polygon', 'bbox_to_polygon', 'polygon_to_bbox', 'polygon_to_rbbox',
           'resolve_enclosing_bbox', 'resolve_enclosing_bbox', 'fit_rbbox', 'rotate_boxes']


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


def rotate_boxes(
    boxes: np.ndarray,
    angle: float = 0.,
    min_angle: float = 1.
) -> np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax) of an angle,
    if angle > min_angle, around the center of the page.

    Args:
        boxes: (N, 4) array of RELATIVE boxes
        angle: angle between -90 and +90 degrees
        min_angle: minimum angle to rotate boxes

    Returns:
        A batch of rotated boxes (N, 5): (x, y, w, h, alpha) or a batch of straight bounding boxes
    """
    # If small angle, return boxes (no rotation)
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return boxes
    # Compute rotation matrix
    angle_rad = angle * np.pi / 180.  # compute radian angle for np functions
    rotation_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ], dtype=boxes.dtype)
    # Compute unrotated boxes
    x_unrotated, y_unrotated = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    width, height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    # Rotate centers
    centers = np.stack((x_unrotated, y_unrotated), axis=-1)
    rotated_centers = .5 + np.matmul(centers - .5, np.transpose(rotation_mat))
    x_center, y_center = rotated_centers[:, 0], rotated_centers[:, 1]
    # Compute rotated boxes
    rotated_boxes = np.stack((x_center, y_center, width, height, angle * np.ones_like(boxes[:, 0])), axis=1)
    return rotated_boxes
