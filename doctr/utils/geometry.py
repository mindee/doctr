# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
from typing import List, Union, Tuple
import numpy as np
import cv2
from .common_types import BoundingBox, Polygon4P, RotatedBbox

__all__ = ['rbbox_to_polygon', 'bbox_to_polygon', 'polygon_to_bbox', 'polygon_to_rbbox',
           'resolve_enclosing_bbox', 'resolve_enclosing_bbox', 'fit_rbbox', 'rotate_boxes', 'rotate_abs_boxes',
           'compute_expanded_shape', 'rotate_image']


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


def rotate_abs_points(points: np.ndarray, center: np.ndarray, angle: float = 0.) -> np.ndarray:

    # Y-axis is inverted by convention
    rel_points = np.stack((points[:, 0] - center[0], center[1] - points[:, 1]), axis=1)

    angle_rad = angle * np.pi / 180.  # compute radian angle for np functions
    rotation_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ], dtype=rel_points.dtype)

    rotated_rel_points = np.matmul(rel_points, rotation_mat.T)
    rotated_rel_points[:, 0] += center[0]
    rotated_rel_points[:, 1] = center[1] - rotated_rel_points[:, 1]

    return rotated_rel_points


def compute_expanded_shape(img_shape: Tuple[int, int], angle: float) -> Tuple[int, int]:
    """Compute the shape of an expanded rotated image

    Args:
        img_shape: the height and width of the image
        angle: angle between -90 and +90 degrees

    Returns:
        the height and width of the rotated image
    """

    points = np.array([
        [img_shape[1] / 2, img_shape[0] / 2],
        [-img_shape[1] / 2, img_shape[0] / 2],
    ])

    rotated_points = rotate_abs_points(points, np.zeros(2), angle)

    wh_shape = 2 * np.abs(rotated_points).max(axis=0)

    return wh_shape[1], wh_shape[0]


def rotate_abs_boxes(boxes: np.ndarray, angle: float, img_shape: Tuple[int, int]) -> np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax) by an angle around the image center.

    Args:
        boxes: (N, 4) array of absolute coordinate boxes
        angle: angle between -90 and +90 degrees
        img_shape: the height and width of the image

    Returns:
        A batch of rotated boxes (N, 5): (x, y, w, h, alpha) or a batch of straight bounding boxes
    """

    # Get box centers
    box_centers = np.stack((boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]), axis=1) / 2

    # Rotate them around image center
    box_centers = rotate_abs_points(box_centers, np.array(img_shape[::-1]) / 2, angle)

    rotated_boxes = np.concatenate((
        box_centers,
        np.stack((boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), axis=1),
        np.full((boxes.shape[0], 1), angle, dtype=box_centers.dtype)
    ), axis=1)

    return rotated_boxes


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


def rotate_image(
    image: np.ndarray,
    angle: float,
    expand=False,
) -> np.ndarray:
    """Rotate an image counterclockwise by an given angle.

    Args:
        image: numpy tensor to rotate
        angle: rotation angle in degrees, between -90 and +90
        expand: whether the image should be padded before the rotation

    Returns:
        Rotated array, padded by 0 by default.
    """

    # Compute the expanded padding
    if expand:
        exp_shape = compute_expanded_shape(image.shape[:-1], angle)
        h_pad, w_pad = int(math.ceil(exp_shape[0] - image.shape[0])), int(math.ceil(exp_shape[1] - image.shape[1]))
        exp_img = np.pad(image, ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)))
    else:
        exp_img = image

    height, width = exp_img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rot_img = cv2.warpAffine(exp_img, rot_mat, (width, height))
    if expand:
        # Pad to get the same aspect ratio
        if (image.shape[0] / image.shape[1]) != (rot_img.shape[0] / rot_img.shape[1]):
            # Pad width
            if (rot_img.shape[0] / rot_img.shape[1]) > (image.shape[0] / image.shape[1]):
                h_pad, w_pad = 0, int(rot_img.shape[0] * image.shape[1] / image.shape[0] - rot_img.shape[1])
            # Pad height
            else:
                h_pad, w_pad = int(rot_img.shape[1] * image.shape[0] / image.shape[1] - rot_img.shape[0]), 0
            rot_img = np.pad(rot_img, ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)))
        # rescale
        rot_img = cv2.resize(rot_img, image.shape[:-1][::-1], interpolation=cv2.INTER_LINEAR)

    return rot_img
