# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from math import ceil
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .common_types import BoundingBox, Polygon4P

__all__ = ['bbox_to_polygon', 'polygon_to_bbox', 'resolve_enclosing_bbox', 'resolve_enclosing_rbbox',
           'rotate_boxes', 'rotate_abs_boxes', 'compute_expanded_shape', 'rotate_image', 'estimate_page_angle']


def bbox_to_polygon(bbox: BoundingBox) -> Polygon4P:
    return bbox[0], (bbox[1][0], bbox[0][1]), (bbox[0][0], bbox[1][1]), bbox[1]


def polygon_to_bbox(polygon: Polygon4P) -> BoundingBox:
    x, y = zip(*polygon)
    return (min(x), min(y)), (max(x), max(y))


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


def resolve_enclosing_rbbox(rbboxes: List[np.ndarray]) -> np.ndarray:
    cloud = np.concatenate(rbboxes, axis=0)
    # Convert to absolute for minAreaRect
    cloud *= 1024
    rect = cv2.minAreaRect(cloud.astype(np.int32))
    return cv2.boxPoints(rect) / 1024


def rotate_abs_points(points: np.ndarray, angle: float = 0.) -> np.ndarray:
    """Rotate points counter-clockwise"""

    angle_rad = angle * np.pi / 180.  # compute radian angle for np functions
    rotation_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ], dtype=points.dtype)

    return np.matmul(points, rotation_mat.T)


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

    rotated_points = rotate_abs_points(points, angle)

    wh_shape = 2 * np.abs(rotated_points).max(axis=0)

    return wh_shape[1], wh_shape[0]


def rotate_abs_boxes(boxes: np.ndarray, angle: float, img_shape: Tuple[int, int], expand: bool = True) -> np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax) by an angle around the image center.

    Args:
        boxes: (N, 4) array of absolute coordinate boxes
        angle: angle between -90 and +90 degrees
        img_shape: the height and width of the image
        expand: whether the image should be padded to avoid information loss

    Returns:
        A batch of rotated boxes (N, 4, 2) or a batch of straight bounding boxes
    """

    # Get box corners
    box_corners = np.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        axis=1
    )
    img_corners = np.array([[0, 0], [0, img_shape[0]], [*img_shape[::-1]], [img_shape[1], 0]], dtype=boxes.dtype)

    stacked_points = np.concatenate((img_corners, np.expand_dims(box_corners, 0)), axis=0)
    # Y-axis is inverted by conversion
    stacked_rel_points = np.stack(
        (stacked_points[:, :, 0] - img_shape[1] / 2, img_shape[0] / 2 - stacked_points[:, :, 1]),
        axis=1
    )

    # Rotate them around image center
    rot_points = rotate_abs_points(stacked_rel_points, angle)
    img_rot_corners, box_rot_corners = rot_points[:4], rot_points[4:]

    # Expand the image to fit all the original info
    if expand:
        new_corners = np.abs(img_rot_corners).max(axis=0)
        box_rot_corners[:, :, 0] += new_corners[0]
        box_rot_corners[:, :, 1] = new_corners[1] - box_rot_corners[:, :, 1]
    else:
        box_rot_corners[:, :, 0] += img_shape[1] / 2
        box_rot_corners[:, :, 1] = img_shape[0] / 2 - box_rot_corners[:, :, 1]

    return box_rot_corners


# TODO: add a deprecation warning, since we don't need to remap polygons points
def remap_boxes(loc_preds: np.ndarray, orig_shape: Tuple[int, int], dest_shape: Tuple[int, int]) -> np.ndarray:
    """ Remaps a batch of rotated locpred (x, y, w, h, alpha, c) expressed for an origin_shape to a destination_shape.
    This does not impact the absolute shape of the boxes, but allow to calculate the new relative RotatedBbox
    coordinates after a resizing of the image.

    Args:
        loc_preds: (N, 6) array of RELATIVE locpred (x, y, w, h, alpha, c)
        orig_shape: shape of the origin image
        dest_shape: shape of the destination image

    Returns:
        A batch of rotated loc_preds (N, 6): (x, y, w, h, alpha, c) expressed in the destination referencial

    """

    if len(dest_shape) != 2:
        raise ValueError(f"Mask length should be 2, was found at: {len(dest_shape)}")
    if len(orig_shape) != 2:
        raise ValueError(f"Image_shape length should be 2, was found at: {len(orig_shape)}")
    orig_height, orig_width = orig_shape
    dest_height, dest_width = dest_shape
    mboxes = loc_preds.copy()
    # remaps position of the box center for the destination image shape
    mboxes[:, 0] = ((loc_preds[:, 0] * orig_width) + (dest_width - orig_width) / 2) / dest_width
    mboxes[:, 1] = ((loc_preds[:, 1] * orig_height) + (dest_height - orig_height) / 2) / dest_height
    # remaps box dimension for the destination image shape
    mboxes[:, 2] = loc_preds[:, 2] * orig_width / dest_width
    mboxes[:, 3] = loc_preds[:, 3] * orig_height / dest_height
    return mboxes


def rotate_boxes(
    loc_preds: np.ndarray,
    angle: float,
    orig_shape: Tuple[int, int],
    min_angle: float = 1.,
) -> np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax, c) or rotated bounding boxes
    (4, 2) of an angle, if angle > min_angle, around the center of the page.
    If target_shape is specified, the boxes are remapped to the target shape after the rotation. This
    is done to remove the padding that is created by rotate_page(expand=True)

    Args:
        loc_preds: (N, 5) or (N, 4, 2) array of RELATIVE boxes
        angle: angle between -90 and +90 degrees
        orig_shape: shape of the origin image
        min_angle: minimum angle to rotate boxes
        target_shape: shape of the target image

    Returns:
        A batch of rotated boxes (N, 4, 2): or a batch of straight bounding boxes
    """

    # Change format of the boxes to rotated boxes
    _boxes = loc_preds.copy()
    if _boxes.shape[1] == 5:
        _boxes = np.stack(
            [
                _boxes[:, [0, 1]],
                _boxes[:, [2, 1]],
                _boxes[:, [2, 3]],
                _boxes[:, [0, 3]],
            ],
            axis=1
        )
    # If small angle, return boxes (no rotation)
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return _boxes
    # Compute rotation matrix
    angle_rad = angle * np.pi / 180.  # compute radian angle for np functions
    rotation_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ], dtype=_boxes.dtype)
    # Rotate absolute points
    points = np.stack((_boxes[:, :, 0] * orig_shape[1], _boxes[:, :, 1] * orig_shape[0]), axis=-1)
    image_center = (orig_shape[1] / 2, orig_shape[0] / 2)
    rotated_points = image_center + np.matmul(points - image_center, rotation_mat)
    rotated_boxes = np.stack((rotated_points[:, :, 0] / orig_shape[1], rotated_points[:, :, 1] / orig_shape[0]), axis=-1)

    return rotated_boxes


def rotate_image(
    image: np.ndarray,
    angle: float,
    expand: bool = False,
    preserve_origin_shape: bool = False,
) -> np.ndarray:
    """Rotate an image counterclockwise by an given angle.

    Args:
        image: numpy tensor to rotate
        angle: rotation angle in degrees, between -90 and +90
        expand: whether the image should be padded before the rotation
        preserve_origin_shape: if expand is set to True, resizes the final output to the original image size

    Returns:
        Rotated array, padded by 0 by default.
    """

    # Compute the expanded padding
    if expand:
        exp_shape = compute_expanded_shape(image.shape[:-1], angle)
        h_pad, w_pad = int(max(0, ceil(exp_shape[0] - image.shape[0]))), int(
            max(0, ceil(exp_shape[1] - image.shape[1])))
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
        if preserve_origin_shape:
            # rescale
            rot_img = cv2.resize(rot_img, image.shape[:-1][::-1], interpolation=cv2.INTER_LINEAR)

    return rot_img


def estimate_page_angle(boxes: np.ndarray) -> float:
    """Takes a batch of rotated previously ORIENTED boxes (N, 4, 2) (rectified by the classifier) and return the
    estimated angle ccw in degrees 
    """
    return np.mean(np.arctan2(
        (boxes[:, 1, 1] - boxes[:, 0, 1]),
        (boxes[:, 1, 0] - boxes[:, 0, 0])
    )) * 180 / np.pi
