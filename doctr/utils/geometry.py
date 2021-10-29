# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from math import ceil
from typing import List, Union, Tuple, Optional
import numpy as np
import cv2
from .common_types import BoundingBox, Polygon4P, RotatedBbox, Bbox

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


def bbox_to_rbbox(bbox: Bbox) -> RotatedBbox:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1], 0


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
        A batch of rotated boxes (N, 5): (x, y, w, h, alpha) or a batch of straight bounding boxes
    """

    # Get box centers
    box_centers = np.stack((boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]), axis=1) / 2
    img_corners = np.array([[0, 0], [0, img_shape[0]], [*img_shape[::-1]], [img_shape[1], 0]], dtype=boxes.dtype)

    stacked_points = np.concatenate((img_corners, box_centers), axis=0)
    # Y-axis is inverted by conversion
    stacked_rel_points = np.stack(
        (stacked_points[:, 0] - img_shape[1] / 2, img_shape[0] / 2 - stacked_points[:, 1]),
        axis=1
    )

    # Rotate them around image center
    rot_points = rotate_abs_points(stacked_rel_points, angle)
    rot_corners, rot_centers = rot_points[:4], rot_points[4:]

    # Expand the image to fit all the original info
    if expand:
        new_corners = np.abs(rot_corners).max(axis=0)
        rot_centers[:, 0] += new_corners[0]
        rot_centers[:, 1] = new_corners[1] - rot_centers[:, 1]
    else:
        rot_centers[:, 0] += img_shape[1] / 2
        rot_centers[:, 1] = img_shape[0] / 2 - rot_centers[:, 1]

    # Rotated bbox conversion
    rotated_boxes = np.concatenate((
        rot_centers,
        np.stack((boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), axis=1),
        np.full((boxes.shape[0], 1), angle, dtype=box_centers.dtype)
    ), axis=1)

    return rotated_boxes


def remap_boxes(boxes: np.ndarray, orig_shape: Tuple[int, int], dest_shape: Tuple[int, int]) -> np.ndarray:
    """ Remaps a batch of RotatedBbox (x, y, w, h, alpha) expressed for an origin_shape to a destination_shape,
    This does not impact the absolute shape of the boxes

    Args:
        boxes: (N, 5) array of RELATIVE RotatedBbox (x, y, w, h, alpha)
        orig_shape: shape of the origin image
        dest_shape: shape of the destination image

    Returns:
        A batch of rotated boxes (N, 5): (x, y, w, h, alpha) expressed in the destination referencial

    """

    if len(dest_shape) != 2:
        raise ValueError(f"Mask length should be 2, was found at: {len(dest_shape)}")
    if len(orig_shape) != 2:
        raise ValueError(f"Image_shape length should be 2, was found at: {len(orig_shape)}")
    orig_height, orig_width = orig_shape
    dest_height, dest_width = dest_shape
    mboxes = boxes.copy()
    mboxes[:, 0] = ((boxes[:, 0] * orig_width) + (dest_width - orig_width) / 2) / dest_width
    mboxes[:, 1] = ((boxes[:, 1] * orig_height) + (dest_height - orig_height) / 2) / dest_height
    mboxes[:, 2] = boxes[:, 2] * orig_width / dest_width
    mboxes[:, 3] = boxes[:, 3] * orig_height / dest_height
    return mboxes


def rotate_boxes(
    boxes: np.ndarray,
    angle: float = 0.,
    min_angle: float = 1.,
    orig_shape: Optional[Tuple[int, int]] = None,
    mask_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax) of an angle,
    if angle > min_angle, around the center of the page.

    Args:
        boxes: (N, 4) array of RELATIVE boxes
        angle: angle between -90 and +90 degrees
        min_angle: minimum angle to rotate boxes
        orig_shape: shape of the origin image
        mask_shape: shape of the mask if the image is cropped after the rotation

    Returns:
        A batch of rotated boxes (N, 5): (x, y, w, h, alpha) or a batch of straight bounding boxes
    """
    # Change format of the boxes to rotated boxes
    boxes = np.apply_along_axis(bbox_to_rbbox, 1, boxes)
    # If small angle, return boxes (no rotation)
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return boxes
    # Compute rotation matrix
    angle_rad = angle * np.pi / 180.  # compute radian angle for np functions
    rotation_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ], dtype=boxes.dtype)
    # Rotate centers
    centers = np.stack((boxes[:, 0], boxes[:, 1]), axis=-1)
    rotated_centers = .5 + np.matmul(centers - .5, rotation_mat)
    x_center, y_center = rotated_centers[:, 0], rotated_centers[:, 1]
    # Compute rotated boxes
    rotated_boxes = np.stack((x_center, y_center, boxes[:, 2], boxes[:, 3], angle * np.ones_like(boxes[:, 0])), axis=1)
    # Apply a mask if requested
    if mask_shape is not None:
        rotated_boxes = remap_boxes(rotated_boxes, orig_shape=orig_shape, dest_shape=mask_shape)
    return rotated_boxes


def rotate_image(
    image: np.ndarray,
    angle: float,
    expand: bool = False,
    preserve_aspect_ratio: bool = False
) -> np.ndarray:
    """Rotate an image counterclockwise by an given angle.

    Args:
        image: numpy tensor to rotate
        angle: rotation angle in degrees, between -90 and +90
        expand: whether the image should be padded before the rotation
        preserve_aspect_ratio: whether the image should be resized to the original image size after the rotation

    Returns:
        Rotated array, padded by 0 by default.
    """

    # Compute the expanded padding
    if expand:
        exp_shape = compute_expanded_shape(image.shape[:-1], angle)
        h_pad, w_pad = int(max(0, ceil(exp_shape[0] - image.shape[0]))), int(max(0, ceil(exp_shape[1] - image.shape[1])))
        exp_img = np.pad(image, ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)))
    else:
        exp_img = image

    height, width = exp_img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rot_img = cv2.warpAffine(exp_img, rot_mat, (width, height))
    if preserve_aspect_ratio:
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
