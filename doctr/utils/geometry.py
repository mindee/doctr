# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from math import ceil
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .common_types import BoundingBox, Polygon4P

__all__ = [
    "bbox_to_polygon",
    "polygon_to_bbox",
    "resolve_enclosing_bbox",
    "resolve_enclosing_rbbox",
    "rotate_boxes",
    "compute_expanded_shape",
    "rotate_image",
    "estimate_page_angle",
    "convert_to_relative_coords",
    "rotate_abs_geoms",
    "extract_crops",
    "extract_rcrops",
]


def bbox_to_polygon(bbox: BoundingBox) -> Polygon4P:
    """Convert a bounding box to a polygon

    Args:
    ----
        bbox: a bounding box

    Returns:
    -------
        a polygon
    """
    return bbox[0], (bbox[1][0], bbox[0][1]), (bbox[0][0], bbox[1][1]), bbox[1]


def polygon_to_bbox(polygon: Polygon4P) -> BoundingBox:
    """Convert a polygon to a bounding box

    Args:
    ----
        polygon: a polygon

    Returns:
    -------
        a bounding box
    """
    x, y = zip(*polygon)
    return (min(x), min(y)), (max(x), max(y))


def resolve_enclosing_bbox(bboxes: Union[List[BoundingBox], np.ndarray]) -> Union[BoundingBox, np.ndarray]:
    """Compute enclosing bbox either from:

    Args:
    ----
        bboxes: boxes in one of the following formats:

            - an array of boxes: (*, 5), where boxes have this shape:
            (xmin, ymin, xmax, ymax, score)

            - a list of BoundingBox

    Returns:
    -------
        a (1, 5) array (enclosing boxarray), or a BoundingBox
    """
    if isinstance(bboxes, np.ndarray):
        xmin, ymin, xmax, ymax, score = np.split(bboxes, 5, axis=1)
        return np.array([xmin.min(), ymin.min(), xmax.max(), ymax.max(), score.mean()])
    else:
        x, y = zip(*[point for box in bboxes for point in box])
        return (min(x), min(y)), (max(x), max(y))


def resolve_enclosing_rbbox(rbboxes: List[np.ndarray], intermed_size: int = 1024) -> np.ndarray:
    """Compute enclosing rotated bbox either from:

    Args:
    ----
        rbboxes: boxes in one of the following formats:

            - an array of boxes: (*, 5), where boxes have this shape:
            (xmin, ymin, xmax, ymax, score)

            - a list of BoundingBox
        intermed_size: size of the intermediate image

    Returns:
    -------
        a (1, 5) array (enclosing boxarray), or a BoundingBox
    """
    cloud: np.ndarray = np.concatenate(rbboxes, axis=0)
    # Convert to absolute for minAreaRect
    cloud *= intermed_size
    rect = cv2.minAreaRect(cloud.astype(np.int32))
    return cv2.boxPoints(rect) / intermed_size


def rotate_abs_points(points: np.ndarray, angle: float = 0.0) -> np.ndarray:
    """Rotate points counter-clockwise.

    Args:
    ----
        points: array of size (N, 2)
        angle: angle between -90 and +90 degrees

    Returns:
    -------
        Rotated points
    """
    angle_rad = angle * np.pi / 180.0  # compute radian angle for np functions
    rotation_mat = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]], dtype=points.dtype
    )
    return np.matmul(points, rotation_mat.T)


def compute_expanded_shape(img_shape: Tuple[int, int], angle: float) -> Tuple[int, int]:
    """Compute the shape of an expanded rotated image

    Args:
    ----
        img_shape: the height and width of the image
        angle: angle between -90 and +90 degrees

    Returns:
    -------
        the height and width of the rotated image
    """
    points: np.ndarray = np.array(
        [
            [img_shape[1] / 2, img_shape[0] / 2],
            [-img_shape[1] / 2, img_shape[0] / 2],
        ]
    )

    rotated_points = rotate_abs_points(points, angle)

    wh_shape = 2 * np.abs(rotated_points).max(axis=0)
    return wh_shape[1], wh_shape[0]


def rotate_abs_geoms(
    geoms: np.ndarray,
    angle: float,
    img_shape: Tuple[int, int],
    expand: bool = True,
) -> np.ndarray:
    """Rotate a batch of bounding boxes or polygons by an angle around the
    image center.

    Args:
    ----
        geoms: (N, 4) or (N, 4, 2) array of ABSOLUTE coordinate boxes
        angle: anti-clockwise rotation angle in degrees
        img_shape: the height and width of the image
        expand: whether the image should be padded to avoid information loss

    Returns:
    -------
        A batch of rotated polygons (N, 4, 2)
    """
    # Switch to polygons
    polys = (
        np.stack([geoms[:, [0, 1]], geoms[:, [2, 1]], geoms[:, [2, 3]], geoms[:, [0, 3]]], axis=1)
        if geoms.ndim == 2
        else geoms
    )
    polys = polys.astype(np.float32)

    # Switch to image center as referential
    polys[..., 0] -= img_shape[1] / 2
    polys[..., 1] = img_shape[0] / 2 - polys[..., 1]

    # Rotated them around image center
    rotated_polys = rotate_abs_points(polys.reshape(-1, 2), angle).reshape(-1, 4, 2)
    # Switch back to top-left corner as referential
    target_shape = compute_expanded_shape(img_shape, angle) if expand else img_shape
    # Clip coords to fit since there is no expansion
    rotated_polys[..., 0] = (rotated_polys[..., 0] + target_shape[1] / 2).clip(0, target_shape[1])
    rotated_polys[..., 1] = (target_shape[0] / 2 - rotated_polys[..., 1]).clip(0, target_shape[0])

    return rotated_polys


def remap_boxes(loc_preds: np.ndarray, orig_shape: Tuple[int, int], dest_shape: Tuple[int, int]) -> np.ndarray:
    """Remaps a batch of rotated locpred (N, 4, 2) expressed for an origin_shape to a destination_shape.
    This does not impact the absolute shape of the boxes, but allow to calculate the new relative RotatedBbox
    coordinates after a resizing of the image.

    Args:
    ----
        loc_preds: (N, 4, 2) array of RELATIVE loc_preds
        orig_shape: shape of the origin image
        dest_shape: shape of the destination image

    Returns:
    -------
        A batch of rotated loc_preds (N, 4, 2) expressed in the destination referencial
    """
    if len(dest_shape) != 2:
        raise ValueError(f"Mask length should be 2, was found at: {len(dest_shape)}")
    if len(orig_shape) != 2:
        raise ValueError(f"Image_shape length should be 2, was found at: {len(orig_shape)}")
    orig_height, orig_width = orig_shape
    dest_height, dest_width = dest_shape
    mboxes = loc_preds.copy()
    mboxes[:, :, 0] = ((loc_preds[:, :, 0] * orig_width) + (dest_width - orig_width) / 2) / dest_width
    mboxes[:, :, 1] = ((loc_preds[:, :, 1] * orig_height) + (dest_height - orig_height) / 2) / dest_height

    return mboxes


def rotate_boxes(
    loc_preds: np.ndarray,
    angle: float,
    orig_shape: Tuple[int, int],
    min_angle: float = 1.0,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax, c) or rotated bounding boxes
    (4, 2) of an angle, if angle > min_angle, around the center of the page.
    If target_shape is specified, the boxes are remapped to the target shape after the rotation. This
    is done to remove the padding that is created by rotate_page(expand=True)

    Args:
    ----
        loc_preds: (N, 5) or (N, 4, 2) array of RELATIVE boxes
        angle: angle between -90 and +90 degrees
        orig_shape: shape of the origin image
        min_angle: minimum angle to rotate boxes
        target_shape: shape of the destination image

    Returns:
    -------
        A batch of rotated boxes (N, 4, 2): or a batch of straight bounding boxes
    """
    # Change format of the boxes to rotated boxes
    _boxes = loc_preds.copy()
    if _boxes.ndim == 2:
        _boxes = np.stack(
            [
                _boxes[:, [0, 1]],
                _boxes[:, [2, 1]],
                _boxes[:, [2, 3]],
                _boxes[:, [0, 3]],
            ],
            axis=1,
        )
    # If small angle, return boxes (no rotation)
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return _boxes
    # Compute rotation matrix
    angle_rad = angle * np.pi / 180.0  # compute radian angle for np functions
    rotation_mat = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]], dtype=_boxes.dtype
    )
    # Rotate absolute points
    points: np.ndarray = np.stack((_boxes[:, :, 0] * orig_shape[1], _boxes[:, :, 1] * orig_shape[0]), axis=-1)
    image_center = (orig_shape[1] / 2, orig_shape[0] / 2)
    rotated_points = image_center + np.matmul(points - image_center, rotation_mat)
    rotated_boxes: np.ndarray = np.stack(
        (rotated_points[:, :, 0] / orig_shape[1], rotated_points[:, :, 1] / orig_shape[0]), axis=-1
    )

    # Apply a mask if requested
    if target_shape is not None:
        rotated_boxes = remap_boxes(rotated_boxes, orig_shape=orig_shape, dest_shape=target_shape)

    return rotated_boxes


def rotate_image(
    image: np.ndarray,
    angle: float,
    expand: bool = False,
    preserve_origin_shape: bool = False,
) -> np.ndarray:
    """Rotate an image counterclockwise by an given angle.

    Args:
    ----
        image: numpy tensor to rotate
        angle: rotation angle in degrees, between -90 and +90
        expand: whether the image should be padded before the rotation
        preserve_origin_shape: if expand is set to True, resizes the final output to the original image size

    Returns:
    -------
        Rotated array, padded by 0 by default.
    """
    # Compute the expanded padding
    exp_img: np.ndarray
    if expand:
        exp_shape = compute_expanded_shape(image.shape[:2], angle)  # type: ignore[arg-type]
        h_pad, w_pad = (
            int(max(0, ceil(exp_shape[0] - image.shape[0]))),
            int(max(0, ceil(exp_shape[1] - image.shape[1]))),
        )
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


def estimate_page_angle(polys: np.ndarray) -> float:
    """Takes a batch of rotated previously ORIENTED polys (N, 4, 2) (rectified by the classifier) and return the
    estimated angle ccw in degrees
    """
    # Compute mean left points and mean right point with respect to the reading direction (oriented polygon)
    xleft = polys[:, 0, 0] + polys[:, 3, 0]
    yleft = polys[:, 0, 1] + polys[:, 3, 1]
    xright = polys[:, 1, 0] + polys[:, 2, 0]
    yright = polys[:, 1, 1] + polys[:, 2, 1]
    return float(np.median(np.arctan((yleft - yright) / (xright - xleft))) * 180 / np.pi)  # Y axis from top to bottom!


def convert_to_relative_coords(geoms: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Convert a geometry to relative coordinates

    Args:
    ----
        geoms: a set of polygons of shape (N, 4, 2) or of straight boxes of shape (N, 4)
        img_shape: the height and width of the image

    Returns:
    -------
        the updated geometry
    """
    # Polygon
    if geoms.ndim == 3 and geoms.shape[1:] == (4, 2):
        polygons: np.ndarray = np.empty(geoms.shape, dtype=np.float32)
        polygons[..., 0] = geoms[..., 0] / img_shape[1]
        polygons[..., 1] = geoms[..., 1] / img_shape[0]
        return polygons.clip(0, 1)
    if geoms.ndim == 2 and geoms.shape[1] == 4:
        boxes: np.ndarray = np.empty(geoms.shape, dtype=np.float32)
        boxes[:, ::2] = geoms[:, ::2] / img_shape[1]
        boxes[:, 1::2] = geoms[:, 1::2] / img_shape[0]
        return boxes.clip(0, 1)

    raise ValueError(f"invalid format for arg `geoms`: {geoms.shape}")


def extract_crops(img: np.ndarray, boxes: np.ndarray, channels_last: bool = True) -> List[np.ndarray]:
    """Created cropped images from list of bounding boxes

    Args:
    ----
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)
        channels_last: whether the channel dimensions is the last one instead of the last one

    Returns:
    -------
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError("boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)")

    # Project relative coordinates
    _boxes = boxes.copy()
    h, w = img.shape[:2] if channels_last else img.shape[-2:]
    if not np.issubdtype(_boxes.dtype, np.integer):
        _boxes[:, [0, 2]] *= w
        _boxes[:, [1, 3]] *= h
        _boxes = _boxes.round().astype(int)
        # Add last index
        _boxes[2:] += 1
    if channels_last:
        return deepcopy([img[box[1] : box[3], box[0] : box[2]] for box in _boxes])

    return deepcopy([img[:, box[1] : box[3], box[0] : box[2]] for box in _boxes])


def extract_rcrops(
    img: np.ndarray, polys: np.ndarray, dtype=np.float32, channels_last: bool = True
) -> List[np.ndarray]:
    """Created cropped images from list of rotated bounding boxes

    Args:
    ----
        img: input image
        polys: bounding boxes of shape (N, 4, 2)
        dtype: target data type of bounding boxes
        channels_last: whether the channel dimensions is the last one instead of the last one

    Returns:
    -------
        list of cropped images
    """
    if polys.shape[0] == 0:
        return []
    if polys.shape[1:] != (4, 2):
        raise AssertionError("polys are expected to be quadrilateral, of shape (N, 4, 2)")

    # Project relative coordinates
    _boxes = polys.copy()
    height, width = img.shape[:2] if channels_last else img.shape[-2:]
    if not np.issubdtype(_boxes.dtype, np.integer):
        _boxes[:, :, 0] *= width
        _boxes[:, :, 1] *= height

    src_pts = _boxes[:, :3].astype(np.float32)
    # Preserve size
    d1 = np.linalg.norm(src_pts[:, 0] - src_pts[:, 1], axis=-1)
    d2 = np.linalg.norm(src_pts[:, 1] - src_pts[:, 2], axis=-1)
    # (N, 3, 2)
    dst_pts = np.zeros((_boxes.shape[0], 3, 2), dtype=dtype)
    dst_pts[:, 1, 0] = dst_pts[:, 2, 0] = d1 - 1
    dst_pts[:, 2, 1] = d2 - 1
    # Use a warp transformation to extract the crop
    crops = [
        cv2.warpAffine(
            img if channels_last else img.transpose(1, 2, 0),
            # Transformation matrix
            cv2.getAffineTransform(src_pts[idx], dst_pts[idx]),
            (int(d1[idx]), int(d2[idx])),
        )
        for idx in range(_boxes.shape[0])
    ]
    return crops
