# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
import tensorflow as tf
from typing import List, Union

__all__ = ['extract_crops', 'extract_rcrops', 'rotate_page', 'get_bitmap_angle']


def extract_crops(img: Union[np.ndarray, tf.Tensor], boxes: np.ndarray) -> List[Union[np.ndarray, tf.Tensor]]:
    """Created cropped images from list of bounding boxes

    Args:
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)

    Returns:
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError("boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)")

    # Project relative coordinates
    _boxes = boxes.copy()
    if _boxes.dtype != np.int:
        _boxes[:, [0, 2]] *= img.shape[1]
        _boxes[:, [1, 3]] *= img.shape[0]
        _boxes = _boxes.round().astype(int)
        # Add last index
        _boxes[2:] += 1
    return [img[box[1]: box[3], box[0]: box[2]] for box in _boxes]


def extract_rcrops(img: Union[np.ndarray, tf.Tensor], boxes: np.ndarray) -> List[np.ndarray]:
    """Created cropped images from list of rotated bounding boxes

    Args:
        img: input image
        boxes: bounding boxes of shape (N, 5) where N is the number of boxes, and the relative
            coordinates (x, y, w, h, alpha)

    Returns:
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 5:
        raise AssertionError("boxes are expected to be relative and in order (x, y, w, h, alpha)")

    if isinstance(img, tf.Tensor):
        img = img.numpy().astype(np.uint8)

    # Project relative coordinates
    _boxes = boxes.copy()
    if _boxes.dtype != np.int:
        _boxes[:, [0, 2]] *= img.shape[1]
        _boxes[:, [1, 3]] *= img.shape[0]

    crops = []
    # Determine rotation direction (clockwise/counterclockwise)
    # Angle coverage: [-90°, +90°], half of the quadrant
    clockwise = False
    if np.sum(boxes[:, 2]) > np.sum(boxes[:, 3]):
        clockwise = True

    for box in _boxes:
        x, y, w, h, alpha = box.astype(np.float32)
        src_pts = cv2.boxPoints(((x, y), (w, h), alpha))[1:, :]
        # Preserve size
        if clockwise:
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)
        else:
            dst_pts = np.array([[h - 1, 0], [h - 1, w - 1], [0, w - 1]], dtype=np.float32)
        # The transformation matrix
        M = cv2.getAffineTransform(src_pts, dst_pts)
        # Warp the rotated rectangle
        if clockwise:
            crop = cv2.warpAffine(img, M, (int(w), int(h)))
        else:
            crop = cv2.warpAffine(img, M, (int(h), int(w)))
        crops.append(crop)

    return crops


def rotate_page(image: np.array, angle: float = 0., min_angle: float = 1.) -> np.array:
    """Rotate an image counterclockwise by an ange alpha

    Args:
        alpha: angle, degrees, between -90 and +90
        image: np array to rotate
        min_angle: min. angle in degrees to rotate a page

    Returns:
        Rotated np array
    """
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return image
    height, width = image.shape[:2]
    center=tuple(np.array([height, width]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (width, height))
    return rotated


def get_bitmap_angle(bitmap: np.array, n_ct: int = 5, std_max: float = 3.) -> float:
    """From a binarized segmentation map, find contours and fit min area rectangles to determine page angle

    Args:
        bitmap: binarized segmentation map
        n_ct: number of contours to use to fit page angle
        std_max: maximum deviation of the angle distribution to consider the mean angle reliable

    Returns:
        The angle of the page
    """
    # Find all contours on binarized seg map
    contours, hierarchy = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contours and fit angles
    # Track heights and widths to find aspect ratio (determine is rotation is clockwise)
    angles, heights, widths = [], [], []
    for ct in contours[:n_ct]:
        _, (w, h), alpha = cv2.minAreaRect(ct)
        widths.append(w)
        heights.append(h)
        angles.append(alpha)

    if np.std(angles) > std_max:
        # Edge case with angles of both 0 and 90°, or multi_oriented docs
        angle = 0.
    else:
        angle = -np.mean(angles)
        # Determine rotation direction (clockwise/counterclockwise)
        # Angle coverage: [-90°, +90°], half of the quadrant
        if np.sum(widths) < np.sum(heights):  # CounterClockwise
            angle = 90 + angle

    return angle
