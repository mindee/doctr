# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from math import floor
from statistics import median_low
from typing import List

import cv2
import numpy as np

__all__ = ['estimate_orientation', 'extract_crops', 'extract_rcrops', 'get_bitmap_angle']


def extract_crops(img: np.ndarray, boxes: np.ndarray, channels_last: bool = True) -> List[np.ndarray]:
    """Created cropped images from list of bounding boxes

    Args:
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)
        channels_last: whether the channel dimensions is the last one instead of the last one

    Returns:
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError("boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)")

    # Project relative coordinates
    _boxes = boxes.copy()
    h, w = img.shape[:2] if channels_last else img.shape[-2:]
    if _boxes.dtype != int:
        _boxes[:, [0, 2]] *= w
        _boxes[:, [1, 3]] *= h
        _boxes = _boxes.round().astype(int)
        # Add last index
        _boxes[2:] += 1
    if channels_last:
        return [img[box[1]: box[3], box[0]: box[2]] for box in _boxes]
    else:
        return [img[:, box[1]: box[3], box[0]: box[2]] for box in _boxes]


def extract_rcrops(
    img: np.ndarray,
    polys: np.ndarray,
    dtype=np.float32,
    channels_last: bool = True
) -> List[np.ndarray]:
    """Created cropped images from list of rotated bounding boxes

    Args:
        img: input image
        polys: bounding boxes of shape (N, 4, 2)
        dtype: target data type of bounding boxes
        channels_last: whether the channel dimensions is the last one instead of the last one

    Returns:
        list of cropped images
    """
    if polys.shape[0] == 0:
        return []
    if polys.shape[1:] != (4, 2):
        raise AssertionError("polys are expected to be quadrilateral, of shape (N, 4, 2)")

    # Project relative coordinates
    _boxes = polys.copy()
    height, width = img.shape[:2] if channels_last else img.shape[-2:]
    if _boxes.dtype != np.int:
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


def get_max_width_length_ratio(contour: np.ndarray) -> float:
    """Get the maximum shape ratio of a contour.

    Args:
        contour: the contour from cv2.findContour

    Returns: the maximum shape ratio
    """
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w / h, h / w)


def estimate_orientation(img: np.ndarray, n_ct: int = 50, ratio_threshold_for_lines: float = 5) -> float:
    """Estimate the angle of the general document orientation based on the
     lines of the document and the assumption that they should be horizontal.

    Args:
        img: the img to analyze
        n_ct: the number of contours used for the orientation estimation
        ratio_threshold_for_lines: this is the ratio w/h used to discriminates lines

    Returns:
        the angle of the general document orientation
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 5)
    thresh = cv2.threshold(gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # try to merge words in lines
    (h, w) = img.shape[:2]
    k_x = max(1, (floor(w / 100)))
    k_y = max(1, (floor(h / 100)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x, k_y))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours
    contours = sorted(contours, key=get_max_width_length_ratio, reverse=True)

    angles = []
    for contour in contours[:n_ct]:
        _, (w, h), angle = cv2.minAreaRect(contour)
        if w / h > ratio_threshold_for_lines:  # select only contours with ratio like lines
            angles.append(angle)
        elif w / h < 1 / ratio_threshold_for_lines:  # if lines are vertical, substract 90 degree
            angles.append(angle - 90)

    if len(angles) == 0:
        return 0  # in case no angles is found
    else:
        return -median_low(angles)


def get_bitmap_angle(bitmap: np.ndarray, n_ct: int = 20, std_max: float = 3.) -> float:
    """From a binarized segmentation map, find contours and fit min area rectangles to determine page angle

    Args:
        bitmap: binarized segmentation map
        n_ct: number of contours to use to fit page angle
        std_max: maximum deviation of the angle distribution to consider the mean angle reliable

    Returns:
        The angle of the page
    """
    # Find all contours on binarized seg map
    contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

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


def rectify_crops(
    crops: List[np.ndarray],
    orientations: List[int],
) -> List[np.ndarray]:
    """Rotate each crop of the list according to the predicted orientation:
    0: already straight, no rotation
    1: 90 ccw, rotate 3 times ccw
    2: 180, rotate 2 times ccw
    3: 270 ccw, rotate 1 time ccw
    """
    # Inverse predictions (if angle of +90 is detected, rotate by -90)
    orientations = [4 - pred if pred != 0 else 0 for pred in orientations]
    return [
        crop if orientation == 0 else np.rot90(crop, orientation)
        for orientation, crop in zip(orientations, crops)
    ] if len(orientations) > 0 else []


def rectify_loc_preds(
    page_loc_preds: np.ndarray,
    orientations: List[int],
) -> np.ndarray:
    """Orient the quadrangle (Polygon4P) according to the predicted orientation,
    so that the points are in this order: top L, top R, bot R, bot L if the crop is readable
    """
    return np.stack(
        [np.roll(
            page_loc_pred,
            orientation,
            axis=0) for orientation, page_loc_pred in zip(orientations, page_loc_preds)],
        axis=0
    ) if len(orientations) > 0 else None
