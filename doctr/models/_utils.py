# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
from math import floor
from typing import List
from statistics import median_low

__all__ = ['estimate_orientation', 'extract_crops', 'extract_rcrops', 'rotate_page', 'get_bitmap_angle', 'rotate_boxes']


def extract_crops(img: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
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


def extract_rcrops(img: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
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


def rotate_page(
    image: np.ndarray,
    angle: float = 0.,
    min_angle: float = 1.
) -> np.ndarray:
    """Rotate an image counterclockwise by an ange alpha (negative angle to go clockwise).

    Args:
        image: numpy tensor to rotate
        angle: rotation angle in degrees, between -90 and +90
        min_angle: min. angle in degrees to rotate a page

    Returns:
        Rotated array or tf.Tensor, padded by 0 by default.
    """
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return image
    elif abs(angle) > 45:
        # if image is closer to portrait mode, go for a full 90° rotation
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        angle += 90  # track the rotation by updating angle

    height, width = image.shape[:2]
    center = (height / 2, width / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (width, height),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def estimate_orientation(img: np.ndarray, n_ct: int = 50, ratio_threshold_for_lines: int = 5) -> int:
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
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    angles = []
    for contour in contours[:n_ct]:
        _, (w, h), angle = cv2.minAreaRect(contour)
        if w / h > ratio_threshold_for_lines:  # select only contours with ratio like lines
            angles.append(angle)
        elif w / h < 1 / ratio_threshold_for_lines:  # if lines are vertical, substract 90 degree
            angle -= 90
            angles.append(angle)
    return median_low(angles)


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


def rotate_boxes(
    boxes: np.ndarray,
    angle: float = 0.,
    min_angle: float = 1.
) -> np.ndarray:
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax) of an angle,
    if angle > min_angle, around the center of the page.

    Args:
        boxes: (N, 4) array of (relative) boxes
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
    rotation_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
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
