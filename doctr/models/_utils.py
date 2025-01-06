# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from math import floor
from statistics import median_low
from typing import Any

import cv2
import numpy as np
from langdetect import LangDetectException, detect_langs

from doctr.utils.geometry import rotate_image

__all__ = ["estimate_orientation", "get_language", "invert_data_structure"]


def get_max_width_length_ratio(contour: np.ndarray) -> float:
    """Get the maximum shape ratio of a contour.

    Args:
        contour: the contour from cv2.findContour

    Returns:
        the maximum shape ratio
    """
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w / h, h / w)


def estimate_orientation(
    img: np.ndarray,
    general_page_orientation: tuple[int, float] | None = None,
    n_ct: int = 70,
    ratio_threshold_for_lines: float = 3,
    min_confidence: float = 0.2,
    lower_area: int = 100,
) -> int:
    """Estimate the angle of the general document orientation based on the
     lines of the document and the assumption that they should be horizontal.

    Args:
        img: the img or bitmap to analyze (H, W, C)
        general_page_orientation: the general orientation of the page (angle [0, 90, 180, 270 (-90)], confidence)
            estimated by a model
        n_ct: the number of contours used for the orientation estimation
        ratio_threshold_for_lines: this is the ratio w/h used to discriminates lines
        min_confidence: the minimum confidence to consider the general_page_orientation
        lower_area: the minimum area of a contour to be considered

    Returns:
        the estimated angle of the page (clockwise, negative for left side rotation, positive for right side rotation)
    """
    assert len(img.shape) == 3 and img.shape[-1] in [1, 3], f"Image shape {img.shape} not supported"
    thresh = None
    # Convert image to grayscale if necessary
    if img.shape[-1] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, 5)
        thresh = cv2.threshold(gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    else:
        thresh = img.astype(np.uint8)

    page_orientation, orientation_confidence = general_page_orientation or (None, 0.0)
    if page_orientation and orientation_confidence >= min_confidence:
        # We rotate the image to the general orientation which improves the detection
        # No expand needed bitmap is already padded
        thresh = rotate_image(thresh, -page_orientation)
    else:  # That's only required if we do not work on the detection models bin map
        # try to merge words in lines
        (h, w) = img.shape[:2]
        k_x = max(1, (floor(w / 100)))
        k_y = max(1, (floor(h / 100)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x, k_y))
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    # extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter & Sort contours
    contours = sorted(
        [contour for contour in contours if cv2.contourArea(contour) > lower_area],
        key=get_max_width_length_ratio,
        reverse=True,
    )

    angles = []
    for contour in contours[:n_ct]:
        _, (w, h), angle = cv2.minAreaRect(contour)  # type: ignore[assignment]
        if w / h > ratio_threshold_for_lines:  # select only contours with ratio like lines
            angles.append(angle)
        elif w / h < 1 / ratio_threshold_for_lines:  # if lines are vertical, substract 90 degree
            angles.append(angle - 90)

    if len(angles) == 0:
        estimated_angle = 0  # in case no angles is found
    else:
        median = -median_low(angles)
        estimated_angle = -round(median) if abs(median) != 0 else 0

    # combine with the general orientation and the estimated angle
    if page_orientation and orientation_confidence >= min_confidence:
        # special case where the estimated angle is mostly wrong:
        # case 1: - and + swapped
        # case 2: estimated angle is completely wrong
        # so in this case we prefer the general page orientation
        if abs(estimated_angle) == abs(page_orientation):
            return page_orientation
        estimated_angle = estimated_angle if page_orientation == 0 else page_orientation + estimated_angle
        if estimated_angle > 180:
            estimated_angle -= 360

    return estimated_angle  # return the clockwise angle (negative - left side rotation, positive - right side rotation)


def rectify_crops(
    crops: list[np.ndarray],
    orientations: list[int],
) -> list[np.ndarray]:
    """Rotate each crop of the list according to the predicted orientation:
    0: already straight, no rotation
    1: 90 ccw, rotate 3 times ccw
    2: 180, rotate 2 times ccw
    3: 270 ccw, rotate 1 time ccw
    """
    # Inverse predictions (if angle of +90 is detected, rotate by -90)
    orientations = [4 - pred if pred != 0 else 0 for pred in orientations]
    return (
        [crop if orientation == 0 else np.rot90(crop, orientation) for orientation, crop in zip(orientations, crops)]
        if len(orientations) > 0
        else []
    )


def rectify_loc_preds(
    page_loc_preds: np.ndarray,
    orientations: list[int],
) -> np.ndarray | None:
    """Orient the quadrangle (Polygon4P) according to the predicted orientation,
    so that the points are in this order: top L, top R, bot R, bot L if the crop is readable
    """
    return (
        np.stack(
            [
                np.roll(page_loc_pred, orientation, axis=0)
                for orientation, page_loc_pred in zip(orientations, page_loc_preds)
            ],
            axis=0,
        )
        if len(orientations) > 0
        else None
    )


def get_language(text: str) -> tuple[str, float]:
    """Get languages of a text using langdetect model.
    Get the language with the highest probability or no language if only a few words or a low probability

    Args:
        text (str): text

    Returns:
        The detected language in ISO 639 code and confidence score
    """
    try:
        lang = detect_langs(text.lower())[0]
    except LangDetectException:
        return "unknown", 0.0
    if len(text) <= 1 or (len(text) <= 5 and lang.prob <= 0.2):
        return "unknown", 0.0
    return lang.lang, lang.prob


def invert_data_structure(
    x: list[dict[str, Any]] | dict[str, list[Any]],
) -> list[dict[str, Any]] | dict[str, list[Any]]:
    """Invert a list of dict of elements to a dict of list of elements and the other way around

    Args:
        x: a list of dictionaries with the same keys or a dictionary of lists of the same length

    Returns:
        dictionary of list when x is a list of dictionaries or a list of dictionaries when x is dictionary of lists
    """
    if isinstance(x, dict):
        assert len({len(v) for v in x.values()}) == 1, "All the lists in the dictionnary should have the same length."
        return [dict(zip(x, t)) for t in zip(*x.values())]
    elif isinstance(x, list):
        return {k: [dic[k] for dic in x] for k in x[0]}
    else:
        raise TypeError(f"Expected input to be either a dict or a list, got {type(input)} instead.")
