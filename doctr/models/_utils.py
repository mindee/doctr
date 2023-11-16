# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from math import floor
from statistics import median_low
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from langdetect import LangDetectException, detect_langs

__all__ = ["estimate_orientation", "get_language", "invert_data_structure"]


def get_max_width_length_ratio(contour: np.ndarray) -> float:
    """Get the maximum shape ratio of a contour.

    Args:
    ----
        contour: the contour from cv2.findContour

    Returns:
    -------
        the maximum shape ratio
    """
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w / h, h / w)


def estimate_orientation(img: np.ndarray, n_ct: int = 50, ratio_threshold_for_lines: float = 5) -> int:
    """Estimate the angle of the general document orientation based on the
     lines of the document and the assumption that they should be horizontal.

    Args:
    ----
        img: the img or bitmap to analyze (H, W, C)
        n_ct: the number of contours used for the orientation estimation
        ratio_threshold_for_lines: this is the ratio w/h used to discriminates lines

    Returns:
    -------
        the angle of the general document orientation
    """
    assert len(img.shape) == 3 and img.shape[-1] in [1, 3], f"Image shape {img.shape} not supported"
    max_value = np.max(img)
    min_value = np.min(img)
    if max_value <= 1 and min_value >= 0 or (max_value <= 255 and min_value >= 0 and img.shape[-1] == 1):
        thresh = img.astype(np.uint8)
    if max_value <= 255 and min_value >= 0 and img.shape[-1] == 3:
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
        median = -median_low(angles)
        return round(median) if abs(median) != 0 else 0


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
    return (
        [crop if orientation == 0 else np.rot90(crop, orientation) for orientation, crop in zip(orientations, crops)]
        if len(orientations) > 0
        else []
    )


def rectify_loc_preds(
    page_loc_preds: np.ndarray,
    orientations: List[int],
) -> Optional[np.ndarray]:
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


def get_language(text: str) -> Tuple[str, float]:
    """Get languages of a text using langdetect model.
    Get the language with the highest probability or no language if only a few words or a low probability

    Args:
    ----
        text (str): text

    Returns:
    -------
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
    x: Union[List[Dict[str, Any]], Dict[str, List[Any]]]
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Invert a List of Dict of elements to a Dict of list of elements and the other way around

    Args:
    ----
        x: a list of dictionaries with the same keys or a dictionary of lists of the same length

    Returns:
    -------
        dictionary of list when x is a list of dictionaries or a list of dictionaries when x is dictionary of lists
    """
    if isinstance(x, dict):
        assert len({len(v) for v in x.values()}) == 1, "All the lists in the dictionnary should have the same length."
        return [dict(zip(x, t)) for t in zip(*x.values())]
    elif isinstance(x, list):
        return {k: [dic[k] for dic in x] for k in x[0]}
    else:
        raise TypeError(f"Expected input to be either a dict or a list, got {type(input)} instead.")
