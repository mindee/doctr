# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple, Union

import numpy as np

__all__ = ["crop_boxes"]


def crop_boxes(
    boxes: np.ndarray,
    crop_box: Union[Tuple[int, int, int, int], Tuple[float, float, float, float]],
) -> np.ndarray:
    """Crop localization boxes

    Args:
        boxes: ndarray of shape (N, 4) in relative or abs coordinates
        crop_box: box (xmin, ymin, xmax, ymax) to crop the image, in the same coord format that the boxes

    Returns:
        the cropped boxes
    """
    is_box_rel = boxes.max() <= 1
    is_crop_rel = max(crop_box) <= 1

    if is_box_rel ^ is_crop_rel:
        raise AssertionError("both the boxes and the crop need to have the same coordinate convention")

    xmin, ymin, xmax, ymax = crop_box
    # Clip boxes & correct offset
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(xmin, xmax) - xmin
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(ymin, ymax) - ymin
    # Rescale relative coords
    if is_box_rel:
        boxes[:, [0, 2]] /= (xmax - xmin)
        boxes[:, [1, 3]] /= (ymax - ymin)

    # Remove 0-sized boxes
    is_valid = np.logical_and(boxes[:, 1] < boxes[:, 3], boxes[:, 0] < boxes[:, 2])

    return boxes[is_valid]
