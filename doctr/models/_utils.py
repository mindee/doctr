# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
from typing import List

__all__ = ['extract_crops']


def extract_crops(img: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
    """Created cropped images from list of bounding boxes

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
        _boxes = _boxes.round().astype(int)

    crops = []
    for box in _boxes:
        x, y, w, h, alpha = box
        src_pts = cv2.boxPoints(((x, y), (w, h), alpha))
        # Preserve size
        dst_pts = np.array([[0, h-1], [0, 0], [w-1, 0], [w-1, h-1]], dtype=np.float32)
        # The transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # Warp the rotated rectangle
        crop = cv2.warpPerspective(img, M, (w, h))
        crops.append(crop)

    return crops
