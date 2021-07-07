# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torchvision.transforms import functional as F
from copy import deepcopy
import numpy as np
from typing import Tuple
from doctr.utils.geometry import rotate_boxes

__all__ = ["invert_colors", "rotate"]


def invert_colors(img: torch.Tensor, min_val: float = 0.6) -> torch.Tensor:
    out = F.rgb_to_grayscale(img, num_output_channels=3)
    # Random RGB shift
    shift_shape = [img.shape[0], 3, 1, 1] if img.ndim == 4 else [3, 1, 1]
    rgb_shift = min_val + (1 - min_val) * torch.rand(shift_shape)
    # Inverse the color
    if out.dtype == torch.uint8:
        out = (out.to(dtype=torch.float32) * rgb_shift).to(dtype=torch.uint8)
    else:
        out = out * rgb_shift
    # Inverse the color
    out = 255 - out if out.dtype == torch.uint8 else 1 - out
    return out


def rotate(
    img: torch.Tensor,
    boxes: np.ndarray,
    angle: float
) -> Tuple[torch.Tensor, np.ndarray]:
    """Rotate image around the center, interpolation=NEAREST, pad with 0 (black)

    Args:
        img: image to rotate
        boxes: array of boxes to rotate as well
        angle: angle in degrees. +: counter-clockwise, -: clockwise

    Returns:
        A tuple of rotated img (tensor), rotated boxes (np array)
    """
    rotated_img = F.rotate(img, angle=angle, fill=0)  # Interpolation NEAREST by default
    _boxes = deepcopy(boxes)
    if boxes.dtype == int:
        # Compute relative boxes
        _boxes = _boxes.astype(float)
        _boxes[:, [0, 2]] = _boxes[:, [0, 2]] / img.shape[2]
        _boxes[:, [1, 3]] = _boxes[:, [1, 3]] / img.shape[1]
    # Compute rotated bboxes: xmin, ymin, xmax, ymax --> x, y, w, h, alpha
    r_boxes = rotate_boxes(_boxes, angle=angle, min_angle=1)
    if boxes.dtype == int:
        # Back to absolute boxes
        r_boxes[:, [0, 2]] *= img.shape[2]
        r_boxes[:, [1, 3]] *= img.shape[1]
    return rotated_img, r_boxes
