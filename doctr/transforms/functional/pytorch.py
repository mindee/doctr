# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Optional, Tuple

import numpy as np
import torch
from torchvision.transforms import functional as F

from doctr.utils.geometry import rotate_rel_geoms

from .base import create_shadow_mask, crop_boxes

__all__ = ["invert_colors", "rotate_sample", "crop_detection", "random_shadow"]


def invert_colors(img: torch.Tensor, min_val: float = 0.6) -> torch.Tensor:
    out = F.rgb_to_grayscale(img, num_output_channels=3)
    # Random RGB shift
    shift_shape = [img.shape[0], 3, 1, 1] if img.ndim == 4 else [3, 1, 1]
    rgb_shift = min_val + (1 - min_val) * torch.rand(shift_shape)
    # Inverse the color
    if out.dtype == torch.uint8:
        out = (out.to(dtype=rgb_shift.dtype) * rgb_shift).to(dtype=torch.uint8)
    else:
        out = out * rgb_shift.to(dtype=out.dtype)
    # Inverse the color
    out = 255 - out if out.dtype == torch.uint8 else 1 - out
    return out


def rotate_sample(
    img: torch.Tensor,
    angle: float,
    geoms: Optional[np.ndarray] = None,
    expand: bool = False,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Rotate image around the center, interpolation=NEAREST, pad with 0 (black)

    Args:
        img: image to rotate
        angle: angle in degrees. +: counter-clockwise, -: clockwise
        geoms: array of geometries of shape (N, 4) or (N, 4, 2)
        expand: whether the image should be padded before the rotation

    Returns:
        A tuple of rotated img (tensor), rotated geometries of shape (N, 4, 2)
    """
    rotated_img = F.rotate(img, angle=angle, fill=0, expand=expand)  # Interpolation NEAREST by default
    rotated_img = rotated_img[:3]  # when expand=True, it expands to RGBA channels
    rotated_geoms = geoms

    if isinstance(geoms, np.ndarray) and angle != 0:
        rotated_geoms = rotate_rel_geoms(geoms, angle, img.shape[-2:], rotated_img.shape[-2:], expand=expand)

    return rotated_img, rotated_geoms


def crop_detection(
    img: torch.Tensor,
    boxes: np.ndarray,
    crop_box: Tuple[float, float, float, float]
) -> Tuple[torch.Tensor, np.ndarray]:
    """Crop and image and associated bboxes

    Args:
        img: image to crop
        boxes: array of boxes to clip, absolute (int) or relative (float)
        crop_box: box (xmin, ymin, xmax, ymax) to crop the image. Relative coords.

    Returns:
        A tuple of cropped image, cropped boxes, where the image is not resized.
    """
    if any(val < 0 or val > 1 for val in crop_box):
        raise AssertionError("coordinates of arg `crop_box` should be relative")
    h, w = img.shape[-2:]
    xmin, ymin = int(round(crop_box[0] * (w - 1))), int(round(crop_box[1] * (h - 1)))
    xmax, ymax = int(round(crop_box[2] * (w - 1))), int(round(crop_box[3] * (h - 1)))
    cropped_img = F.crop(
        img, ymin, xmin, ymax - ymin, xmax - xmin
    )
    # Crop the box
    boxes = crop_boxes(boxes, crop_box if boxes.max() <= 1 else (xmin, ymin, xmax, ymax))

    return cropped_img, boxes


def random_shadow(img: torch.Tensor, opacity_range: Tuple[float, float], **kwargs) -> torch.Tensor:
    """Crop and image and associated bboxes

    Args:
        img: image to modify
        opacity_range: the minimum and maximum desired opacity of the shadow

    Returns:
        shaded image
    """

    shadow_mask = create_shadow_mask(img.shape[1:], **kwargs)  # type: ignore[arg-type]

    opacity = np.random.uniform(*opacity_range)
    shadow_tensor = 1 - torch.from_numpy(shadow_mask[None, ...])

    # Add some blur to make it believable
    k = 7 + 2 * int(4 * np.random.rand(1))
    sigma = np.random.uniform(.5, 5.)
    shadow_tensor = F.gaussian_blur(shadow_tensor, k, sigma=[sigma, sigma])

    return opacity * shadow_tensor * img + (1 - opacity) * img
