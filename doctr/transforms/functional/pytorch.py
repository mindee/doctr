# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import functional as F

from doctr.utils.geometry import rotate_abs_geoms

from .base import create_shadow_mask, crop_boxes

__all__ = ["invert_colors", "rotate_sample", "crop_detection", "random_shadow"]


def invert_colors(img: torch.Tensor, min_val: float = 0.6) -> torch.Tensor:
    """Invert the colors of an image

    Args:
    ----
        img : torch.Tensor, the image to invert
        min_val : minimum value of the random shift

    Returns:
    -------
        the inverted image
    """
    out = F.rgb_to_grayscale(img, num_output_channels=3)
    # Random RGB shift
    shift_shape = [img.shape[0], 3, 1, 1] if img.ndim == 4 else [3, 1, 1]
    rgb_shift = min_val + (1 - min_val) * torch.rand(shift_shape)
    # Inverse the color
    if out.dtype == torch.uint8:
        out = (out.to(dtype=rgb_shift.dtype) * rgb_shift).to(dtype=torch.uint8)  # type: ignore[attr-defined]
    else:
        out = out * rgb_shift.to(dtype=out.dtype)  # type: ignore[attr-defined]
    # Inverse the color
    out = 255 - out if out.dtype == torch.uint8 else 1 - out
    return out


def rotate_sample(
    img: torch.Tensor,
    geoms: np.ndarray,
    angle: float,
    expand: bool = False,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Rotate image around the center, interpolation=NEAREST, pad with 0 (black)

    Args:
    ----
        img: image to rotate
        geoms: array of geometries of shape (N, 4) or (N, 4, 2)
        angle: angle in degrees. +: counter-clockwise, -: clockwise
        expand: whether the image should be padded before the rotation

    Returns:
    -------
        A tuple of rotated img (tensor), rotated geometries of shape (N, 4, 2)
    """
    rotated_img = F.rotate(img, angle=angle, fill=0, expand=expand)  # Interpolation NEAREST by default
    rotated_img = rotated_img[:3]  # when expand=True, it expands to RGBA channels
    # Get absolute coords
    _geoms = deepcopy(geoms)
    if _geoms.shape[1:] == (4,):
        if np.max(_geoms) <= 1:
            _geoms[:, [0, 2]] *= img.shape[-1]
            _geoms[:, [1, 3]] *= img.shape[-2]
    elif _geoms.shape[1:] == (4, 2):
        if np.max(_geoms) <= 1:
            _geoms[..., 0] *= img.shape[-1]
            _geoms[..., 1] *= img.shape[-2]
    else:
        raise AssertionError("invalid format for arg `geoms`")

    # Rotate the boxes: xmin, ymin, xmax, ymax or polygons --> (4, 2) polygon
    rotated_geoms: np.ndarray = rotate_abs_geoms(
        _geoms,
        angle,
        img.shape[1:],
        expand,
    ).astype(np.float32)

    # Always return relative boxes to avoid label confusions when resizing is performed aferwards
    rotated_geoms[..., 0] = rotated_geoms[..., 0] / rotated_img.shape[2]
    rotated_geoms[..., 1] = rotated_geoms[..., 1] / rotated_img.shape[1]

    return rotated_img, np.clip(rotated_geoms, 0, 1)


def crop_detection(
    img: torch.Tensor, boxes: np.ndarray, crop_box: Tuple[float, float, float, float]
) -> Tuple[torch.Tensor, np.ndarray]:
    """Crop and image and associated bboxes

    Args:
    ----
        img: image to crop
        boxes: array of boxes to clip, absolute (int) or relative (float)
        crop_box: box (xmin, ymin, xmax, ymax) to crop the image. Relative coords.

    Returns:
    -------
        A tuple of cropped image, cropped boxes, where the image is not resized.
    """
    if any(val < 0 or val > 1 for val in crop_box):
        raise AssertionError("coordinates of arg `crop_box` should be relative")
    h, w = img.shape[-2:]
    xmin, ymin = int(round(crop_box[0] * (w - 1))), int(round(crop_box[1] * (h - 1)))
    xmax, ymax = int(round(crop_box[2] * (w - 1))), int(round(crop_box[3] * (h - 1)))
    cropped_img = F.crop(img, ymin, xmin, ymax - ymin, xmax - xmin)
    # Crop the box
    boxes = crop_boxes(boxes, crop_box if boxes.max() <= 1 else (xmin, ymin, xmax, ymax))

    return cropped_img, boxes


def random_shadow(img: torch.Tensor, opacity_range: Tuple[float, float], **kwargs) -> torch.Tensor:
    """Crop and image and associated bboxes

    Args:
    ----
        img: image to modify
        opacity_range: the minimum and maximum desired opacity of the shadow
        **kwargs: additional arguments to pass to `create_shadow_mask`

    Returns:
    -------
        shaded image
    """
    shadow_mask = create_shadow_mask(img.shape[1:], **kwargs)

    opacity = np.random.uniform(*opacity_range)
    shadow_tensor = 1 - torch.from_numpy(shadow_mask[None, ...])

    # Add some blur to make it believable
    k = 7 + 2 * int(4 * np.random.rand(1))
    sigma = np.random.uniform(0.5, 5.0)
    shadow_tensor = F.gaussian_blur(shadow_tensor, k, sigma=[sigma, sigma])

    return opacity * shadow_tensor * img + (1 - opacity) * img
