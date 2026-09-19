# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torchvision.transforms import functional as F

from doctr.utils.geometry import rotate_abs_geoms

from .base import create_shadow_mask, crop_boxes

__all__ = [
    "invert_colors",
    "rotate_sample",
    "crop_detection",
    "random_shadow",
    "random_glare",
    "random_lighting",
    "perspective_sample",
]


def invert_colors(img: torch.Tensor, min_val: float = 0.6) -> torch.Tensor:
    """Invert the colors of an image

    Args:
        img : torch.Tensor, the image to invert
        min_val : minimum value of the random shift

    Returns:
        the inverted image
    """
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
    geoms: np.ndarray,
    angle: float,
    expand: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    """Rotate image around the center, interpolation=NEAREST, pad with 0 (black)

    Args:
        img: image to rotate
        geoms: array of geometries of shape (N, 4) or (N, 4, 2)
        angle: angle in degrees. +: counter-clockwise, -: clockwise
        expand: whether the image should be padded before the rotation

    Returns:
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
        img.shape[1:],  # type: ignore[arg-type]
        expand,
    ).astype(np.float32)

    # Always return relative boxes to avoid label confusions when resizing is performed aferwards
    rotated_geoms[..., 0] = rotated_geoms[..., 0] / rotated_img.shape[2]
    rotated_geoms[..., 1] = rotated_geoms[..., 1] / rotated_img.shape[1]

    return rotated_img, np.clip(np.around(rotated_geoms, decimals=15), 0, 1)


def crop_detection(
    img: torch.Tensor, boxes: np.ndarray, crop_box: tuple[float, float, float, float]
) -> tuple[torch.Tensor, np.ndarray]:
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
    cropped_img = F.crop(img, ymin, xmin, ymax - ymin, xmax - xmin)
    # Crop the box
    boxes = crop_boxes(boxes, crop_box if boxes.max() <= 1 else (xmin, ymin, xmax, ymax))

    return cropped_img, boxes


def random_shadow(img: torch.Tensor, opacity_range: tuple[float, float], **kwargs) -> torch.Tensor:
    """Apply a random shadow effect to an image using NumPy for blurring.

    Args:
        img: Image to modify (C, H, W) as a PyTorch tensor.
        opacity_range: The minimum and maximum desired opacity of the shadow.
        **kwargs: Additional arguments to pass to `create_shadow_mask`.

    Returns:
        Shadowed image as a PyTorch tensor (same shape as input).
    """
    shadow_mask = create_shadow_mask(img.shape[1:], **kwargs)  # type: ignore[arg-type]
    opacity = np.random.uniform(*opacity_range)

    # Apply Gaussian blur to the shadow mask
    sigma = np.random.uniform(0.5, 5.0)
    blurred_mask = gaussian_filter(shadow_mask, sigma=sigma)

    shadow_tensor = 1 - torch.from_numpy(blurred_mask).float()
    shadow_tensor = shadow_tensor.to(img.device).unsqueeze(0)  # Add channel dimension

    return opacity * shadow_tensor * img + (1 - opacity) * img


def random_glare(
    img: torch.Tensor, opacity_range: tuple[float, float], num_spots: tuple[int, int] = (1, 3)
) -> torch.Tensor:
    """Add soft bright spots, like the specular reflections of a light source on a photographed screen or a glossy
    sheet.

    Args:
        img: image to modify (C, H, W), float in [0, 1]
        opacity_range: minimum and maximum strength of the reflections
        num_spots: minimum and maximum number of reflections

    Returns:
        the image with the reflections blended towards white (same shape as input)
    """
    _, h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(np.random.randint(num_spots[0], num_spots[1] + 1)):
        cx, cy = np.random.uniform(0, w), np.random.uniform(0, h)
        rx, ry = np.random.uniform(0.08, 0.35) * w, np.random.uniform(0.08, 0.35) * h
        theta = np.random.uniform(0, np.pi)
        dx, dy = xs - cx, ys - cy
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        mask = np.maximum(mask, np.exp(-((u / rx) ** 2 + (v / ry) ** 2)))
    opacity = float(np.random.uniform(*opacity_range))
    glare = torch.from_numpy(mask).to(device=img.device, dtype=img.dtype).unsqueeze(0) * opacity
    return img + glare * (1 - img)


def random_lighting(
    img: torch.Tensor, strength_range: tuple[float, float], grid: tuple[int, int] = (2, 4)
) -> torch.Tensor:
    """Multiply the image by a smooth, low-frequency brightness field: uneven lighting, the shading of a crumpled
    or curved sheet, vignetting.

    Args:
        img: image to modify (C, H, W), float in [0, 1]
        strength_range: minimum and maximum amplitude of the field (0.3 = brightness between 0.7 and 1.3)
        grid: minimum and maximum size of the random grid that is upsampled into the field

    Returns:
        the modulated image (same shape as input)
    """
    _, h, w = img.shape
    n = np.random.randint(grid[0], grid[1] + 1)
    strength = np.random.uniform(*strength_range)
    coarse = np.random.uniform(-1, 1, size=(n, n)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    field = (1 + strength * np.clip(field, -1, 1)).astype(np.float32)
    return (img * torch.from_numpy(field).to(device=img.device, dtype=img.dtype).unsqueeze(0)).clamp(0, 1)


def perspective_sample(
    img: torch.Tensor,
    geoms: np.ndarray,
    distortion: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor | None]:
    """Warp an image (and its boxes / polygons) with a random perspective, as if it were photographed at an angle.

    The four corners of the image are moved inwards by up to `distortion` times the image size, and the image is
    warped so that the original corners land on the moved ones (the content shrinks, the borders are filled with
    zeros like a background around a photographed page).

    Args:
        img: image to warp (C, H, W)
        geoms: relative boxes (N, 4) or polygons (N, 4, 2) in [0, 1]
        distortion: maximum relative displacement of every corner
        mask: optional (H, W) boolean validity mask warped alongside

    Returns:
        the warped image, the warped geometries (same format, clipped to [0, 1]) and the warped mask
    """
    _, h, w = img.shape
    start = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dx, dy = distortion * w, distortion * h
    end = start + np.array(
        [
            [np.random.uniform(0, dx), np.random.uniform(0, dy)],
            [-np.random.uniform(0, dx), np.random.uniform(0, dy)],
            [-np.random.uniform(0, dx), -np.random.uniform(0, dy)],
            [np.random.uniform(0, dx), -np.random.uniform(0, dy)],
        ],
        dtype=np.float32,
    )
    warped = F.perspective(img, start.tolist(), end.tolist(), interpolation=F.InterpolationMode.BILINEAR, fill=0)
    warped_mask = None
    if mask is not None:
        warped_mask = (
            F
            .perspective(
                mask.unsqueeze(0).to(torch.uint8), start.tolist(), end.tolist(), F.InterpolationMode.NEAREST, fill=0
            )
            .squeeze(0)
            .to(torch.bool)
        )
    if geoms.shape[0] == 0:
        return warped, geoms.copy(), warped_mask

    matrix = cv2.getPerspectiveTransform(start, end)
    is_polygon = geoms.ndim == 3
    if is_polygon:
        pts = geoms.astype(np.float32).reshape(-1, 4, 2)
    else:
        x1, y1, x2, y2 = (geoms[:, i].astype(np.float32) for i in range(4))
        pts = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1), np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)
    abs_pts = pts * np.array([w, h], dtype=np.float32)
    out = cv2.perspectiveTransform(abs_pts.reshape(-1, 1, 2), matrix).reshape(-1, 4, 2)
    rel = out / np.array([w, h], dtype=np.float32)
    rel = np.clip(rel, 0, 1)
    if is_polygon:
        return warped, rel.astype(geoms.dtype), warped_mask
    boxes = np.concatenate([rel.min(axis=1), rel.max(axis=1)], axis=1).astype(geoms.dtype)
    return warped, boxes, warped_mask
