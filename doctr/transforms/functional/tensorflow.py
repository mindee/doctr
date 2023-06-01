# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from doctr.utils.geometry import compute_expanded_shape, rotate_abs_geoms

from .base import create_shadow_mask, crop_boxes

__all__ = ["invert_colors", "rotate_sample", "crop_detection", "random_shadow"]


def invert_colors(img: tf.Tensor, min_val: float = 0.6) -> tf.Tensor:
    out = tf.image.rgb_to_grayscale(img)  # Convert to gray
    # Random RGB shift
    shift_shape = [img.shape[0], 1, 1, 3] if img.ndim == 4 else [1, 1, 3]
    rgb_shift = tf.random.uniform(shape=shift_shape, minval=min_val, maxval=1)
    # Inverse the color
    if out.dtype == tf.uint8:
        out = tf.cast(tf.cast(out, dtype=rgb_shift.dtype) * rgb_shift, dtype=tf.uint8)
    else:
        out *= tf.cast(rgb_shift, dtype=out.dtype)
    # Inverse the color
    out = 255 - out if out.dtype == tf.uint8 else 1 - out
    return out


def rotated_img_tensor(img: tf.Tensor, angle: float, expand: bool = False) -> tf.Tensor:
    """Rotate image around the center, interpolation=NEAREST, pad with 0 (black)

    Args:
        img: image to rotate
        angle: angle in degrees. +: counter-clockwise, -: clockwise
        expand: whether the image should be padded before the rotation

    Returns:
        the rotated image (tensor)
    """
    # Compute the expanded padding
    h_crop, w_crop = 0, 0
    if expand:
        exp_h, exp_w = compute_expanded_shape(img.shape[:-1], angle)
        h_diff, w_diff = int(math.ceil(exp_h - img.shape[0])), int(math.ceil(exp_w - img.shape[1]))
        h_pad, w_pad = max(h_diff, 0), max(w_diff, 0)
        exp_img = tf.pad(img, tf.constant([[h_pad // 2, h_pad - h_pad // 2], [w_pad // 2, w_pad - w_pad // 2], [0, 0]]))
        h_crop, w_crop = int(round(max(exp_img.shape[0] - exp_h, 0))), int(round(min(exp_img.shape[1] - exp_w, 0)))
    else:
        exp_img = img
    # Rotate the padded image
    rotated_img = tfa.image.rotate(exp_img, angle * math.pi / 180)  # Interpolation NEAREST by default
    # Crop the rest
    if h_crop > 0 or w_crop > 0:
        h_slice = slice(h_crop // 2, -h_crop // 2) if h_crop > 0 else slice(rotated_img.shape[0])
        w_slice = slice(-w_crop // 2, -w_crop // 2) if w_crop > 0 else slice(rotated_img.shape[1])
        rotated_img = rotated_img[h_slice, w_slice]

    return rotated_img


def rotate_sample(
    img: tf.Tensor,
    geoms: np.ndarray,
    angle: float,
    expand: bool = False,
) -> Tuple[tf.Tensor, np.ndarray]:
    """Rotate image around the center, interpolation=NEAREST, pad with 0 (black)

    Args:
        img: image to rotate
        geoms: array of geometries of shape (N, 4) or (N, 4, 2)
        angle: angle in degrees. +: counter-clockwise, -: clockwise
        expand: whether the image should be padded before the rotation

    Returns:
        A tuple of rotated img (tensor), rotated boxes (np array)
    """
    # Rotated the image
    rotated_img = rotated_img_tensor(img, angle, expand)

    # Get absolute coords
    _geoms = deepcopy(geoms)
    if _geoms.shape[1:] == (4,):
        if np.max(_geoms) <= 1:
            _geoms[:, [0, 2]] *= img.shape[1]
            _geoms[:, [1, 3]] *= img.shape[0]
    elif _geoms.shape[1:] == (4, 2):
        if np.max(_geoms) <= 1:
            _geoms[..., 0] *= img.shape[1]
            _geoms[..., 1] *= img.shape[0]
    else:
        raise AssertionError

    # Rotate the boxes: xmin, ymin, xmax, ymax or polygons --> (4, 2) polygon
    rotated_geoms: np.ndarray = rotate_abs_geoms(_geoms, angle, img.shape[:-1], expand).astype(np.float32)

    # Always return relative boxes to avoid label confusions when resizing is performed aferwards
    rotated_geoms[..., 0] = rotated_geoms[..., 0] / rotated_img.shape[1]
    rotated_geoms[..., 1] = rotated_geoms[..., 1] / rotated_img.shape[0]

    return rotated_img, np.clip(rotated_geoms, 0, 1)


def crop_detection(
    img: tf.Tensor, boxes: np.ndarray, crop_box: Tuple[float, float, float, float]
) -> Tuple[tf.Tensor, np.ndarray]:
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
    h, w = img.shape[:2]
    xmin, ymin = int(round(crop_box[0] * (w - 1))), int(round(crop_box[1] * (h - 1)))
    xmax, ymax = int(round(crop_box[2] * (w - 1))), int(round(crop_box[3] * (h - 1)))
    cropped_img = tf.image.crop_to_bounding_box(img, ymin, xmin, ymax - ymin, xmax - xmin)
    # Crop the box
    boxes = crop_boxes(boxes, crop_box if boxes.max() <= 1 else (xmin, ymin, xmax, ymax))

    return cropped_img, boxes


def random_shadow(img: tf.Tensor, opacity_range: Tuple[float, float], **kwargs) -> tf.Tensor:
    """Apply a random shadow to a given image

    Args:
        img: image to modify
        opacity_range: the minimum and maximum desired opacity of the shadow

    Returns:
        shaded image
    """

    shadow_mask = create_shadow_mask(img.shape[:2], **kwargs)

    opacity = np.random.uniform(*opacity_range)
    shadow_tensor = 1 - tf.convert_to_tensor(shadow_mask[..., None], dtype=tf.float32)

    # Add some blur to make it believable
    k = 7 + int(2 * 4 * np.random.rand(1))
    shadow_tensor = tfa.image.gaussian_filter2d(
        shadow_tensor,
        filter_shape=k,
        sigma=np.random.uniform(0.5, 5.0),
    )

    return opacity * shadow_tensor * img + (1 - opacity) * img
