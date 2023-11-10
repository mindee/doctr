# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
import random
from copy import deepcopy
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from doctr.utils.geometry import compute_expanded_shape, rotate_abs_geoms

from .base import create_shadow_mask, crop_boxes

__all__ = ["invert_colors", "rotate_sample", "crop_detection", "random_shadow"]


def invert_colors(img: tf.Tensor, min_val: float = 0.6) -> tf.Tensor:
    """Invert the colors of an image

    Args:
    ----
        img : tf.Tensor, the image to invert
        min_val : minimum value of the random shift

    Returns:
    -------
        the inverted image
    """
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
    ----
        img: image to rotate
        angle: angle in degrees. +: counter-clockwise, -: clockwise
        expand: whether the image should be padded before the rotation

    Returns:
    -------
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

    # Compute the rotation matrix
    height, width = tf.cast(tf.shape(exp_img)[0], tf.float32), tf.cast(tf.shape(exp_img)[1], tf.float32)
    cos_angle, sin_angle = tf.math.cos(angle * math.pi / 180.0), tf.math.sin(angle * math.pi / 180.0)
    x_offset = ((width - 1) - (cos_angle * (width - 1) - sin_angle * (height - 1))) / 2.0
    y_offset = ((height - 1) - (sin_angle * (width - 1) + cos_angle * (height - 1))) / 2.0

    rotation_matrix = tf.convert_to_tensor(
        [cos_angle, -sin_angle, x_offset, sin_angle, cos_angle, y_offset, 0.0, 0.0],
        dtype=tf.float32,
    )
    # Rotate the image
    rotated_img = tf.squeeze(
        tf.raw_ops.ImageProjectiveTransformV3(
            images=exp_img[None],  # Add a batch dimension for compatibility with ImageProjectiveTransformV3
            transforms=rotation_matrix[None],  # Add a batch dimension for compatibility with ImageProjectiveTransformV3
            output_shape=tf.shape(exp_img)[:2],
            interpolation="NEAREST",
            fill_mode="CONSTANT",
            fill_value=tf.constant(0.0, dtype=tf.float32),
        )
    )
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
    ----
        img: image to rotate
        geoms: array of geometries of shape (N, 4) or (N, 4, 2)
        angle: angle in degrees. +: counter-clockwise, -: clockwise
        expand: whether the image should be padded before the rotation

    Returns:
    -------
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
    h, w = img.shape[:2]
    xmin, ymin = int(round(crop_box[0] * (w - 1))), int(round(crop_box[1] * (h - 1)))
    xmax, ymax = int(round(crop_box[2] * (w - 1))), int(round(crop_box[3] * (h - 1)))
    cropped_img = tf.image.crop_to_bounding_box(img, ymin, xmin, ymax - ymin, xmax - xmin)
    # Crop the box
    boxes = crop_boxes(boxes, crop_box if boxes.max() <= 1 else (xmin, ymin, xmax, ymax))

    return cropped_img, boxes


def _gaussian_filter(
    img: tf.Tensor,
    kernel_size: Union[int, Iterable[int]],
    sigma: float,
    mode: Optional[str] = None,
    pad_value: Optional[int] = 0,
):
    """Apply Gaussian filter to image.
    Adapted from: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/image/filters.py

    Args:
    ----
        img: image to filter of shape (N, H, W, C)
        kernel_size: kernel size of the filter
        sigma: standard deviation of the Gaussian filter
        mode: padding mode, one of "CONSTANT", "REFLECT", "SYMMETRIC"
        pad_value: value to pad the image with

    Returns:
    -------
        A tensor of shape (N, H, W, C)
    """
    ksize = tf.convert_to_tensor(tf.broadcast_to(kernel_size, [2]), dtype=tf.int32)
    sigma = tf.convert_to_tensor(tf.broadcast_to(sigma, [2]), dtype=img.dtype)
    assert mode in ("CONSTANT", "REFLECT", "SYMMETRIC"), "mode should be one of 'CONSTANT', 'REFLECT', 'SYMMETRIC'"
    mode = "CONSTANT" if mode is None else str.upper(mode)
    constant_values = (
        tf.zeros([], dtype=img.dtype) if pad_value is None else tf.convert_to_tensor(pad_value, dtype=img.dtype)
    )

    def kernel1d(ksize: tf.Tensor, sigma: tf.Tensor, dtype: tf.DType):
        x = tf.range(ksize, dtype=dtype)
        x = x - tf.cast(tf.math.floordiv(ksize, 2), dtype=dtype)
        x = x + tf.where(tf.math.equal(tf.math.mod(ksize, 2), 0), tf.cast(0.5, dtype), 0)
        g = tf.math.exp(-(tf.math.pow(x, 2) / (2 * tf.math.pow(sigma, 2))))
        g = g / tf.reduce_sum(g)
        return g

    def kernel2d(ksize: tf.Tensor, sigma: tf.Tensor, dtype: tf.DType):
        kernel_x = kernel1d(ksize[0], sigma[0], dtype)
        kernel_y = kernel1d(ksize[1], sigma[1], dtype)
        return tf.matmul(
            tf.expand_dims(kernel_x, axis=-1),
            tf.transpose(tf.expand_dims(kernel_y, axis=-1)),
        )

    g = kernel2d(ksize, sigma, img.dtype)
    # Pad the image
    height, width = ksize[0], ksize[1]
    paddings = [
        [0, 0],
        [(height - 1) // 2, height - 1 - (height - 1) // 2],
        [(width - 1) // 2, width - 1 - (width - 1) // 2],
        [0, 0],
    ]
    img = tf.pad(img, paddings, mode=mode, constant_values=constant_values)

    channel = tf.shape(img)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    g = tf.reshape(g, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    g = tf.broadcast_to(g, shape)
    return tf.nn.depthwise_conv2d(img, g, [1, 1, 1, 1], padding="VALID", data_format="NHWC")


def random_shadow(img: tf.Tensor, opacity_range: Tuple[float, float], **kwargs) -> tf.Tensor:
    """Apply a random shadow to a given image

    Args:
    ----
        img: image to modify
        opacity_range: the minimum and maximum desired opacity of the shadow
        **kwargs: additional arguments to pass to `create_shadow_mask`

    Returns:
    -------
        shadowed image
    """
    shadow_mask = create_shadow_mask(img.shape[:2], **kwargs)

    opacity = np.random.uniform(*opacity_range)
    shadow_tensor = 1 - tf.convert_to_tensor(shadow_mask[..., None], dtype=tf.float32)

    # Add some blur to make it believable
    k = 7 + int(2 * 4 * random.random())
    sigma = random.uniform(0.5, 5.0)
    shadow_tensor = _gaussian_filter(
        shadow_tensor[tf.newaxis, ...],
        kernel_size=k,
        sigma=sigma,
        mode="REFLECT",
    )

    return tf.squeeze(opacity * shadow_tensor * img + (1 - opacity) * img, axis=0)
