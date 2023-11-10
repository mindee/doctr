# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import img_to_array

from doctr.utils.common_types import AbstractPath

__all__ = ["tensor_from_pil", "read_img_as_tensor", "decode_img_as_tensor", "tensor_from_numpy", "get_img_shape"]


def tensor_from_pil(pil_img: Image, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """Convert a PIL Image to a TensorFlow tensor

    Args:
    ----
        pil_img: a PIL image
        dtype: the output tensor data type

    Returns:
    -------
        decoded image as tensor
    """
    npy_img = img_to_array(pil_img)

    return tensor_from_numpy(npy_img, dtype)


def read_img_as_tensor(img_path: AbstractPath, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """Read an image file as a TensorFlow tensor

    Args:
    ----
        img_path: location of the image file
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
    -------
        decoded image as a tensor
    """
    if dtype not in (tf.uint8, tf.float16, tf.float32):
        raise ValueError("insupported value for dtype")

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)

    if dtype != tf.uint8:
        img = tf.image.convert_image_dtype(img, dtype=dtype)
        img = tf.clip_by_value(img, 0, 1)

    return img


def decode_img_as_tensor(img_content: bytes, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """Read a byte stream as a TensorFlow tensor

    Args:
    ----
        img_content: bytes of a decoded image
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
    -------
        decoded image as a tensor
    """
    if dtype not in (tf.uint8, tf.float16, tf.float32):
        raise ValueError("insupported value for dtype")

    img = tf.io.decode_image(img_content, channels=3)

    if dtype != tf.uint8:
        img = tf.image.convert_image_dtype(img, dtype=dtype)
        img = tf.clip_by_value(img, 0, 1)

    return img


def tensor_from_numpy(npy_img: np.ndarray, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """Read an image file as a TensorFlow tensor

    Args:
    ----
        npy_img: image encoded as a numpy array of shape (H, W, C) in np.uint8
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
    -------
        same image as a tensor of shape (H, W, C)
    """
    if dtype not in (tf.uint8, tf.float16, tf.float32):
        raise ValueError("insupported value for dtype")

    if dtype == tf.uint8:
        img = tf.convert_to_tensor(npy_img, dtype=dtype)
    else:
        img = tf.image.convert_image_dtype(npy_img, dtype=dtype)
        img = tf.clip_by_value(img, 0, 1)

    return img


def get_img_shape(img: tf.Tensor) -> Tuple[int, int]:
    """Get the shape of an image"""
    return img.shape[:2]
