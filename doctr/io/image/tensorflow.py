# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf

from doctr.utils.common_types import AbstractPath

__all__ = ['read_img_as_tensor', 'decode_img_as_tensor']


def read_img_as_tensor(img_path: AbstractPath, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """Read an image file as a TensorFlow tensor

    Args:
        img_path: location of the image file
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
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
        img_content: bytes of a decoded image
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
        decoded image as a tensor
    """

    if dtype not in (tf.uint8, tf.float16, tf.float32):
        raise ValueError("insupported value for dtype")

    img = tf.io.decode_image(img_content, channels=3)

    if dtype != tf.uint8:
        img = tf.image.convert_image_dtype(img, dtype=dtype)
        img = tf.clip_by_value(img, 0, 1)

    return img
