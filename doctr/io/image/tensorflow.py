# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf

from ..reader import AbstractPath

__all__ = ['read_img_as_tensor']


def read_img_as_tensor(img_path: AbstractPath, out_dtype: tf.dtypes.DType) -> tf.Tensor:
    """Read an image file as a TensorFlow tensor

    Args:
        img_path: location of the image file
        out_dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
        decoded image as a tensor
    """

    if out_dtype not in (tf.uint8, tf.float16, tf.float32):
        raise ValueError("insupported value for out_dtype")

    img = tf.io.read_file(os.path.join(self.root, img_name))
    img = tf.image.decode_jpeg(img, channels=3)

    if out_dtype != tf.uint8:
        img = tf.image.convert_image_dtype(img, dtype=out_dtype)
        img = tf.clip_by_value(img, 0, 1)

    return img
