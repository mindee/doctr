# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from typing import Optional


__all__ = ["invert_colors"]


def invert_colors(img: tf.Tensor, min_val: Optional[float] = 0.6) -> tf.Tensor:
    out = tf.image.rgb_to_grayscale(img)  # Convert to gray
    # Random RGB shift
    shift_shape = [img.shape[0], 1, 1, 3] if img.ndim == 4 else [1, 1, 3]
    rgb_shift = tf.random.uniform(shape=shift_shape, minval=min_val, maxval=1)
    # Inverse the color
    if out.dtype == tf.uint8:
        out = tf.cast(tf.cast(out, dtype=tf.float32) * rgb_shift, dtype=tf.uint8)
    else:
        out *= rgb_shift
    # Inverse the color
    out = 255 - out if out.dtype == tf.uint8 else 1 - out
    return out
