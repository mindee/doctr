# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf


__all__ = ["to_gray"]


def to_gray(img: tf.Tensor) -> tf.Tensor:
    return tf.repeat(tf.image.rgb_to_grayscale(img), repeats=3, axis=-1)
