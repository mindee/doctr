# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union

AbstractPath = Union[str, Path]


__all__ = ["to_gray"]


def read_image_as_numpy(img_path: AbstractPath) -> np.array:
    return 0


def read_image_as_tf(img_path: AbstractPath) -> tf.Tensor:
    img = tf.io.read_file(img_path)
    return tf.image.decode_jpeg(img, channels=3)


def to_tensor(np_tensor: np.ndarray, **kwargs) -> tf.Tensor:
    return tf.convert_to_tensor(np_tensor, **kwargs)


def to_gray(img: tf.Tensor) -> tf.Tensor:
    return tf.repeat(tf.image.rgb_to_grayscale(img), repeats=3, axis=-1)
