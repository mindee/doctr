# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf

__all__ = ['erode', 'dilate']


def erode(x: tf.Tensor, kernel_size: int) -> tf.Tensor:
    """Performs erosion on a given tensor

    Args:
        x: boolean tensor of shape (N, H, W, C)
        kernel_size: the size of the kernel to use for erosion
    Returns:
        the eroded tensor
    """

    return 1 - tf.nn.max_pool2d(1 - x, kernel_size, strides=1, padding="SAME")


def dilate(x: tf.Tensor, kernel_size: int) -> tf.Tensor:
    """Performs dilation on a given tensor

    Args:
        x: boolean tensor of shape (N, H, W, C)
        kernel_size: the size of the kernel to use for dilation
    Returns:
        the dilated tensor
    """

    return tf.nn.max_pool2d(x, kernel_size, strides=1, padding="SAME")
