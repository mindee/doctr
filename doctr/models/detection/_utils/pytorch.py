# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from torch import Tensor
from torch.nn.functional import max_pool2d

__all__ = ["erode", "dilate"]


def erode(x: Tensor, kernel_size: int) -> Tensor:
    """Performs erosion on a given tensor

    Args:
        x: boolean tensor of shape (N, C, H, W)
        kernel_size: the size of the kernel to use for erosion

    Returns:
        the eroded tensor
    """
    _pad = (kernel_size - 1) // 2

    return 1 - max_pool2d(1 - x, kernel_size, stride=1, padding=_pad)


def dilate(x: Tensor, kernel_size: int) -> Tensor:
    """Performs dilation on a given tensor

    Args:
        x: boolean tensor of shape (N, C, H, W)
        kernel_size: the size of the kernel to use for dilation

    Returns:
        the dilated tensor
    """
    _pad = (kernel_size - 1) // 2

    return max_pool2d(x, kernel_size, stride=1, padding=_pad)
