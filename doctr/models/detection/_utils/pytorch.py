# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torch import Tensor
from torch.nn.functional import max_pool2d

__all__ = ['erode', 'dilate', 'generate_bin_map']


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


def generate_bin_map(prob_map: Tensor, bin_thresh: float, kernel_size: int = 3) -> Tensor:
    """Binarizes a tensor using a morphological opening (erosion, then dilation)

    Args:
        prob_map: probability tensor of shape (N, C, H, W)
        bin_thresh: the minimum confidence to consider a prediction as positive
        kernel_size: the size of the kernel to use for dilation

    Returns:
        binary tensor of shape (N, C, H, W)
    """

    return dilate(erode((prob_map >= bin_thresh).to(dtype=torch.float32), 3), 3)
