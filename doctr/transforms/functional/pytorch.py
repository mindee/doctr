# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torchvision.transforms import functional as F


__all__ = ["invert_colors"]


def invert_colors(img: torch.Tensor, min_val: float = 0.6) -> torch.Tensor:
    out = F.rgb_to_grayscale(img, num_output_channels=3)
    # Random RGB shift
    shift_shape = [img.shape[0], 3, 1, 1] if img.ndim == 4 else [3, 1, 1]
    rgb_shift = min_val + (1 - min_val) * torch.rand(shift_shape)
    # Inverse the color
    if out.dtype == torch.uint8:
        out = (out.to(dtype=torch.float32) * rgb_shift).to(dtype=torch.uint8)
    else:
        out = out * rgb_shift
    # Inverse the color
    out = 255 - out if out.dtype == torch.uint8 else 1 - out
    return out
