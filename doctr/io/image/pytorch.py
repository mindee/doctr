# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

from ..reader import AbstractPath

__all__ = ['read_img_as_tensor']


def read_img_as_tensor(img_path: AbstractPath, out_dtype: torch.dtype) -> torch.Tensor:
    """Read an image file as a PyTorch tensor

    Args:
        img_path: location of the image file
        out_dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
        decoded image as a tensor
    """

    if out_dtype not in (torch.uint8, torch.float16, torch.float32):
        raise ValueError("insupported value for out_dtype")

    pil_img = Image.open(img_path, mode='r').convert('RGB')

    if out_dtype == torch.float32:
        img = to_tensor(pil_img)
    else:
        img = torch.from_numpy(
            np.array(pil_img, np.uint8, copy=True)
        )
        img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        if out_dtype == torch.float16:
            # Switch to FP16
            img = img.to(dtype=torch.float16).div(255)

    return img
