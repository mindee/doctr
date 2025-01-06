# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from doctr.utils.common_types import AbstractPath

__all__ = ["tensor_from_pil", "read_img_as_tensor", "decode_img_as_tensor", "tensor_from_numpy", "get_img_shape"]


def tensor_from_pil(pil_img: Image.Image, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch tensor

    Args:
        pil_img: a PIL image
        dtype: the output tensor data type

    Returns:
        decoded image as tensor
    """
    if dtype == torch.float32:
        img = to_tensor(pil_img)
    else:
        img = tensor_from_numpy(np.array(pil_img, np.uint8, copy=True), dtype)

    return img


def read_img_as_tensor(img_path: AbstractPath, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Read an image file as a PyTorch tensor

    Args:
        img_path: location of the image file
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
        decoded image as a tensor
    """
    if dtype not in (torch.uint8, torch.float16, torch.float32):
        raise ValueError("insupported value for dtype")

    with Image.open(img_path, mode="r") as pil_img:
        return tensor_from_pil(pil_img.convert("RGB"), dtype)


def decode_img_as_tensor(img_content: bytes, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Read a byte stream as a PyTorch tensor

    Args:
        img_content: bytes of a decoded image
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
        decoded image as a tensor
    """
    if dtype not in (torch.uint8, torch.float16, torch.float32):
        raise ValueError("insupported value for dtype")

    with Image.open(BytesIO(img_content), mode="r") as pil_img:
        return tensor_from_pil(pil_img.convert("RGB"), dtype)


def tensor_from_numpy(npy_img: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Read an image file as a PyTorch tensor

    Args:
        npy_img: image encoded as a numpy array of shape (H, W, C) in np.uint8
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
        same image as a tensor of shape (C, H, W)
    """
    if dtype not in (torch.uint8, torch.float16, torch.float32):
        raise ValueError("insupported value for dtype")

    if dtype == torch.float32:
        img = to_tensor(npy_img)
    else:
        img = torch.from_numpy(npy_img)
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        if dtype == torch.float16:
            # Switch to FP16
            img = img.to(dtype=torch.float16).div(255)

    return img


def get_img_shape(img: torch.Tensor) -> tuple[int, int]:
    """Get the shape of an image"""
    return img.shape[-2:]
