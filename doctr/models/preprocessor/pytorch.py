# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from doctr.transforms import Resize
from doctr.utils.multithreading import multithread_exec

__all__ = ["PreProcessor"]


class PreProcessor(nn.Module):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        **kwargs: additional arguments for the resizing operation
    """

    def __init__(
        self,
        output_size: tuple[int, int],
        batch_size: int,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.resize: T.Resize = Resize(output_size, **kwargs)
        # Perform the division by 255 at the same time
        self.normalize = T.Normalize(mean, std)

    def batch_inputs(self, samples: list[torch.Tensor]) -> list[torch.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples of shape (C, H, W)

        Returns:
            list of batched samples (*, C, H, W)
        """
        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [
            torch.stack(samples[idx * self.batch_size : min((idx + 1) * self.batch_size, len(samples))], dim=0)
            for idx in range(int(num_batches))
        ]

        return batches

    def sample_transforms(self, x: np.ndarray) -> torch.Tensor:
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if x.dtype not in (np.uint8, np.float32, np.float16):
            raise TypeError("unsupported data type for numpy.ndarray")
        tensor = torch.from_numpy(x.copy()).permute(2, 0, 1)
        # Resizing
        tensor = self.resize(tensor)
        # Data type
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(dtype=torch.float32).div(255).clip(0, 1)
        else:
            tensor = tensor.to(dtype=torch.float32)

        return tensor

    def __call__(self, x: np.ndarray | list[np.ndarray]) -> list[torch.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or a single image (np.array) of shape (H, W, C)

        Returns:
            list of page batches (*, C, H, W) ready for model inference
        """
        # Input type check
        if isinstance(x, np.ndarray):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if x.dtype not in (np.uint8, np.float32, np.float16):
                raise TypeError("unsupported data type for numpy.ndarray")
            tensor = torch.from_numpy(x.copy()).permute(0, 3, 1, 2)

            # Resizing
            if tensor.shape[-2] != self.resize.size[0] or tensor.shape[-1] != self.resize.size[1]:
                tensor = F.resize(
                    tensor, self.resize.size, interpolation=self.resize.interpolation, antialias=self.resize.antialias
                )
            # Data type
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(dtype=torch.float32).div(255).clip(0, 1)
            else:
                tensor = tensor.to(dtype=torch.float32)
            batches = [tensor]

        elif isinstance(x, list) and all(isinstance(sample, np.ndarray) for sample in x):
            # Sample transform (to tensor, resize)
            samples = list(multithread_exec(self.sample_transforms, x))
            # Batching
            batches = self.batch_inputs(samples)
        else:
            raise TypeError(f"invalid input type: {type(x)}")

        # Batch transforms (normalize)
        batches = list(multithread_exec(self.normalize, batches))

        return batches
