# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import torch
from torch import nn
import numpy as np
from typing import List, Tuple, Union, Any
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F

from doctr.transforms import Resize
from doctr.utils.multithreading import multithread_exec

__all__ = ['PreProcessor']


class PreProcessor(nn.Module):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        fp16: whether returned batches should be in FP16
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        batch_size: int,
        mean: Tuple[float, float, float] = (.5, .5, .5),
        std: Tuple[float, float, float] = (1., 1., 1.),
        fp16: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.resize: T.Resize = Resize(output_size, **kwargs)
        # Perform the division by 255 at the same time
        self.normalize = T.Normalize(mean, std)
        self.fp16 = fp16

    def batch_inputs(
        self,
        samples: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples of shape (C, H, W)

        Returns:
            list of batched samples (*, C, H, W)
        """

        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [
            torch.stack(samples[idx * self.batch_size: min((idx + 1) * self.batch_size, len(samples))], dim=0)
            for idx in range(int(num_batches))
        ]

        return batches

    def sample_transforms(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float16, np.float32):
                raise TypeError("unsupported data type for numpy.ndarray")
            x = torch.from_numpy(x.copy()).permute(2, 0, 1)
        elif x.dtype not in (torch.uint8, torch.float16, torch.float32):
            raise TypeError("unsupported data type for torch.Tensor")
        # Resizing
        x = self.resize(x)
        # Data type
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div(255).clip(0, 1)
        x = x.to(dtype=torch.float16 if self.fp16 else torch.float32)

        return x

    def __call__(
        self,
        x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]]
    ) -> List[torch.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tensors (already resized and batched)
        Returns:
            list of page batches
        """

        # Input type check
        if isinstance(x, (np.ndarray, torch.Tensor)):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if isinstance(x, np.ndarray):
                if x.dtype not in (np.uint8, np.float16, np.float32):
                    raise TypeError("unsupported data type for numpy.ndarray")
                x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2)
            elif x.dtype not in (torch.uint8, torch.float16, torch.float32):
                raise TypeError("unsupported data type for torch.Tensor")
            # Resizing
            if x.shape[-2] != self.resize.size[0] or x.shape[-1] != self.resize.size[1]:
                x = F.resize(x, self.resize.size, interpolation=self.resize.interpolation)
            # Data type
            if x.dtype == torch.uint8:
                x = x.to(dtype=torch.float32).div(255).clip(0, 1)
            x = x.to(dtype=torch.float16 if self.fp16 else torch.float32)
            batches = [x]

        elif isinstance(x, list) and all(isinstance(sample, (np.ndarray, torch.Tensor)) for sample in x):
            # Sample transform (to tensor, resize)
            samples = multithread_exec(self.sample_transforms, x)
            # Batching
            batches = self.batch_inputs(samples)  # type: ignore[arg-type]
        else:
            raise TypeError(f"invalid input type: {type(x)}")

        # Batch transforms (normalize)
        batches = multithread_exec(self.normalize, batches)  # type: ignore[assignment]

        return batches
