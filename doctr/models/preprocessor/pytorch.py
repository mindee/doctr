# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torch import nn
import numpy as np
from typing import List, Tuple, Union, Any

from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F
from doctr.transforms import Resize

__all__ = ['PreProcessor']


class PreProcessor(nn.Module):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        interpolation: one of 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5'

    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        batch_size: int,
        mean: Tuple[float, float, float] = (.5, .5, .5),
        std: Tuple[float, float, float] = (1., 1., 1.),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.resize: T.Resize = Resize(output_size, **kwargs)
        # Perform the division by 255 at the same time
        self.normalize = T.Normalize(mean, std)

    def batch_inputs(
        self,
        x: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            x: list of samples

        Returns:
            list of batched samples
        """

        num_batches = len(x) / self.batch_size
        # Deal with fixed-size batches
        b_images = [torch.stack(x[idx * self.batch_size: (idx + 1) * self.batch_size], dim=0)
                    for idx in range(int(num_batches))]
        # Deal with the last batch
        if num_batches > int(num_batches):
            b_images.append(torch.stack(x[int(num_batches) * self.batch_size:], dim=0))
        return b_images

    def __call__(
        self,
        x: Union[torch.Tensor, List[np.ndarray]]
    ) -> List[torch.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tensors (already resized and batched)
        Returns:
            list of page batches
        """
        # Check input type
        if isinstance(x, torch.Tensor):
            # Tf tensor from data loader: check if tensor size is output_size
            if x.shape[-2] != self.resize.size[0] or x.shape[-1] != self.resize.size[1]:
                x = F.resize(x, self.resize.size, interpolation=self.resize.interpolation)
            processed_batches = [x]
        elif isinstance(x, list):
            # Resize (and eventually pad) the inputs
            images: List[torch.Tensor] = [self.resize(torch.from_numpy(sample.copy()).permute(2, 0, 1)) for sample in x]
            # Batch them
            processed_batches = self.batch_inputs(images)  # type: ignore[assignment]
        else:
            raise AssertionError("invalid input type")
        # Normalize
        processed_batches = [
            self.normalize(b.to(dtype=torch.float32) / 255) for b in processed_batches  # type: ignore[union-attr]
        ]

        return processed_batches  # type: ignore[return-value]
