# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torch import nn
import numpy as np
from typing import List, Tuple, Union, Any

from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F
from torch.nn.functional import pad

__all__ = ['PreProcessor']


class Resize(T.Resize):
    def __init__(
        self,
        size: Tuple[int, int],
        interpolation=InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[0] / img.shape[1]
        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio):
            return super().forward(img)
        else:
            if actual_ratio > target_ratio:
                tmp_size = (self.size[0], int(self.size[0] / actual_ratio))
            else:
                tmp_size = (int(self.size[1] * actual_ratio), self.size[1])
            # Scale image
            img = F.resize(img, tmp_size, self.interpolation)
            # Pad
            _pad = (self.size[0] - img.shape[0], self.size[1] - img.shape[1])
            if self.symmetric_pad:
                half_pad = (math.ceil(_pad[0] / 2), math.ceil(_pad[1] / 2))
                _pad = (half_pad[0], _pad[0] - half_pad[0], half_pad[1], _pad[1] - half_pad[1])
            else:
                _pad = (0, _pad[0], 0, _pad[1])
            return pad(img, _pad)


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

        self.batch_size = batch_size
        self.resize = Resize(output_size, **kwargs)
        # Perform the division by 255 at the same time
        self.normalize = T.Compose([
            lambda x: x / 255,
            T.Normalize(mean, std),
        ])

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
            b_images.append(torch.stack(x[int(num_batches) * self.batch_size:], axis=0))
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
            if x.shape[1] != self.resize.output_size[0] or x.shape[2] != self.resize.output_size[1]:
                x = F.resize(x, self.resize.output_size, method=self.resize.method)
            processed_batches = [x]
        elif isinstance(x, list):
            # convert images to tf
            tensors = [sample.to(dtype=torch.float32) for sample in x]
            # Resize (and eventually pad) the inputs
            images = [self.resize(sample) for sample in tensors]
            # Batch them
            processed_batches = self.batch_inputs(images)
        else:
            raise AssertionError("invalid input type")
        # Normalize
        processed_batches = [self.normalize(b) for b in processed_batches]

        return processed_batches
