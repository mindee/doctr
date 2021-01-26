# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import math
import json
import numpy as np
import cv2
from typing import Union, List, Tuple, Optional, Any, Dict


__all__ = ['PreProcessor']


class PreProcessor:
    """Implements an abstract preprocessor object

    Example::
        >>> from doctr.documents import read_pdf
        >>> from doctr.models import Preprocessor
        >>> processor = Preprocessor(output_size=(600, 600), batch_size=8)
        >>> processed_doc = processor([read_pdf("path/to/your/doc.pdf")])

    Args:
        output_size: expected size of each page in format (H, W)
        normalize: whether tensor should be normalized
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        batch_size: int,
        mean: Tuple[float, float, float] = (.5, .5, .5),
        std: Tuple[float, float, float] = (1., 1., 1.),
    ) -> None:

        self.output_size = output_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.batch_size = batch_size

    def normalize(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """Takes a uint8 ndarray and moves it to [-1, 1] range

        Args:
            input_images: nested list of images encoded in uint8
        Returns:
            normalized tensors encoded in float32
        """

        # Re-center and scale the distribution to [-1, 1]
        return x.astype(np.float32) * (self.std / 255) - (self.mean / self.std)

    def resize(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """Resize each sample to a fixed size so that it could be batched

        Args:
            input_samples: list of unconstrained size ndarrays
        Returns:
            nested list of fixed-size ndarray
        """
        return cv2.resize(x, self.output_size, cv2.INTER_LINEAR)

    def batch_inputs(
        self,
        x: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Gather samples into batches for inference purposes

        Args:
            x: list of samples (numpy ndarray)

        Returns:
            list of batched samples
        """

        num_batches = len(x) / self.batch_size

        # Deal with fixed-size batches
        b_images = [np.stack(x[idx * self.batch_size: (idx + 1) * self.batch_size])
                    for idx in range(int(num_batches))]
        # Deal with the last batch
        if num_batches > int(num_batches):
            b_images.append(np.asarray(x[int(num_batches) * self.batch_size:]))

        return b_images

    def __call__(
        self,
        x: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (numpy ndarray)
        Returns:
            list of page batches
        """

        # Resize the inputs
        images = [self.resize(sample) for sample in x]
        # Batch them
        processed_batches = self.batch_inputs(images)
        # Normalize
        processed_batches = [self.normalize(b) for b in processed_batches]

        return processed_batches
