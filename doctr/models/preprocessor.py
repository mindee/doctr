# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import math
import json
import numpy as np
import cv2
from typing import Union, List, Tuple, Optional, Any, Dict


__all__ = ['Preprocessor']


class Preprocessor:
    """Implements an abstract preprocessor object

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

    def normalize_inputs(
        self,
        input_batches: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Takes a uint8 ndarray and moves it to [-1, 1] range

        Args:
            input_images: nested list of images encoded in uint8
        Returns:
            normalized tensors encoded in float32
        """

        # Re-center and scale the distribution to [-1, 1]
        return [batch.astype(np.float32) * (self.std / 255) - (self.mean / self.std)
                for batch in input_batches]

    def resize_inputs(
        self,
        input_samples: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """Resize each sample to a fixed size so that it could be batched

        Args:
            input_samples: nested list of unconstrained size ndarrays
        Returns:
            nested list of fixed-size ndarray
        """
        return [[cv2.resize(img, self.output_size, cv2.INTER_LINEAR) for img in doc]
                for doc in input_samples]

    def batch_inputs(
        self,
        documents: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Gather pages into batches for inference purposes

        Args:
            documents: list of documents, which is expressed as list of pages (numpy ndarray)

        Returns:
            list of batched samples
        """

        # flatten structure
        page_list = [image for doc in documents for image in doc]

        num_batches = len(page_list) / self.batch_size

        # Deal with fixed-size batches
        b_images = [np.stack(page_list[idx * self.batch_size: (idx + 1) * self.batch_size])
                    for idx in range(int(num_batches))]
        # Deal with the last batch
        if num_batches > int(num_batches):
            b_images.append(page_list[(int(num_batches) + 1) * self.batch_size:])

        return b_images

    def __call__(
        self,
        documents: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Prepare document data for model forwarding

        Args:
            documents: list of documents, where each document is a list of pages (numpy ndarray)
        Returns:
            list of page batches
        """

        # Resize the inputs
        images = self.resize_inputs(documents)
        # Batch them
        processed_batches = self.batch_inputs(images)
        # Normalize
        processed_batches = self.normalize_inputs(processed_batches)

        return processed_batches
