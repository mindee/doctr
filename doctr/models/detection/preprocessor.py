# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import math
import json
import tensorflow as tf
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict

from ..preprocessor import PreProcessor

__all__ = ['DetectionPreProcessor']


class DetectionPreProcessor(PreProcessor):
    """Implements an abstract preprocessor object

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
        interpolation: str = 'bilinear'
    ) -> None:

        super().__init__(output_size, batch_size, mean, std, interpolation)

    def resize_fixed_h_and_w(
        self,
        x: tf.Tensor,
    ) -> tf.Tensor:
        """Resize images using tensorflow backend.

        Args:
            x: image as a tf.Tensor

        Returns:
            the processed image after being resized
        """

        return tf.image.resize(x, [self.output_size[0], self.output_size[1]], method=self.interpolation)

    def __call__(
        self,
        x: List[np.ndarray]
    ) -> List[tf.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array)
        Returns:
            list of page batches
        """
        # convert images to tf
        tensors = [tf.cast(sample, dtype=tf.float32) for sample in x]
        # Resize the inputs
        images = [self.resize_fixed_h_and_w(sample) for sample in tensors]
        # Batch them
        processed_batches = self.batch_inputs(images)
        # Normalize
        processed_batches = [self.normalize(b) for b in processed_batches]

        return processed_batches
