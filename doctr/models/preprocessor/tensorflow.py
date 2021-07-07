# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
import numpy as np
from typing import List, Tuple, Union, Any

from doctr.utils.repr import NestedObject
from doctr.transforms import Normalize, Resize


__all__ = ['PreProcessor']


class PreProcessor(NestedObject):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        interpolation: one of 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5'

    """

    _children_names: List[str] = ['resize', 'normalize']

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
        self.normalize = Normalize(mean, std)

    def batch_inputs(
        self,
        x: List[tf.Tensor]
    ) -> List[tf.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            x: list of samples (tf.Tensor)

        Returns:
            list of batched samples
        """

        num_batches = len(x) / self.batch_size
        # Deal with fixed-size batches
        b_images = [tf.stack(x[idx * self.batch_size: (idx + 1) * self.batch_size], axis=0)
                    for idx in range(int(num_batches))]
        # Deal with the last batch
        if num_batches > int(num_batches):
            b_images.append(tf.stack(x[int(num_batches) * self.batch_size:], axis=0))
        return b_images

    def __call__(
        self,
        x: Union[tf.Tensor, List[np.ndarray]]
    ) -> List[tf.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tf.Tensor (already resized and batched)
        Returns:
            list of page batches
        """
        # Check input type
        if isinstance(x, tf.Tensor):
            # Inspect the data type before resizing (depending on interpolation method, it may cast it to fp32)
            input_dtype = x.dtype
            # Tf tensor from data loader: check if tensor size is output_size
            if x.shape[1] != self.resize.output_size[0] or x.shape[2] != self.resize.output_size[1]:
                x = tf.image.resize(x, self.resize.output_size, method=self.resize.method)
            if input_dtype == tf.uint8:
                x = tf.cast(x, dtype=tf.float32) / 255
            processed_batches = [x]
        elif isinstance(x, list):
            input_dtype = x[0].dtype
            # Convert to tensors & resize (and eventually pad) the inputs
            images = [self.resize(tf.convert_to_tensor(sample)) for sample in x]
            # Batch them
            processed_batches = self.batch_inputs(images)
            # Casting & 255 division
            if input_dtype == np.uint8:
                processed_batches = [tf.cast(b, dtype=tf.float32) / 255 for b in processed_batches]
        else:
            raise AssertionError("invalid input type")

        # Normalize
        processed_batches = [self.normalize(b) for b in processed_batches]

        return processed_batches
