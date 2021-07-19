# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Union, Any

from doctr.utils.repr import NestedObject
from doctr.transforms import Normalize, Resize
from doctr.utils.multithreading import multithread_exec


__all__ = ['PreProcessor']


class PreProcessor(NestedObject):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        fp16: whether returned batches should be in FP16
    """

    _children_names: List[str] = ['resize', 'normalize']

    def __init__(
        self,
        output_size: Tuple[int, int],
        batch_size: int,
        mean: Tuple[float, float, float] = (.5, .5, .5),
        std: Tuple[float, float, float] = (1., 1., 1.),
        fp16: bool = False,
        **kwargs: Any,
    ) -> None:

        self.batch_size = batch_size
        self.resize = Resize(output_size, **kwargs)
        # Perform the division by 255 at the same time
        self.normalize = Normalize(mean, std)
        self.fp16 = fp16

    def batch_inputs(
        self,
        samples: List[tf.Tensor]
    ) -> List[tf.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples (tf.Tensor)

        Returns:
            list of batched samples
        """

        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [
            tf.stack(samples[idx * self.batch_size: min((idx + 1) * self.batch_size, len(samples))], axis=0)
            for idx in range(int(num_batches))
        ]

        return batches

    def sample_transforms(self, x: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float16, np.float32):
                raise TypeError("unsupported data type for numpy.ndarray")
            x = tf.convert_to_tensor(x)
        elif x.dtype not in (tf.uint8, tf.float16, tf.float32):
            raise TypeError("unsupported data type for torch.Tensor")
        # Data type & 255 division
        if x.dtype == tf.uint8:
            x = tf.image.convert_image_dtype(x, dtype=tf.float32)
        # Resizing
        x = self.resize(x)

        return x

    def __call__(
        self,
        x: Union[tf.Tensor, np.ndarray, List[Union[tf.Tensor, np.ndarray]]]
    ) -> List[tf.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tensors (already resized and batched)
        Returns:
            list of page batches
        """

        # Input type check
        if isinstance(x, (np.ndarray, tf.Tensor)):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if isinstance(x, np.ndarray):
                if x.dtype not in (np.uint8, np.float16, np.float32):
                    raise TypeError("unsupported data type for numpy.ndarray")
                x = tf.convert_to_tensor(x)
            elif x.dtype not in (tf.uint8, tf.float16, tf.float32):
                raise TypeError("unsupported data type for torch.Tensor")

            # Data type & 255 division
            if x.dtype == tf.uint8:
                x = tf.image.convert_image_dtype(x, dtype=tf.float32)
            # Resizing
            if x.shape[1] != self.resize.output_size[0] or x.shape[2] != self.resize.output_size[1]:
                x = tf.image.resize(x, self.resize.output_size, method=self.resize.method)

            batches = [x]

        elif isinstance(x, list) and all(isinstance(sample, (np.ndarray, tf.Tensor)) for sample in x):
            # Sample transform (to tensor, resize)
            samples = multithread_exec(self.sample_transforms, x)
            # Batching
            batches = self.batch_inputs(samples)  # type: ignore[arg-type]
        else:
            raise TypeError(f"invalid input type: {type(x)}")

        # Batch transforms (normalize)
        batches = multithread_exec(self.normalize, batches)  # type: ignore[assignment]

        # Resize outputs tf.float32
        if self.fp16:
            batches = [tf.cast(b, dtype=tf.float16) for b in batches]

        return batches
