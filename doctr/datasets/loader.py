# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Any

__all__ = ["DataLoader"]


def default_collate(samples):

    images, boxes, flags = zip(*samples)
    images = tf.stack(images, axis=0)

    return images, list(boxes), list(flags)


class DataLoader:
    def __init__(
        self,
        dataset,
        shuffle: bool = True,
        batch_size: int = 1,
        drop_last: bool = False,
        collate_fn: Optional[] = None
    ) -> None:
        self.data = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_batches = math.floor(len(self.data) / batch_size) if drop_last else math.ceil(len(self.data) / batch_size)
        self.collate_fn = default_collate if collate_fn is None else collate_fn
        self.reset()

    def reset(self) -> None:
        # Updates indices after each epoch
        self.indices = np.arange(len(self.data))
        if self.shuffle is True:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self._num_yielded = 0
        return self

    def __next__(self):
        if self._num_yielded < self.num_batches:
            # Get next indices
            indices = self.indices[self._num_yielded * self.batch_size: min(len(self.data), (self._num_yielded + 1) * self.batch_size)]

            samples = map(self.data.__getitem__, indices)

            batch_data = self.collate_fn(samples)

            self._num_yielded += 1
            return batch_data

        raise StopIteration()
