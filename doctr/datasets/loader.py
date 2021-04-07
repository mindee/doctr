# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Any

__all__ = ["DataLoader"]


def default_collate(samples):

    images, targets = zip(*samples)

    return tf.stack(images, axis=0), tf.stack(targets, axis=0)


class DataLoader:
    def __init__(
        self,
        dataset,
        shuffle: bool = True,
        batch_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_batches = math.floor(len(self.dataset) / batch_size) if drop_last else math.ceil(len(self.dataset) / batch_size)
        self.collate_fn = self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else default_collate
        self.reset()

    def reset(self) -> None:
        # Updates indices after each epoch
        self._num_yielded = 0
        self.indices = np.arange(len(self.dataset))
        if self.shuffle is True:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self._num_yielded <= self.num_batches:
            # Get next indices
            indices = self.indices[self._num_yielded * self.batch_size: min(len(self.dataset), (self._num_yielded + 1) * self.batch_size)]

            samples = map(self.dataset.__getitem__, indices)

            batch_data = self.collate_fn(samples)

            self._num_yielded += 1
            return batch_data
        else:
            raise StopIteration
