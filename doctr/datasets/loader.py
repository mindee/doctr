# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from doctr.utils.multithreading import multithread_exec

__all__ = ["DataLoader"]


def default_collate(samples):
    """Collate multiple elements into batches

    Args:
    ----
        samples: list of N tuples containing M elements

    Returns:
    -------
        Tuple of M sequences contianing N elements each
    """
    batch_data = zip(*samples)

    tf_data = tuple(tf.stack(elt, axis=0) for elt in batch_data)

    return tf_data


class DataLoader:
    """Implements a dataset wrapper for fast data loading

    >>> from doctr.datasets import CORD, DataLoader
    >>> train_set = CORD(train=True, download=True)
    >>> train_loader = DataLoader(train_set, batch_size=32)
    >>> train_iter = iter(train_loader)
    >>> images, targets = next(train_iter)

    Args:
    ----
        dataset: the dataset
        shuffle: whether the samples should be shuffled before passing it to the iterator
        batch_size: number of elements in each batch
        drop_last: if `True`, drops the last batch if it isn't full
        num_workers: number of workers to use for data loading
        collate_fn: function to merge samples into a batch
    """

    def __init__(
        self,
        dataset,
        shuffle: bool = True,
        batch_size: int = 1,
        drop_last: bool = False,
        num_workers: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        nb = len(self.dataset) / batch_size
        self.num_batches = math.floor(nb) if drop_last else math.ceil(nb)
        if collate_fn is None:
            self.collate_fn = self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else default_collate
        else:
            self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.reset()

    def __len__(self) -> int:
        return self.num_batches

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
        if self._num_yielded < self.num_batches:
            # Get next indices
            idx = self._num_yielded * self.batch_size
            indices = self.indices[idx : min(len(self.dataset), idx + self.batch_size)]

            samples = list(multithread_exec(self.dataset.__getitem__, indices, threads=self.num_workers))

            batch_data = self.collate_fn(samples)

            self._num_yielded += 1
            return batch_data
        else:
            raise StopIteration
