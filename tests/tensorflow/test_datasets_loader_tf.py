from typing import List, Tuple

import tensorflow as tf

from doctr.datasets import DataLoader


class MockDataset:
    def __init__(self, input_size):
        self.data: List[Tuple[float, bool]] = [
            (1, True),
            (0, False),
            (0.5, True),
        ]
        self.input_size = input_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        val, label = self.data[index]
        return tf.cast(tf.fill(self.input_size, val), dtype=tf.float32), tf.constant(label, dtype=tf.bool)


class MockDatasetBis(MockDataset):
    @staticmethod
    def collate_fn(samples):
        x, y = zip(*samples)
        return tf.stack(x, axis=0), list(y)


def test_dataloader():
    loader = DataLoader(
        MockDataset((32, 32)),
        shuffle=True,
        batch_size=2,
        drop_last=True,
    )

    ds_iter = iter(loader)
    num_batches = 0
    for x, y in ds_iter:
        num_batches += 1
    assert len(loader) == 1
    assert num_batches == 1
    assert isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor)
    assert x.shape == (2, 32, 32)
    assert y.shape == (2,)

    # Drop last
    loader = DataLoader(
        MockDataset((32, 32)),
        shuffle=True,
        batch_size=2,
        drop_last=False,
    )
    ds_iter = iter(loader)
    num_batches = 0
    for x, y in ds_iter:
        num_batches += 1
    assert loader.num_batches == 2
    assert num_batches == 2

    # Custom collate
    loader = DataLoader(
        MockDatasetBis((32, 32)),
        shuffle=True,
        batch_size=2,
        drop_last=False,
    )

    ds_iter = iter(loader)
    x, y = next(ds_iter)
    assert isinstance(x, tf.Tensor) and isinstance(y, list)
    assert x.shape == (2, 32, 32)
    assert len(y) == 2
