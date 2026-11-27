# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from typing import Any

import numpy as np

from .datasets import AbstractDataset

__all__ = ["OrientationDataset"]


class OrientationDataset(AbstractDataset):
    """Implements a basic image dataset where targets are filled with zeros.

    >>> from doctr.datasets import OrientationDataset
    >>> train_set = OrientationDataset(img_folder="/path/to/images")
    >>> sample = train_set[0]
    >>> img, target = sample.image, sample.target

    Args:
        img_folder: folder with all the images of the dataset
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            img_folder,
            **kwargs,
        )

        # initialize dataset with 0 degree rotation targets
        # a list-like sequence (only `len` and indexing are used by the base class), not a list: cf. _ZeroTargetSamples
        self.data = _ZeroTargetSamples(os.listdir(self.root))  # type: ignore[assignment]


class _ZeroTargetSamples:
    """The (image name, 0 degree target) samples, with all the names in a single numpy array:
    a list of Python objects per image would be copied by every DataLoader worker (refcount updates trigger
    copy-on-write), which takes tens of GB with millions of images.
    """

    def __init__(self, img_names: list[str]) -> None:
        self.img_names = np.array(img_names)

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray]:
        return str(self.img_names[index]), np.array([0])
