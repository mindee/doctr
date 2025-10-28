# Copyright (C) 2021-2025, Mindee.

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
    >>> img, target = train_set[0]

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
        self.data: list[tuple[str, np.ndarray]] = [(img_name, np.array([0])) for img_name in os.listdir(self.root)]
