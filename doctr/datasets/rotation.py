# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from typing import Any, List, Tuple

import numpy as np

from .datasets import AbstractDataset

__all__ = ["RotationDataset"]


class RotationDataset(AbstractDataset):
    """Implements a basic image dataset where targets are filled with zeros.

    >>> from doctr.datasets import RotationDataset
    >>> train_set = RotationDataset(img_folder="/path/to/images")
    >>> img, target = train_set[0]

    Args:
    ----
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

        self.data: List[Tuple[str, np.ndarray]] = []
        for img_name in os.listdir(self.root):
            # File existence check
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")
            self.data.append((img_name, np.array([0])))
