# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from pathlib import Path
from typing import Any

from .datasets import AbstractDataset

__all__ = ["RecognitionDataset"]


class RecognitionDataset(AbstractDataset):
    """Dataset implementation for text recognition tasks

    >>> from doctr.datasets import RecognitionDataset
    >>> train_set = RecognitionDataset(img_folder="/path/to/images",
    >>>                                labels_path="/path/to/labels.json")
    >>> img, target = train_set[0]

    Args:
        img_folder: path to the images folder
        labels_path: pathe to the json file containing all labels (character sequences)
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        labels_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)

        self.data: list[tuple[str, str]] = []
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)

        for img_name, label in labels.items():
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")

            self.data.append((img_name, label))

    def merge_dataset(self, ds: AbstractDataset) -> None:
        # Update data with new root for self
        self.data = [(str(Path(self.root).joinpath(img_path)), label) for img_path, label in self.data]
        # Define new root
        self.root = Path("/")
        # Merge with ds data
        for img_path, label in ds.data:
            self.data.append((str(Path(ds.root).joinpath(img_path)), label))
