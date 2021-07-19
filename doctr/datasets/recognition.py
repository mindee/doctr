# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
from typing import Tuple, List, Optional, Callable, Any
from pathlib import Path

from .datasets import AbstractDataset

__all__ = ["RecognitionDataset"]


class RecognitionDataset(AbstractDataset):
    """Dataset implementation for text recognition tasks

    Example::
        >>> from doctr.datasets import RecognitionDataset
        >>> train_set = RecognitionDataset(img_folder=True, labels_path="/path/to/labels.json")
        >>> img, target = train_set[0]

    Args:
        img_folder: path to the images folder
        labels_path: pathe to the json file containing all labels (character sequences)
        sample_transforms: composable transformations that will be applied to each image
    """
    def __init__(
        self,
        img_folder: str,
        labels_path: str,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)
        self.sample_transforms = (lambda x: x) if sample_transforms is None else sample_transforms

        self.data: List[Tuple[str, str]] = []
        with open(labels_path) as f:
            labels = json.load(f)
        for img_path in os.listdir(self.root):
            # File existence check
            if not os.path.exists(os.path.join(self.root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_path)}")
            label = labels.get(img_path)
            if not isinstance(label, str):
                raise KeyError("Image is not in referenced in label file")
            self.data.append((img_path, label))

    def merge_dataset(self, ds: AbstractDataset) -> None:
        # Update data with new root for self
        self.data = [(str(Path(self.root).joinpath(img_path)), label) for img_path, label in self.data]
        # Define new root
        self.root = Path("/")
        # Merge with ds data
        for img_path, label in ds.data:
            self.data.append((str(Path(ds.root).joinpath(img_path)), label))
