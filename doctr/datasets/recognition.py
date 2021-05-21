# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import tensorflow as tf
from typing import Tuple, List, Optional, Callable

from .core import AbstractDataset

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
        sample_transforms: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> None:
        self.sample_transforms = (lambda x: x) if sample_transforms is None else sample_transforms
        self.root = img_folder

        self.data: List[Tuple[str, str]] = []
        with open(labels_path) as f:
            labels = json.load(f)
        for img_path in os.listdir(self.root):
            if not os.path.exists(os.path.join(self.root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_path)}")
            label = labels.get(img_path)
            if not isinstance(label, str):
                raise KeyError("Image is not in referenced in label file")
            self.data.append((img_path, label))
