# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
from typing import Tuple, List, Optional, Callable, Any
from pathlib import Path, PurePath

from .datasets import AbstractDataset

__all__ = ["RecognitionDataset"]


class RecognitionDataset(AbstractDataset):
    """Dataset implementation for text recognition tasks

    Example::
        >>> from doctr.datasets import RecognitionDataset
        >>> train_set = RecognitionDataset(data_folder="/path/to/folder")
        >>> img, target = train_set[0]

    Args:
        data_folder: folder containing folder(s) of images and labels as json files, each image folder
        must be named "images" and be in the same subdirectory than the corresponding "labels.json"
        sample_transforms: composable transformations that will be applied to each image
    """
    def __init__(
        self,
        data_folder: str,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_folder, **kwargs)
        self.sample_transforms = (lambda x: x) if sample_transforms is None else sample_transforms

        self.data: List[Tuple[str, str]] = []
        # Recursively find json files
        json_files = Path(self.root).glob('**/*.json')
        for json_file in json_files:
            with json_file.open() as f:
                labels = json.load(f)
            imgs_path = PurePath(PurePath.parent(json_file), "images")
            for img_path in os.listdir(imgs_path):
                # File existence check
                if not os.path.exists(os.path.join(imgs_path, img_path)):
                    raise FileNotFoundError(f"unable to locate {os.path.join(imgs_path, img_path)}")
                label = labels.get(img_path)
                if not isinstance(label, str):
                    raise KeyError("Image is not in referenced in label file")
                self.data.append((img_path, label))
