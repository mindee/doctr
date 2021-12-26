# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import json
import os
from typing import Any, List, Tuple

import numpy as np

from .datasets import AbstractDataset

__all__ = ["DetectionDataset"]


class DetectionDataset(AbstractDataset):
    """Implements a text detection dataset

    Example::
        >>> from doctr.datasets import DetectionDataset
        >>> train_set = DetectionDataset(img_folder="/path/to/images", label_path="/path/to/labels.json")
        >>> img, target = train_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_path: path to the annotations of each image
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)

        # File existence check
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"unable to locate {label_path}")
        with open(label_path, 'rb') as f:
            labels = json.load(f)

        self.data: List[Tuple[str, np.ndarray]] = []
        for img_name, label in labels.items():
            # File existence check
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")

            polygons = np.asarray(label['polygons'])
            geoms = polygons if use_polygons else np.concatenate((polygons.min(axis=1), polygons.max(axis=1)), axis=1)

            self.data.append((img_name, np.asarray(geoms, dtype=np.float32)))

    def __getitem__(
        self,
        index: int
    ) -> Tuple[Any, np.ndarray]:

        img, target = self._read_sample(index)
        h, w = self._get_img_shape(img)
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        if self.sample_transforms is not None:
            img, target = self.sample_transforms(img, target)

        # Boxes
        target = target.copy()
        target[..., 0] /= w
        target[..., 1] /= h
        target = target.clip(0, 1)

        return img, target
