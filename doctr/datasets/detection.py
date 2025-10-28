# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from typing import Any

import numpy as np

from doctr.file_utils import CLASS_NAME

from .datasets import AbstractDataset
from .utils import pre_transform_multiclass

__all__ = ["DetectionDataset"]


class DetectionDataset(AbstractDataset):
    """Implements a text detection dataset

    >>> from doctr.datasets import DetectionDataset
    >>> train_set = DetectionDataset(img_folder="/path/to/images",
    >>>                              label_path="/path/to/labels.json")
    >>> img, target = train_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_path: path to the annotations of each image
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            img_folder,
            pre_transforms=pre_transform_multiclass,
            **kwargs,
        )

        # File existence check
        self._class_names: list = []
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"unable to locate {label_path}")
        with open(label_path, "rb") as f:
            labels = json.load(f)

        self.data: list[tuple[str, tuple[np.ndarray, list[str]]]] = []
        np_dtype = np.float32
        for img_name, label in labels.items():
            # File existence check
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")

            geoms, polygons_classes = self.format_polygons(label["polygons"], use_polygons, np_dtype)

            self.data.append((img_name, (np.asarray(geoms, dtype=np_dtype), polygons_classes)))

    def format_polygons(
        self, polygons: list | dict, use_polygons: bool, np_dtype: type
    ) -> tuple[np.ndarray, list[str]]:
        """Format polygons into an array

        Args:
            polygons: the bounding boxes
            use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
            np_dtype: dtype of array

        Returns:
            geoms: bounding boxes as np array
            polygons_classes: list of classes for each bounding box
        """
        if isinstance(polygons, list):
            self._class_names += [CLASS_NAME]
            polygons_classes = [CLASS_NAME for _ in polygons]
            _polygons: np.ndarray = np.asarray(polygons, dtype=np_dtype)
        elif isinstance(polygons, dict):
            self._class_names += list(polygons.keys())
            polygons_classes = [k for k, v in polygons.items() for _ in v]
            _polygons = np.concatenate([np.asarray(poly, dtype=np_dtype) for poly in polygons.values() if poly], axis=0)
        else:
            raise TypeError(f"polygons should be a dictionary or list, it was {type(polygons)}")
        geoms = _polygons if use_polygons else np.concatenate((_polygons.min(axis=1), _polygons.max(axis=1)), axis=1)
        return geoms, polygons_classes

    @property
    def class_names(self):
        return sorted(set(self._class_names))
