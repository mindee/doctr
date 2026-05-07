# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from typing import Any

import numpy as np

from .datasets import AbstractDataset
from .utils import pre_transform_multiclass

__all__ = ["LayoutDataset"]


class LayoutDataset(AbstractDataset):
    """Implements a document layout detection dataset.

    >>> from doctr.datasets import LayoutDataset
    >>> train_set = LayoutDataset(
    >>>     img_folder="/path/to/images",
    >>>     label_path="/path/to/labels.json",
    >>> )
    >>> img, target = train_set[0]

    Args:
        img_folder: folder containing the dataset images
        label_path: path to the labels.json file
        use_polygons: whether to keep polygons instead of converting to straight boxes
        **kwargs: keyword arguments from `AbstractDataset`
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
        self._class_names: list[str] = []
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"unable to locate {label_path}")
        with open(label_path, "rb") as f:
            labels = json.load(f)

        self.data: list[tuple[str, tuple[np.ndarray, list[str]]]] = []
        np_dtype = np.float32

        for img_name, label in labels.items():
            img_path = os.path.join(self.root, img_name)

            # File existence check
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"unable to locate {img_path}")

            polygons = label.get("polygons")
            class_names = label.get("class_names")

            if polygons is None:
                raise KeyError(f"missing 'polygons' for image: {img_name}")
            if class_names is None:
                raise KeyError(f"missing 'class_names' for image: {img_name}")

            if len(polygons) != len(class_names):
                raise ValueError(
                    f"number of polygons ({len(polygons)}) does not match "
                    f"number of class_names ({len(class_names)}) for image: {img_name}"
                )

            geoms, polygon_classes = self.format_polygons(
                polygons=polygons,
                class_names=class_names,
                use_polygons=use_polygons,
                np_dtype=np_dtype,
            )

            self.data.append((
                img_name,
                (
                    np.asarray(geoms, dtype=np_dtype),
                    polygon_classes,
                ),
            ))

    def format_polygons(
        self,
        polygons: list,
        class_names: list[str],
        use_polygons: bool,
        np_dtype: type,
    ) -> tuple[np.ndarray, list[str]]:
        """Format polygons into an array.

        Args:
            polygons: list of polygons
            class_names: list of class names corresponding to polygons
            use_polygons: whether polygons should be preserved
            np_dtype: numpy dtype

        Returns:
            geoms: polygons or straight boxes
            polygon_classes: list of classes
        """
        self._class_names += class_names

        _polygons: np.ndarray = np.asarray(polygons, dtype=np_dtype)
        if _polygons.ndim != 3 or _polygons.shape[1:] != (4, 2):
            raise ValueError(f"polygons are expected to have shape (N, 4, 2), got {_polygons.shape}")
        geoms = _polygons if use_polygons else np.concatenate((_polygons.min(axis=1), _polygons.max(axis=1)), axis=1)

        return geoms, class_names

    @property
    def class_names(self) -> list[str]:
        return sorted(set(self._class_names))
