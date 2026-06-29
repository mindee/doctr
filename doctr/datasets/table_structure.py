# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from typing import Any, cast

import numpy as np

from doctr.io import read_img_as_tensor
from doctr.io.image import get_img_shape
from doctr.utils import Sample
from doctr.utils.geometry import convert_to_relative_coords

from .datasets import AbstractDataset

__all__ = ["TableStructureDataset"]


class TableStructureDataset(AbstractDataset):
    """Table structure recognition dataset for table structure recognition.

    The labels file maps each image name to its cells and their logical coordinates::

        {
            "table_0.jpg": {
                "cells": [[[x0, y0], [x1, y1], [x2, y2], [x3, y3]], ...],   # quads (TL, TR, BR, BL), abs px
                "logic": [[start_col, end_col, start_row, end_row], ...]      # 0-indexed, per cell
            },
            ...
        }

    Each sample yields the image and a target containing relative cells and their logical coordinates. Cells have
    shape `(N, 4)` by default, or `(N, 4, 2)` when `use_polygons=True`. Logical coordinates have shape
    `(N, 4)`.

    >>> from doctr.datasets import TableStructureDataset
    >>> from doctr.transforms import Resize
    >>> train_set = TableStructureDataset(
    >>>     img_folder="/path/to/images",
    >>>     label_path="/path/to/labels.json",
    >>> )
    >>> img, target = train_set[0]

    Args:
        img_folder: folder with all the dataset images
        label_path: path to the JSON labels
        use_polygons: whether to keep cell polygons instead of converting them to straight boxes
        **kwargs: keyword arguments from `AbstractDataset` (e.g. ``img_transforms``, ``sample_transforms``)
    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"unable to locate {label_path}")
        with open(label_path, "rb") as f:
            labels = json.load(f)

        self.data: list[tuple[str, dict[str, np.ndarray]]] = []
        for img_name, label in labels.items():
            img_path = os.path.join(self.root, img_name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"unable to locate {img_path}")

            cells = np.asarray(label["cells"], dtype=np.float32)
            logic = np.asarray(label["logic"], dtype=np.int64)
            if cells.ndim != 3 or cells.shape[1:] != (4, 2):  # pragma: no cover
                raise ValueError(f"cells are expected to have shape (N, 4, 2), got {cells.shape}")
            if logic.shape[0] != cells.shape[0] or logic.shape[1] != 4:  # pragma: no cover
                raise ValueError(f"logic is expected to have shape (N, 4), got {logic.shape}")
            if not use_polygons:
                cells = np.concatenate((cells.min(axis=1), cells.max(axis=1)), axis=1)
            self.data.append((img_name, {"cells": cells, "logic": logic}))

    # NOTE: Override basic dataset method __getitem__ to handle table-specific targets
    def __getitem__(self, index: int) -> Sample:
        img_name, label = self.data[index]
        img = read_img_as_tensor(os.path.join(self.root, img_name))
        # Convert cells to relative coordinates, then let img/sample transforms resize image + geometry.
        rel_cells = np.clip(convert_to_relative_coords(label["cells"].copy(), get_img_shape(img)), 0, 1)
        sample = Sample(image=img, mask=None, target=rel_cells)
        if self.img_transforms is not None:
            sample = self.img_transforms(sample)
        if self.sample_transforms is not None:
            sample = self.sample_transforms(sample)
        target: dict[str, np.ndarray] = {"cells": cast(np.ndarray, sample.target), "logic": label["logic"]}
        return Sample(image=sample.image, mask=None, target=target)
