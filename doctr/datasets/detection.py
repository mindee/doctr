# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable

from .datasets import AbstractDataset
from doctr.utils.geometry import fit_rbbox

__all__ = ["DetectionDataset"]


class DetectionDataset(AbstractDataset):
    """Implements a text detection dataset

    Example::
        >>> from doctr.datasets import DetectionDataset
        >>> train_set = DetectionDataset(img_folder=True, label_path="/path/to/labels.json")
        >>> img, target = train_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_path: path to the annotations of each image
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
    """
    def __init__(
        self,
        img_folder: str,
        label_path: str,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)
        self.sample_transforms = sample_transforms

        # File existence check
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"unable to locate {label_path}")
        with open(label_path, 'rb') as f:
            labels = json.load(f)

        self.data: List[Tuple[str, np.ndarray]] = []
        np_dtype = np.float16 if self.fp16 else np.float32
        for img_name, label in labels.items():
            polygons = np.asarray(label['polygons'])
            if rotated_bbox:
                # Switch to rotated rects
                boxes = np.asarray([list(fit_rbbox(poly)) for poly in polygons])
            else:
                # Switch to xmin, ymin, xmax, ymax
                boxes = np.concatenate((polygons.min(axis=1), polygons.max(axis=1)), axis=1)

            self.data.append((img_name, np.asarray(boxes, dtype=np_dtype)))

    def __getitem__(
        self,
        index: int
    ) -> Tuple[Any, np.ndarray]:

        img, boxes = self._read_sample(index)
        h, w = self._get_img_shape(img)
        if self.sample_transforms is not None:
            img = self.sample_transforms(img)

        # Boxes
        boxes = boxes.copy()
        boxes[..., [0, 2]] /= w
        boxes[..., [1, 3]] /= h
        boxes = boxes.clip(0, 1)

        return img, boxes
