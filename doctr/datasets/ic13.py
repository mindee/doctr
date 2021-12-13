# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import csv
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .datasets import AbstractDataset

__all__ = ["IC13"]


class IC13(AbstractDataset):
    """IC13 dataset from `"ICDAR 2013 Robust Reading Competition" <https://rrc.cvc.uab.es/>`_.
    Example::
        >>> # NOTE: You need to download both image and label parts from Focused Scene Text challenge Task2.1 2013-2015.
        >>> from doctr.datasets import IC13
        >>> train_set = IC13(img_folder="/path/to/Challenge2_Training_Task12_Images",
        >>>                  label_folder="/path/to/Challenge2_Training_Task1_GT")
        >>> img, target = train_set[0]
        >>> test_set = IC13(img_folder="/path/to/Challenge2_Test_Task12_Images",
        >>>                 label_folder="/path/to/Challenge2_Test_Task1_GT")
        >>> img, target = test_set[0]
    Args:
        img_folder: folder with all the images of the dataset
        label_folder: folder with all annotation files for the images
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
    """

    def __init__(
        self,
        img_folder: str,
        label_folder: str,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
    ) -> None:
        super().__init__(img_folder)
        self.sample_transforms = sample_transforms

        # File existence check
        if not os.path.exists(label_folder) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_folder if not os.path.exists(label_folder) else img_folder}")

        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        np_dtype = np.float32

        img_names = os.listdir(img_folder)

        for img_name in img_names:

            img_path = Path(img_folder, img_name)
            label_path = Path(label_folder, "gt_" + Path(img_name).stem + ".txt")

            with open(label_path, newline='\n') as f:
                _lines = [
                    [val[:-1] if val.endswith(",") else val for val in row]
                    for row in csv.reader(f, delimiter=' ', quotechar="'")
                ]
            labels = [line[-1] for line in _lines]
            # xmin, ymin, xmax, ymax
            box_targets = np.array([list(map(int, line[:4])) for line in _lines], dtype=np_dtype)
            if rotated_bbox:
                # x_center, y_center, width, height, 0
                box_targets = np.array([[coords[0] + (coords[2] - coords[0]) / 2,
                                         coords[1] + (coords[3] - coords[1]) / 2,
                                         (coords[2] - coords[0]),
                                         (coords[3] - coords[1]), 0.0] for coords in box_targets], dtype=np_dtype)

            self.data.append((img_path, dict(boxes=box_targets, labels=labels)))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        img, target = self._read_sample(index)
        h, w = self._get_img_shape(img)
        if self.sample_transforms is not None:
            img = self.sample_transforms(img)

        # Boxes
        boxes = target['boxes'].copy()
        boxes[..., [0, 2]] /= w
        boxes[..., [1, 3]] /= h
        boxes = boxes.clip(0, 1)
        target['boxes'] = boxes

        return img, target
