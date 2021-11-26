# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Callable, List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np

from .datasets import AbstractDataset

__all__ = ["ICDAR2019"]


class ICDAR2019(AbstractDataset):
    """ICDAR2019 dataset from
    `"ICDAR 2019 Robust Reading Challenge on Multi-lingual scene text detection and recognition"
    <https://rrc.cvc.uab.es/>`_.

    Example::
        >>> # NOTE: You need to download both image parts and the label part from MLT 2019 challenge.
        >>> # (https://rrc.cvc.uab.es/) : TrainSetImagesTask1_Part1 | TrainSetImagesTask1_Part2 | TrainSetGT
        >>> # structure: - img_folder: all images (part1 and part2) | - label_folder: all labels
        >>> from doctr.datasets import ICDAR2019
        >>> train_set = ICDAR2019(img_folder="/path/to/images", label_folder="/path/to/labels", train=True)
        >>> img, target = train_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_folder: folder with all annotation files for the images
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
    """

    def __init__(
        self,
        img_folder: str,
        label_folder: str,
        train: bool = True,
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
        split = int(len(img_names) * 0.9)
        img_names = img_names[:split] if train else img_names[split:]

        for img_name in img_names:

            img_path = Path(img_folder, img_name)
            label_path = Path(label_folder, Path(img_name).stem + ".txt")

            with open(label_path, "r") as f:
                _lines = [[x for x in line.split(',')] for line in f.readlines()]
                labels = [line[9].strip() for line in _lines]
                vocab_lang = [line[8] for line in _lines]
                if not labels or not vocab_lang:
                    continue
                pts = [list(map(int, line[:8])) for line in _lines]
                # xmin, ymin, xmax, ymax
                coords = [[min(pt[0], pt[2], pt[4], pt[6]), min(pt[1], pt[3], pt[5], pt[7]),
                           max(pt[0], pt[2], pt[4], pt[6]), max(pt[1], pt[3], pt[5], pt[7])]
                          for pt in pts]
                if rotated_bbox:
                    # x_center, y_center, w, h, 0
                    boxes = [[coord[0] + (coord[2] - coord[0]) / 2,
                              coord[1] + (coord[3] - coord[1]) / 2,
                              coord[2] - coord[0], coord[3] - coord[1], 0] for coord in coords]
                else:
                    # xmin, ymin, xmax, ymax
                    boxes = [[coord[0], coord[1], coord[2], coord[3]] for coord in coords]

                self.data.append((img_path, dict(boxes=np.asarray(
                    boxes, dtype=np_dtype), languages=vocab_lang, labels=labels)))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        img, target = self._read_sample(index)
        h, w = self._get_img_shape(img)
        if self.sample_transforms is not None:
            img = self.sample_transforms(img)

        return img, target
