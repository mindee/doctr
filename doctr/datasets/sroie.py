# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import csv
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .datasets import VisionDataset

__all__ = ['SROIE']


class SROIE(VisionDataset):
    """SROIE dataset from `"ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction"
    <https://arxiv.org/pdf/2103.10213.pdf>`_.

    Example::
        >>> from doctr.datasets import SROIE
        >>> train_set = SROIE(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    TRAIN = ('https://github.com/mindee/doctr/releases/download/v0.1.1/sroie2019_train_task1.zip',
             'd4fa9e60abb03500d83299c845b9c87fd9c9430d1aeac96b83c5d0bb0ab27f6f')
    TEST = ('https://github.com/mindee/doctr/releases/download/v0.1.1/sroie2019_test.zip',
            '41b3c746a20226fddc80d86d4b2a903d43b5be4f521dd1bbe759dbf8844745e2')

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        url, sha256 = self.TRAIN if train else self.TEST
        super().__init__(url, None, sha256, True, **kwargs)
        self.sample_transforms = sample_transforms
        self.train = train

        tmp_root = os.path.join(self.root, 'images')
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float32

        for img_path in os.listdir(tmp_root):

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path)}")

            stem = Path(img_path).stem
            with open(os.path.join(self.root, 'annotations', f"{stem}.txt"), encoding='latin') as f:
                _rows = [row for row in list(csv.reader(f, delimiter=',')) if len(row) > 0]

            labels = [",".join(row[8:]) for row in _rows]
            # reorder coordinates (8 -> (4,2)) and filter empty lines
            coords = np.stack([np.array(list(map(int, row[:8])), dtype=np_dtype).reshape((4, 2))
                              for row in _rows], axis=0)

            if rotated_bbox:
                # x_center, y_center, w, h, alpha = 0
                mins = coords.min(axis=1)
                maxs = coords.max(axis=1)
                box_targets = np.concatenate(
                    ((mins + maxs) / 2, maxs - mins, np.zeros((coords.shape[0], 1))), axis=1)
            else:
                # xmin, ymin, xmax, ymax
                box_targets = np.concatenate((coords.min(axis=1), coords.max(axis=1)), axis=1)

            self.data.append((img_path, dict(boxes=box_targets, labels=labels)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
