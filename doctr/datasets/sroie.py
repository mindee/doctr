# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from .datasets import VisionDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["SROIE"]


class SROIE(VisionDataset):
    """SROIE dataset from `"ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction"
    <https://arxiv.org/pdf/2103.10213.pdf>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/sroie-grid.png&src=0
        :align: center

    >>> from doctr.datasets import SROIE
    >>> train_set = SROIE(train=True, download=True)
    >>> img, target = train_set[0]

    Args:
    ----
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        **kwargs: keyword arguments from `VisionDataset`.
    """

    TRAIN = (
        "https://doctr-static.mindee.com/models?id=v0.1.1/sroie2019_train_task1.zip&src=0",
        "d4fa9e60abb03500d83299c845b9c87fd9c9430d1aeac96b83c5d0bb0ab27f6f",
        "sroie2019_train_task1.zip",
    )
    TEST = (
        "https://doctr-static.mindee.com/models?id=v0.1.1/sroie2019_test.zip&src=0",
        "41b3c746a20226fddc80d86d4b2a903d43b5be4f521dd1bbe759dbf8844745e2",
        "sroie2019_test.zip",
    )

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        recognition_task: bool = False,
        **kwargs: Any,
    ) -> None:
        url, sha256, name = self.TRAIN if train else self.TEST
        super().__init__(
            url,
            name,
            sha256,
            True,
            pre_transforms=convert_target_to_relative if not recognition_task else None,
            **kwargs,
        )
        self.train = train

        tmp_root = os.path.join(self.root, "images")
        self.data: List[Tuple[Union[str, np.ndarray], Union[str, Dict[str, Any]]]] = []
        np_dtype = np.float32

        for img_path in tqdm(iterable=os.listdir(tmp_root), desc="Unpacking SROIE", total=len(os.listdir(tmp_root))):
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path)}")

            stem = Path(img_path).stem
            with open(os.path.join(self.root, "annotations", f"{stem}.txt"), encoding="latin") as f:
                _rows = [row for row in list(csv.reader(f, delimiter=",")) if len(row) > 0]

            labels = [",".join(row[8:]) for row in _rows]
            # reorder coordinates (8 -> (4,2) ->
            # (x, y) coordinates of top left, top right, bottom right, bottom left corners) and filter empty lines
            coords: np.ndarray = np.stack(
                [np.array(list(map(int, row[:8])), dtype=np_dtype).reshape((4, 2)) for row in _rows], axis=0
            )

            if not use_polygons:
                # xmin, ymin, xmax, ymax
                coords = np.concatenate((coords.min(axis=1), coords.max(axis=1)), axis=1)

            if recognition_task:
                crops = crop_bboxes_from_image(img_path=os.path.join(tmp_root, img_path), geoms=coords)
                for crop, label in zip(crops, labels):
                    if crop.shape[0] > 0 and crop.shape[1] > 0 and len(label) > 0:
                        self.data.append((crop, label))
            else:
                self.data.append((img_path, dict(boxes=coords, labels=labels)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
