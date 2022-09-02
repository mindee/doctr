# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from .datasets import VisionDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["FUNSD"]


class FUNSD(VisionDataset):
    """FUNSD dataset from `"FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents"
    <https://arxiv.org/pdf/1905.13538.pdf>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/funsd-grid.png&src=0
        :align: center

    >>> from doctr.datasets import FUNSD
    >>> train_set = FUNSD(train=True, download=True)
    >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    SHA256 = "c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f"
    FILE_NAME = "funsd.zip"

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        recognition_task: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            self.URL,
            self.FILE_NAME,
            self.SHA256,
            True,
            pre_transforms=convert_target_to_relative if not recognition_task else None,
            **kwargs,
        )
        self.train = train
        np_dtype = np.float32

        # Use the subset
        subfolder = os.path.join("dataset", "training_data" if train else "testing_data")

        # # List images
        tmp_root = os.path.join(self.root, subfolder, "images")
        self.data: List[Tuple[Union[str, np.ndarray], Union[str, Dict[str, Any]]]] = []
        for img_path in tqdm(iterable=os.listdir(tmp_root), desc="Unpacking FUNSD", total=len(os.listdir(tmp_root))):
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path)}")

            stem = Path(img_path).stem
            with open(os.path.join(self.root, subfolder, "annotations", f"{stem}.json"), "rb") as f:
                data = json.load(f)

            _targets = [
                (word["text"], word["box"])
                for block in data["form"]
                for word in block["words"]
                if len(word["text"]) > 0
            ]
            text_targets, box_targets = zip(*_targets)
            if use_polygons:
                # xmin, ymin, xmax, ymax -> (x, y) coordinates of top left, top right, bottom right, bottom left corners
                box_targets = [
                    [
                        [box[0], box[1]],
                        [box[2], box[1]],
                        [box[2], box[3]],
                        [box[0], box[3]],
                    ]
                    for box in box_targets
                ]

            if recognition_task:
                crops = crop_bboxes_from_image(
                    img_path=os.path.join(tmp_root, img_path), geoms=np.asarray(box_targets, dtype=np_dtype)
                )
                for crop, label in zip(crops, list(text_targets)):
                    # filter labels with unknown characters
                    if not any(char in label for char in ["☑", "☐", "\uf703", "\uf702"]):
                        self.data.append((crop, label))
            else:
                self.data.append(
                    (
                        img_path,
                        dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=list(text_targets)),
                    )
                )

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
