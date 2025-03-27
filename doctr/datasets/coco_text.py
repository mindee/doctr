# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .datasets import AbstractDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["COCOTEXT"]


class COCOTEXT(AbstractDataset):
    """
    COCO-Text dataset from `"COCO-Text: Dataset and Benchmark for Text Detection and Recognition in Natural Images"
    <https://arxiv.org/pdf/1601.07140v2>`_ |
    `"homepage" <https://bgshih.github.io/cocotext/>`_.

    >>> # NOTE: You need to download the dataset first.
    >>> from doctr.datasets import COCOTEXT
    >>> train_set = COCOTEXT(train=True, img_folder="/path/to/coco_text/train2014/",
    >>>                     label_path="/path/to/coco_text/cocotext.v2.json")
    >>> img, target = train_set[0]
    >>> test_set = COCOTEXT(train=False, img_folder="/path/to/coco_text/train2014/",
    >>> label_path = "/path/to/coco_text/cocotext.v2.json")
    >>> img, target = test_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_path: path to the annotations file of the dataset
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        detection_task: whether the dataset should be used for detection task
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        train: bool = True,
        use_polygons: bool = False,
        recognition_task: bool = False,
        detection_task: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            img_folder, pre_transforms=convert_target_to_relative if not recognition_task else None, **kwargs
        )
        # Task check
        if recognition_task and detection_task:
            raise ValueError(
                " 'recognition' and 'detection task' cannot be set to True simultaneously. "
                + " To get the whole dataset with boxes and labels leave both parameters to False "
            )

        # File existence check
        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(f"unable to find {label_path if not os.path.exists(label_path) else img_folder}")

        tmp_root = img_folder
        self.train = train
        np_dtype = np.float32
        self.data: list[tuple[str | Path | np.ndarray, str | dict[str, Any] | np.ndarray]] = []

        with open(label_path, "r") as file:
            data = json.load(file)

        # Filter images based on the set
        img_items = [img for img in data["imgs"].items() if (img[1]["set"] == "train") == train]
        box: list[float] | np.ndarray

        for img_id, img_info in tqdm(img_items, desc="Preparing and Loading COCOTEXT", total=len(img_items)):
            img_path = os.path.join(img_folder, img_info["file_name"])

            # File existence check
            if not os.path.exists(img_path):  # pragma: no cover
                raise FileNotFoundError(f"Unable to locate {img_path}")

            # Get annotations for the current image (only legible text)
            annotations = [
                ann
                for ann in data["anns"].values()
                if ann["image_id"] == int(img_id) and ann["legibility"] == "legible"
            ]

            # Some images have no annotations with readable text
            if not annotations:  # pragma: no cover
                continue

            _targets = []

            for annotation in annotations:
                x, y, w, h = annotation["bbox"]
                if use_polygons:
                    # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                    box = np.array(
                        [
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h],
                        ],
                        dtype=np_dtype,
                    )
                else:
                    # (xmin, ymin, xmax, ymax) coordinates
                    box = [x, y, x + w, y + h]
                _targets.append((annotation["utf8_string"], box))
            text_targets, box_targets = zip(*_targets)

            if recognition_task:
                crops = crop_bboxes_from_image(
                    img_path=os.path.join(tmp_root, img_path), geoms=np.asarray(box_targets, dtype=int).clip(min=0)
                )
                for crop, label in zip(crops, list(text_targets)):
                    if label and " " not in label:
                        self.data.append((crop, label))

            elif detection_task:
                self.data.append((img_path, np.asarray(box_targets, dtype=int).clip(min=0)))
            else:
                self.data.append((
                    img_path,
                    dict(boxes=np.asarray(box_targets, dtype=int).clip(min=0), labels=list(text_targets)),
                ))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
