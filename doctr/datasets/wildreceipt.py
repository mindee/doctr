# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .datasets import AbstractDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["WILDRECEIPT"]


class WILDRECEIPT(AbstractDataset):
    """WildReceipt is a collection of receipts. It contains, for each photo, of a list of OCRs - with bounding box, text, and class."
    <https://arxiv.org/abs/2103.14470v1>`_ |
    `repository <https://download.openmmlab.com/mmocr/data/wildreceipt.tar>`_.

    >>> # NOTE: You need to download/generate the dataset from the repository.
    >>> from doctr.datasets import WILDRECEIPT
    >>> train_set = WILDRECEIPT(train=True, img_folder="/path/to/wildreceipt/image_files",
    >>>                     label_path="/path/to/wildreceipt/train.txt")
    >>> img, target = train_set[0]
    >>> test_set = WILDRECEIPT(train=False, img_folder="/path/to/wildreceipt/image_files",
    >>>                    label_path="/path/to/wildreceipt/test.txt")
    >>> img, target = test_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_path: path to the annotations file of the dataset
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
            self,
            img_folder: str,
            label_path: str,
            train: bool = True,
            use_polygons: bool = False,
            recognition_task: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            img_folder, pre_transforms=convert_target_to_relative if not recognition_task else None, **kwargs
        )
        # File existence check
        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(f"unable to locate {label_path if not os.path.exists(label_path) else img_folder}")

        tmp_root = os.path.join(self.root, 'wildreceipt/')
        self.train = train
        np_dtype = np.float32
        self.data: List[Tuple[str, Dict[str, Any]]] = []

        self.filename = "train.txt" if self.train else "test.txt"
        file_path = os.path.join(tmp_root, self.filename)

        with open(file_path, 'r') as file:
            data = file.read()
        # Split the text file into separate JSON strings
        json_strings = data.strip().split('\n')
        box: Union[List[float], np.ndarray]
        _targets = []
        for json_string in json_strings:
            json_data = json.loads(json_string)
            file_name = json_data['file_name']
            annotations = json_data['annotations']
            for annotation in annotations:
                coordinates = annotation['box']
                if use_polygons:
                    # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                    box = np.array(
                        [
                            [coordinates[0], coordinates[1]],
                            [coordinates[2], coordinates[3]],
                            [coordinates[4], coordinates[5]],
                            [coordinates[6], coordinates[7]],
                        ],
                        dtype=np_dtype
                    )
                else:
                    box = _convert_xmin_ymin(coordinates)
                _targets = [(_convert_xmin_ymin(annotation['box']), annotation['text'].lower(), annotation['label'])
                            for annotation in annotations]
                if _targets:
                    box_targets, text_units, labels = zip(*_targets)

                    self.data.append((
                        file_name,
                        dict(boxes=np.asarray(box_targets, dtype=int), labels=list(labels),
                             text_units=list(text_units)),
                    ))
        self.root = tmp_root


def extra_repr(self) -> str:
    return f"train={self.train}"


def _read_from_folder(self, path: str) -> None:
    for img_path in glob.glob(os.path.join(path, "*.png")):
        with open(os.path.join(path, f"{os.path.basename(img_path)[:-4]}.txt"), "r") as f:
            self.data.append((img_path, f.read()))


@classmethod
def _convert_xmin_ymin(box: List) -> List:
    if len(box) == 4:
        return box
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return [x_min, y_min, x_max, y_max]
