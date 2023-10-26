# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import glob
import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from .datasets import AbstractDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["WILDRECEIPT"]


class WILDRECEIPT(AbstractDataset):
    """WildReceipt is a collection of receipts. It contains, for each photo, of a list of OCRs - with bounding box, text, and class."
    <https://arxiv.org/abs/2103.14470v1>`_ |
    `repository <https://download.openmmlab.com/mmocr/data/wildreceipt.tar>`_.

    >>> # NOTE: You need to download/generate the dataset from the repository.
    >>> from doctr.datasets import WILDRECEIPT
    >>> train_set = WILDRECEIPT(train=True, img_folder="/path/to/wildreceipt/",
    >>>                     label_path="/path/to/wildreceipt/train.txt")
    >>> img, target = train_set[0]
    >>> test_set = WILDRECEIPT(train=False, img_folder="/path/to/wildreceipt/",
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

        tmp_root = img_folder
        self.train = train
        np_dtype = np.float32
        self.data: List[Tuple[str, Dict[str, Any]]] = []


        # define folder to write IMGUR5K recognition dataset
        reco_folder_name = "WILDRECEIPT_recognition_train" if self.train else "WILDRECEIPT_recognition_test"
        reco_folder_name = "Poly_" + reco_folder_name if use_polygons else reco_folder_name
        reco_folder_path = os.path.join(os.path.dirname(self.root), reco_folder_name)
        reco_images_counter = 0

        if recognition_task and os.path.isdir(reco_folder_path):
            self._read_from_folder(reco_folder_path)
            return
        elif recognition_task and not os.path.isdir(reco_folder_path):
            os.makedirs(reco_folder_path, exist_ok=False)

        with open(label_path, 'r') as file:
            data = file.read()
        # Split the text file into separate JSON strings
        json_strings = data.strip().split('\n')
        box: Union[List[float], np.ndarray]
        _targets = []
        for json_string in json_strings:
            json_data = json.loads(json_string)
            img_path = json_data['file_name']
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
                    box = self._convert_xmin_ymin(coordinates)
                _targets.append((annotation['text'], box))
            text_targets, box_targets = zip(*_targets)

            if recognition_task:
                crops = crop_bboxes_from_image(
                    img_path=os.path.join(tmp_root, img_path), geoms=np.asarray(box_targets, dtype=int).clip(min=0)
                )
                for crop, label in zip(crops, list(text_targets)):
                    with open(os.path.join(reco_folder_path, f"{reco_images_counter}.txt"), "w") as f:
                        f.write(label)
                        tmp_img = Image.fromarray(crop)
                        tmp_img.save(os.path.join(reco_folder_path, f"{reco_images_counter}.png"))
                        reco_images_counter += 1
                    # self.data.append((crop, label))
            else:
                self.data.append(
                    (img_path, dict(boxes=np.asarray(box_targets, dtype=int).clip(min=0), labels=list(text_targets)))
                )
        if recognition_task:
            self._read_from_folder(reco_folder_path)
        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"

    def _read_from_folder(self, path: str) -> None:
        for img_path in glob.glob(os.path.join(path, "*.png")):
            with open(os.path.join(path, f"{os.path.basename(img_path)[:-4]}.txt"), "r") as f:
                self.data.append((img_path, f.read()))

    @staticmethod
    def _convert_xmin_ymin(box: List) -> List:
        if len(box) == 4:
            return box
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        x_min = min(x1, x2, x3, x4)
        x_max = max(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)
        return [x_min, y_min, x_max, y_max]
