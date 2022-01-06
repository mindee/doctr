# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .datasets import AbstractDataset
from .utils import convert_target_to_relative

__all__ = ["IMGUR5K"]


class IMGUR5K(AbstractDataset):
    """IMGUR5K dataset from `"TextStyleBrush: Transfer of Text Aesthetics from a Single Example"
    <https://arxiv.org/abs/2106.08385>`_ |
    `"repository" <https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset>`_.

    Example::
        >>> # NOTE: You need to download/generate the dataset from the repository.
        >>> from doctr.datasets import IMGUR5K
        >>> train_set = IMGUR5K(train=True, img_folder="/path/to/IMGUR5K-Handwriting-Dataset/images",
        >>>                     label_path="/path/to/IMGUR5K-Handwriting-Dataset/dataset_info/imgur5k_annotations.json")
        >>> img, target = train_set[0]
        >>> test_set = IMGUR5K(train=False, img_folder="/path/to/IMGUR5K-Handwriting-Dataset/images",
        >>>                    label_path="/path/to/IMGUR5K-Handwriting-Dataset/dataset_info/imgur5k_annotations.json")
        >>> img, target = test_set[0]
    Args:
        img_folder: folder with all the images of the dataset
        label_path: path to the annotations file of the dataset
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        train: bool = True,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, pre_transforms=convert_target_to_relative, **kwargs)

        # File existence check
        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_path if not os.path.exists(label_path) else img_folder}")

        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        self.train = train
        np_dtype = np.float32

        img_names = os.listdir(img_folder)
        train_samples = int(len(img_names) * 0.9)
        set_slice = slice(train_samples) if self.train else slice(train_samples, None)

        with open(label_path) as f:
            annotation_file = json.load(f)

        for img_name in img_names[set_slice]:
            img_path = Path(img_folder, img_name)
            img_id = img_name.split(".")[0]

            # File existence check
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")

            # some files have no annotations
            if img_id not in annotation_file['index_to_ann_map'].keys():
                continue
            ann_ids = annotation_file['index_to_ann_map'][img_id]
            annotations = [annotation_file['ann_id'][a_id] for a_id in ann_ids]

            labels: List[str] = []
            box_targets: List[np.ndarray] = []
            for ann in annotations:
                if ann['word'] != '.':
                    labels.append(ann['word'])
                    # x_center, y_center, width, height, angle
                    _box = [float(val) for val in ann['bounding_box'].strip('[ ]').split(', ')]
                    # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                    box_targets.append(cv2.boxPoints(((_box[0], _box[1]), (_box[2], _box[3]), _box[4])))

            if not use_polygons:
                # xmin, ymin, xmax, ymax
                box_targets = [np.array([min(box_target[:, 0]), min(box_target[:, 1]), max(
                    box_target[:, 0]), max(box_target[:, 1])], dtype=np_dtype) for box_target in box_targets]

            # filter images without boxes
            if len(box_targets) > 0:
                self.data.append((img_path, dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=labels)))

    def extra_repr(self) -> str:
        return f"train={self.train}"
