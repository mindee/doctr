# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        annotation_file = json.load(open(label_path))
        train_samples = int(len(img_names) * 0.9)
        set_slice = slice(train_samples) if self.train else slice(train_samples, None)

        for img_name in img_names[set_slice]:
            img_path = Path(img_folder, img_name)

            # File existence check
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")

            try:  # some files have no annotations
                ann_ids = annotation_file['index_to_ann_map'][img_name.split('.')[0]]
            except KeyError:
                continue
            annotations = [annotation_file['ann_id'][a_id] for a_id in ann_ids]
            labels = []
            _bboxes = []
            for ann in annotations:
                if ann['word'] != '.':
                    labels.append(ann['word'])
                    # x_center, y_center, width, height, angle
                    _bboxes.append([float(val) for val in ann['bounding_box'].strip('[ ]').split(', ')])

            # TODO: rework !!!
            box_targets = []
            if use_polygons:
                # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                for box in _bboxes:
                    xc, yc, w, h, a = box
                    c = math.cos(box[4])
                    s = math.sin(box[4])
                    r1x = -w / 2 * c - h / 2 * s
                    r1y = -w / 2 * s + h / 2 * c
                    r2x = w / 2 * c - h / 2 * s
                    r2y = w / 2 * s + h / 2 * c

                    box_targets.append([[box[0] + r1x, box[1] + r1y],
                                        [box[0] + r2x, box[1] + r2y],
                                        [box[0] - r2x, box[1] - r2y],
                                        [box[0] - r1x, box[1] - r1y]])

            else:
                # xmin, ymin, xmax, ymax
                for box in _bboxes:
                    xc, yc, w, h, a = box
                    new_w = w * math.cos(a * (math.pi / 180)) + h * math.sin(a * (math.pi / 180))
                    new_h = h * math.cos(a * (math.pi / 180)) + w * math.sin(a * (math.pi / 180))

                    x_min = float(math.floor(xc - new_w / 2))
                    y_min = float(math.floor(yc - new_h / 2))
                    x_max = float(math.floor(x_min + new_w))
                    y_max = float(math.floor(y_min + new_h))

                    box_targets.append([x_min, y_min, x_max, y_max])  # type: ignore[list-item]

            #import cv2
            #for box in box_targets:
            #    img = cv2.imread(str(img_path))
            #    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            #    cv2.imshow('test', img)
            #    cv2.waitKey(0)

            self.data.append((img_path, dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=labels)))

    def extra_repr(self) -> str:
        return f"train={self.train}"
