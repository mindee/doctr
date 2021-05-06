# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
import tensorflow as tf

from doctr.documents.reader import read_img, read_pdf
from .core import AbstractDataset


__all__ = ['TESTSET']


class TESTSET(AbstractDataset):
    """Private TESTSET, dataset contains sensible data and cannot be shared. 

    Args:
        path: local path to the dataset folder. Folder must contain "labels" & "images" folder
        sample_transforms: composable transformations that will be applied to each image
        **kwargs: keyword arguments from `VisionDataset`.
    """

    def __init__(
        self,
        path: str,
        sample_transforms: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        **kwargs: Any,
    ) -> None:

        self.sample_transforms = (lambda x: x) if sample_transforms is None else sample_transforms   
        self.root = path

        # List images
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        with open(os.path.join(self.root, 'labels/typed_word/labels.json'), 'rb') as f:
                data = json.load(f)
    
        for file_dic in data:
            img_name = os.path.join('images', file_dic["raw-archive-filepath"])

            box_targets = []
            for box in file_dic["coordinates"]:
                xs, ys = np.asarray(box)[:, 0], np.asarray(box)[:, 1]
                box_targets.append([min(xs), min(ys), max(xs), max(ys)])

            text_targets = file_dic["string"]

            self.data.append((img_name, dict(boxes=np.asarray(box_targets), labels=text_targets)))

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, Dict[str, Any]]:
        img_name, target = self.data[index]
        # Read image
        img = tf.io.read_file(os.path.join(self.root, img_name))
        img = tf.image.decode_jpeg(img, channels=3)
        img = self.sample_transforms(img)

        return img, target

    @staticmethod
    def collate_fn(samples: List[Tuple[tf.Tensor, Dict[str, Any]]]) -> Tuple[tf.Tensor, List[Dict[str, Any]]]:

        images, targets = zip(*samples)
        images = tf.stack(images, axis=0)

        return images, list(targets)
