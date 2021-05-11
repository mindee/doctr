# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import tensorflow as tf

from doctr.documents.reader import read_img, read_pdf
from .core import AbstractDataset


__all__ = ['OCRDataset']


class OCRDataset(AbstractDataset):
    """Implements an OCR dataset

    Args:
        img_folder: local path to image folder (all jpg at the root)
        label_file: local path to the label file
        sample_transforms: composable transformations that will be applied to each image
        **kwargs: keyword arguments from `VisionDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_file: str,
        sample_transforms: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        **kwargs: Any,
    ) -> None:

        self.sample_transforms = (lambda x: x) if sample_transforms is None else sample_transforms
        self.img_folder = img_folder

        # List images
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        with open(label_file, 'rb') as f:
            data = json.load(f)

        for file_dic in data:
            # Get image path
            img_name = Path(os.path.basename(file_dic["raw-archive-filepath"])).stem + '.jpg'
            box_targets = []
            for box in file_dic["coordinates"]:
                xs, ys = np.asarray(box)[:, 0], np.asarray(box)[:, 1]
                box_targets.append([min(xs), min(ys), max(xs), max(ys)])

            text_targets = file_dic["string"]
            self.data.append((img_name, dict(boxes=np.asarray(box_targets), labels=text_targets)))

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, Dict[str, Any]]:
        img_name, target = self.data[index]
        # Read image
        img = tf.io.read_file(os.path.join(self.img_folder, img_name))
        img = tf.image.decode_jpeg(img, channels=3)
        img = self.sample_transforms(img)

        return img, target

    @staticmethod
    def collate_fn(samples: List[Tuple[tf.Tensor, Dict[str, Any]]]) -> Tuple[tf.Tensor, List[Dict[str, Any]]]:

        images, targets = zip(*samples)
        images = tf.stack(images, axis=0)

        return images, list(targets)
