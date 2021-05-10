# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import tensorflow as tf

from .core import VisionDataset

__all__ = ['CORD']


class CORD(VisionDataset):
    """CORD dataset from `"CORD: A Consolidated Receipt Dataset forPost-OCR Parsing"
    <https://openreview.net/pdf?id=SJl3z659UH>`_.

    Example::
        >>> from doctr.datasets import CORD
        >>> train_set = CORD(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        **kwargs: keyword arguments from `VisionDataset`.
    """
    TRAIN = ('https://github.com/mindee/doctr/releases/download/v0.1.1/cord_train.zip',
             '45f9dc77f126490f3e52d7cb4f70ef3c57e649ea86d19d862a2757c9c455d7f8')

    TEST = ('https://github.com/mindee/doctr/releases/download/v0.1.1/cord_test.zip',
            '8c895e3d6f7e1161c5b7245e3723ce15c04d84be89eaa6093949b75a66fb3c58')

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        **kwargs: Any,
    ) -> None:

        url, sha256 = self.TRAIN if train else self.TEST
        super().__init__(url, None, sha256, True, **kwargs)

        # # List images
        self.root = os.path.join(self._root, 'image')
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        self.train = train
        self.sample_transforms = (lambda x: x) if sample_transforms is None else sample_transforms
        for img_path in os.listdir(self.root):
            stem = Path(img_path).stem
            _targets = []
            with open(os.path.join(self._root, 'json', f"{stem}.json"), 'rb') as f:
                label = json.load(f)
                for line in label["valid_line"]:
                    for word in line["words"]:
                        x = word["quad"]["x1"], word["quad"]["x2"], word["quad"]["x3"], word["quad"]["x4"]
                        y = word["quad"]["y1"], word["quad"]["y2"], word["quad"]["y3"], word["quad"]["y4"]
                        # Reduce 8 coords to 4
                        left, right = min(x), max(x)
                        top, bot = min(y), max(y)
                        if len(word["text"]) > 0:
                            _targets.append((word["text"], [left, top, right, bot]))

            text_targets, box_targets = zip(*_targets)

            self.data.append((img_path, dict(boxes=np.asarray(box_targets, dtype=np.int), labels=text_targets)))

    def extra_repr(self) -> str:
        return f"train={self.train}"

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
