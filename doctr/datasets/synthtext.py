# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .datasets import VisionDataset

__all__ = ['SynthText']


class SynthText(VisionDataset):
    """SynthText dataset from `"Synthetic Data for Text Localisation in Natural Images"
    <https://arxiv.org/abs/1604.06646>`_ | `"repository" <https://github.com/ankush-me/SynthText>`_ |
    `"website" <https://www.robots.ox.ac.uk/~vgg/data/scenetext/>`_.

    Example::
        >>> from doctr.datasets import SynthText
        >>> train_set = SynthText(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip'
    SHA256 = '28ab030485ec8df3ed612c568dd71fb2793b9afbfa3a9d9c6e792aef33265bf1'

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(url=self.URL, file_name=None, file_hash=None, extract_archive=True, **kwargs)
        self.sample_transforms = sample_transforms
        self.train = train

        # Load mat data
        tmp_root = os.path.join(self.root, 'SynthText')
        mat_data = sio.loadmat(os.path.join(tmp_root, 'gt.mat'))
        split = int(len(mat_data['imnames'][0]) * 0.9)
        paths = mat_data['imnames'][0][slice(split) if self.train else slice(split, None)]
        boxes = mat_data['wordBB'][0][slice(split) if self.train else slice(split, None)]
        labels = mat_data['txt'][0][slice(split) if self.train else slice(split, None)]

        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float32

        for img_path, word_boxes, txt in tqdm(iterable=zip(paths, boxes, labels),
                                              desc='Unpacking SynthText', total=len(paths)):

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path[0])):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path[0])}")

            labels = [elt for word in txt.tolist() for elt in word.split()]
            word_boxes = word_boxes.transpose(2, 1, 0) if word_boxes.ndim == 3 else np.expand_dims(word_boxes, axis=0)

            if rotated_bbox:
                # x_center, y_center, w, h, alpha = 0
                mins = word_boxes.min(axis=1)
                maxs = word_boxes.max(axis=1)
                box_targets = np.concatenate(
                    ((mins + maxs) / 2, maxs - mins, np.zeros((word_boxes.shape[0], 1))), axis=1)
            else:
                # xmin, ymin, xmax, ymax
                box_targets = np.concatenate((word_boxes.min(axis=1), word_boxes.max(axis=1)), axis=1)

            self.data.append((img_path[0], dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=labels)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
