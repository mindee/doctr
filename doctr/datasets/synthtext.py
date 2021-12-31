# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .datasets import VisionDataset
from .utils import convert_target_to_relative

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
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip'
    SHA256 = '28ab030485ec8df3ed612c568dd71fb2793b9afbfa3a9d9c6e792aef33265bf1'

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            self.URL,
            None,
            file_hash=None,
            extract_archive=True,
            pre_transforms=convert_target_to_relative,
            **kwargs
        )
        self.train = train

        # Load mat data
        tmp_root = os.path.join(self.root, 'SynthText') if self.SHA256 else self.root
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

            labels = ''.join(txt).split()
            word_boxes = word_boxes.transpose(2, 1, 0) if word_boxes.ndim == 3 else np.expand_dims(word_boxes, axis=0)

            if not use_polygons:
                word_boxes = np.concatenate((word_boxes.min(axis=1), word_boxes.max(axis=1)), axis=1)

            self.data.append((img_path[0], dict(boxes=np.asarray(word_boxes, dtype=np_dtype), labels=labels)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
