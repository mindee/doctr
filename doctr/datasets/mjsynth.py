# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from .datasets import AbstractDataset

__all__ = ["MJSynth"]


class MJSynth(AbstractDataset):
    """MJSynth dataset from `"Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition"
    <https://www.robots.ox.ac.uk/~vgg/data/text/>`_.

    >>> # NOTE: This is a pure recognition dataset without bounding box labels.
    >>> # NOTE: You need to download the dataset.
    >>> from doctr.datasets import MJSynth
    >>> train_set = MJSynth(img_folder="/path/to/mjsynth/mnt/ramdisk/max/90kDICT32px",
    >>>                     label_path="/path/to/mjsynth/mnt/ramdisk/max/90kDICT32px/imlist.txt",
    >>>                     train=True)
    >>> img, target = train_set[0]
    >>> test_set = MJSynth(img_folder="/path/to/mjsynth/mnt/ramdisk/max/90kDICT32px",
    >>>                    label_path="/path/to/mjsynth/mnt/ramdisk/max/90kDICT32px/imlist.txt")
    >>>                    train=False)
    >>> img, target = test_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_path: path to the file with the labels
        train: whether the subset should be the training one
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        train: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)

        # File existence check
        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_path if not os.path.exists(label_path) else img_folder}")

        self.data: List[Tuple[str, Dict[str, Any]]] = []
        self.train = train

        with open(label_path) as f:
            img_paths = f.readlines()

        train_samples = int(len(img_paths) * 0.9)
        set_slice = slice(train_samples) if self.train else slice(train_samples, None)

        for path in tqdm(iterable=img_paths[set_slice], desc='Unpacking MJSynth', total=len(img_paths[set_slice])):
            label = [path.split('_')[1]]
            img_path = os.path.join(img_folder, path[2:]).strip()

            self.data.append((img_path, dict(labels=label)))

    def extra_repr(self) -> str:
        return f"train={self.train}"
