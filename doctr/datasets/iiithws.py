# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from random import sample
from typing import Any, List, Tuple

from tqdm import tqdm

from .datasets import AbstractDataset

__all__ = ["IIITHWS"]


class IIITHWS(AbstractDataset):
    """IIITHWS dataset from `"Generating Synthetic Data for Text Recognition"
    <https://arxiv.org/pdf/1608.04224.pdf>`_ | `"repository" <https://github.com/kris314/hwnet>`_ |
    `"website" <https://cvit.iiit.ac.in/research/projects/cvit-projects/matchdocimgs>`_.

    >>> # NOTE: This is a pure recognition dataset without bounding box labels.
    >>> # NOTE: You need to download the dataset.
    >>> from doctr.datasets import IIITHWS
    >>> train_set = IIITHWS(img_folder="/path/to/iiit-hws/Images_90K_Normalized",
    >>>                     label_path="/path/to/IIIT-HWS-90K.txt",
    >>>                     train=True)
    >>> img, target = train_set[0]
    >>> test_set = IIITHWS(img_folder="/path/to/iiit-hws/Images_90K_Normalized",
    >>>                    label_path="/path/to/IIIT-HWS-90K.txt")
    >>>                    train=False)
    >>> img, target = test_set[0]

    Args:
    ----
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
            raise FileNotFoundError(f"unable to locate {label_path if not os.path.exists(label_path) else img_folder}")

        self.data: List[Tuple[str, str]] = []
        self.train = train

        with open(label_path) as f:
            annotations = f.readlines()

        # Shuffle the dataset otherwise the test set will contain the same labels n times
        annotations = sample(annotations, len(annotations))
        train_samples = int(len(annotations) * 0.9)
        set_slice = slice(train_samples) if self.train else slice(train_samples, None)

        for annotation in tqdm(
            iterable=annotations[set_slice], desc="Unpacking IIITHWS", total=len(annotations[set_slice])
        ):
            img_path, label = annotation.split()[0:2]
            img_path = os.path.join(img_folder, img_path)

            self.data.append((img_path, label))

    def extra_repr(self) -> str:
        return f"train={self.train}"
