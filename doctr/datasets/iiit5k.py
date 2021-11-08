# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import scipy.io as sio

from .datasets import VisionDataset

__all__ = ['IIIT5K']


class IIIT5K(VisionDataset):
    """IIIT-5K dataset from `"BMVC 2012 Scene Text Recognition using Higher Order Language Priors"
    <https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/home/mishraBMVC12.pdf>`_.

    Example::
        >>> from doctr.datasets import IIIT5K
        >>> train_set = IIIT5K(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        **kwargs: keyword arguments from `VisionDataset`.
    """

    # TODO: add dataset
    TRAIN = ('https://github.com/mindee/doctr/releases/download/v0.4.1/IIIT-5K-train-reco.zip',
             'd4fa9e60abb03500d83299c845b9c87fd9c9430d1aeac96b83c5d0bb0ab27f6f')
    TEST = ('https://github.com/mindee/doctr/releases/download/v0.4.1/IIIT-5K-test-reco.zip',
            '41b3c746a20226fddc80d86d4b2a903d43b5be4f521dd1bbe759dbf8844745e2')

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> None:

        url, sha256 = self.TRAIN if train else self.TEST
        super().__init__(url, None, sha256, True, **kwargs)
        self.sample_transforms = sample_transforms
        self.train = train

        # Load mat data
        mat_file = 'traindata' if self.train else 'testdata'
        mat_data = sio.loadmat(os.path.join(self.root, f'{mat_file}.mat'))[mat_file][0]
        tmp_root = os.path.join(self.root, 'train') if train else os.path.join(self.root, 'test')
        img, label = zip(*[(x[0][0], x[1][0]) for x in mat_data])

        self.data: List[Tuple[Path, str]] = []

        for img_path, label in zip(img, label):
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path)}")
            self.data.append((img_path, label))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
