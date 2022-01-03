# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.io as sio

from .datasets import VisionDataset
from .utils import convert_target_to_relative

__all__ = ['IIIT5K']


class IIIT5K(VisionDataset):
    """IIIT-5K character-level localization dataset from
    `"BMVC 2012 Scene Text Recognition using Higher Order Language Priors"
    <https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/home/mishraBMVC12.pdf>`_.

    Example::
        >>> # NOTE: this dataset is for character-level localization
        >>> from doctr.datasets import IIIT5K
        >>> train_set = IIIT5K(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz'
    SHA256 = '7872c9efbec457eb23f3368855e7738f72ce10927f52a382deb4966ca0ffa38e'

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            self.URL,
            None,
            file_hash=self.SHA256,
            extract_archive=True,
            pre_transforms=convert_target_to_relative,
            **kwargs
        )
        self.train = train

        # Load mat data
        tmp_root = os.path.join(self.root, 'IIIT5K') if self.SHA256 else self.root
        mat_file = 'trainCharBound' if self.train else 'testCharBound'
        mat_data = sio.loadmat(os.path.join(tmp_root, f'{mat_file}.mat'))[mat_file][0]

        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        np_dtype = np.float32

        for img_path, label, box_targets in mat_data:
            _raw_path = img_path[0]
            _raw_label = label[0]

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, _raw_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, _raw_path)}")

            if use_polygons:
                # x1, y1, x2, y2, x3, y3, x4, y4
                box_targets = [
                    [
                        [box[0], box[1]],
                        [box[0] + box[2], box[1]],
                        [box[0] + box[2], box[1] + box[3]],
                        [box[0], box[1] + box[3]],
                    ] for box in box_targets
                ]
            else:
                # xmin, ymin, xmax, ymax
                box_targets = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in box_targets]

            # label are casted to list where each char corresponds to the character's bounding box
            self.data.append((_raw_path, dict(boxes=np.asarray(
                box_targets, dtype=np_dtype), labels=list(_raw_label))))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
