# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
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

    DATA_URL = 'https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz'

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(url=self.DATA_URL, extract_archive=True, download=True, **kwargs)
        self.sample_transforms = sample_transforms
        self.train = train

        # Load mat data
        dataset = 'IIIT5K'
        tmp_root = os.path.join(self.root, dataset) if train else os.path.join(self.root, dataset)
        mat_file = 'trainCharBound' if self.train else 'testCharBound'
        mat_data = sio.loadmat(os.path.join(tmp_root, f'{mat_file}.mat'))[mat_file][0]

        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        np_dtype = np.float16 if self.fp16 else np.float32

        for img_path, label, box_targets in mat_data:
            _raw_path = img_path[0]
            _raw_label = label[0]

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, _raw_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, _raw_path)}")

            if rotated_bbox:
                # x, y, width, height -> x, y, width, height, alpha = 0
                box_targets = [[box[0], box[1], box[2], box[3], 0] for box in box_targets]
            else:
                # x, y, width, height -> xmin, ymin, xmax, ymax
                box_targets = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in box_targets]

            if len(_raw_label) != len(box_targets):
                raise ValueError(f"{_raw_label} and {box_targets} are not same length")

            self.data.append((_raw_path, dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=tuple(_raw_label))))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
