# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from typing import Any

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .datasets import VisionDataset
from .utils import convert_target_to_relative

__all__ = ["IIIT5K"]


class IIIT5K(VisionDataset):
    """IIIT-5K character-level localization dataset from
    `"BMVC 2012 Scene Text Recognition using Higher Order Language Priors"
    <https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/home/mishraBMVC12.pdf>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/iiit5k-grid.png&src=0
        :align: center

    >>> # NOTE: this dataset is for character-level localization
    >>> from doctr.datasets import IIIT5K
    >>> train_set = IIIT5K(train=True, download=True)
    >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        detection_task: whether the dataset should be used for detection task
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = "https://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz"
    SHA256 = "7872c9efbec457eb23f3368855e7738f72ce10927f52a382deb4966ca0ffa38e"

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        recognition_task: bool = False,
        detection_task: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            self.URL,
            None,
            file_hash=self.SHA256,
            extract_archive=True,
            pre_transforms=convert_target_to_relative if not recognition_task else None,
            **kwargs,
        )
        if recognition_task and detection_task:
            raise ValueError(
                "`recognition_task` and `detection_task` cannot be set to True simultaneously. "
                + "To get the whole dataset with boxes and labels leave both parameters to False."
            )

        self.train = train

        # Load mat data
        tmp_root = os.path.join(self.root, "IIIT5K") if self.SHA256 else self.root
        mat_file = "trainCharBound" if self.train else "testCharBound"
        mat_data = sio.loadmat(os.path.join(tmp_root, f"{mat_file}.mat"))[mat_file][0]

        self.data: list[tuple[str | np.ndarray, str | dict[str, Any] | np.ndarray]] = []
        np_dtype = np.float32

        for img_path, label, box_targets in tqdm(
            iterable=mat_data, desc="Preparing and Loading IIIT5K", total=len(mat_data)
        ):
            _raw_path = img_path[0]
            _raw_label = label[0]

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, _raw_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, _raw_path)}")

            if use_polygons:
                # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                box_targets = [
                    [
                        [box[0], box[1]],
                        [box[0] + box[2], box[1]],
                        [box[0] + box[2], box[1] + box[3]],
                        [box[0], box[1] + box[3]],
                    ]
                    for box in box_targets
                ]
            else:
                # xmin, ymin, xmax, ymax
                box_targets = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in box_targets]

            if recognition_task:
                self.data.append((_raw_path, _raw_label))
            elif detection_task:
                self.data.append((_raw_path, np.asarray(box_targets, dtype=np_dtype)))
            else:
                # label are casted to list where each char corresponds to the character's bounding box
                self.data.append((
                    _raw_path,
                    dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=list(_raw_label)),
                ))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
