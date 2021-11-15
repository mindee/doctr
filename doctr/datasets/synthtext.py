# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

from .datasets import VisionDataset

__all__ = ['SynthText']


class SynthText(VisionDataset):
    """SynthText dataset from `"Synthetic Data for Text Localisation in Natural Images"
    <https://arxiv.org/abs/1604.06646>`_.

    Example::
        >>> from doctr.datasets import SynthText
        >>> data_set = SynthText(download=True)
        >>> img, target = data_set[0]

    Args:
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip'
    SHA256 = '28ab030485ec8df3ed612c568dd71fb2793b9afbfa3a9d9c6e792aef33265bf1'

    def __init__(
        self,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(url=self.URL, file_name=None, file_hash=self.SHA256, extract_archive=True, **kwargs)
        self.sample_transforms = sample_transforms

        # Load mat data
        tmp_root = os.path.join(self.root, 'SynthText')
        mat_data = sio.loadmat(os.path.join(tmp_root, 'gt.mat'))

        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        np_dtype = np.float16 if self.fp16 else np.float32

        for img_path, boxes, labels in zip(mat_data['imnames'], mat_data['charBB'], mat_data['txt']):

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path)}")

            if rotated_bbox:
                pass
                # x_center, y_center, w, h, alpha = 0
                #box_targets = [[box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3], 0] for box in box_targets]
            else:
                pass
                # x, y, width, height -> xmin, ymin, xmax, ymax
                #box_targets = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in box_targets]

            # label are casted to list where each char corresponds to the character's bounding box
            #self.data.append((_raw_path, dict(boxes=np.asarray(
            #    box_targets, dtype=np_dtype), labels=list(_raw_label))))

        self.root = tmp_root
