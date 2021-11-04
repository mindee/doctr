# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from .datasets import VisionDataset


__all__ = ['DocArtefacts']


class DocArtefacts(VisionDataset):
    """Dataset containing ....

    Example::
        >>> from doctr.datasets import DocArtefacts
        >>> train_set = DocArtefacts(download=True)
        >>> img, target = train_set[0]

    Args:
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'https://github.com/mindee/doctr/releases/download/v0.4.0/artefact_detection-6c401d4d.zip'
    SHA256 = '6c401d4d5d4ebaf086c3ed81a7d8142f48161420ab693bf8ac384e413a9c7d19'
    FILE_NAME = 'artefact_detection-6c401d4d.zip'

    def __init__(
        self,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(self.URL, self.FILE_NAME, self.SHA256, True, **kwargs)
        self.sample_transforms = sample_transforms

        # List images
        tmp_root = os.path.join(self.root, 'images')
        with open(os.path.join(self.root, "labels.json"), "rb") as f:
            labels = json.load(f)
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        img_list = os.listdir(tmp_root)
        if len(labels) != len(img_list):
            raise AssertionError('the number of images and labels do not match')
        for img_name, label in labels.items():
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_name)}")
            boxes = np.asarray([obj['geometry'] for obj in label], dtype=np.float32)
            classes = [obj['label'] for obj in label]
            if rotated_bbox:
                # box_targets: xmin, ymin, xmax, ymax -> x, y, w, h, alpha = 0
                boxes = np.stack((
                    boxes[:, [0, 2]].mean(dim=1),
                    boxes[:, [1, 3]].mean(dim=1),
                    boxes[:, 2] - boxes[:, 0],
                    boxes[:, 3] - boxes[:, 1],
                    0,
                ), axis=1)
            self.data.append((img_name, dict(boxes=boxes, labels=classes)))
        self.root = tmp_root
