# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

from .datasets import VisionDataset

__all__ = ['FUNSD']


class FUNSD(VisionDataset):
    """FUNSD dataset from `"FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents"
    <https://arxiv.org/pdf/1905.13538.pdf>`_.

    Example::
        >>> from doctr.datasets import FUNSD
        >>> train_set = FUNSD(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'https://guillaumejaume.github.io/FUNSD/dataset.zip'
    SHA256 = 'c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f'
    FILE_NAME = 'funsd.zip'

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(self.URL, self.FILE_NAME, self.SHA256, True, **kwargs)
        self.train = train
        self.sample_transforms = sample_transforms

        # Use the subset
        subfolder = os.path.join('dataset', 'training_data' if train else 'testing_data')

        # # List images
        tmp_root = os.path.join(self.root, subfolder, 'images')
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        for img_path in os.listdir(tmp_root):
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path)}")
            stem = Path(img_path).stem
            with open(os.path.join(self.root, subfolder, 'annotations', f"{stem}.json"), 'rb') as f:
                data = json.load(f)

            _targets = [(word['text'], word['box']) for block in data['form']
                        for word in block['words'] if len(word['text']) > 0]
            text_targets, box_targets = zip(*_targets)
            if rotated_bbox:
                # box_targets: xmin, ymin, xmax, ymax -> x, y, w, h, alpha = 0
                box_targets = [
                    [
                        (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1], 0
                    ] for box in box_targets
                ]

            self.data.append((img_path, dict(boxes=np.asarray(box_targets, dtype=int), labels=text_targets)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
