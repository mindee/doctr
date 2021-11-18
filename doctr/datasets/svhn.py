# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import h5py
from tqdm import tqdm

from .datasets import VisionDataset

__all__ = ['SVHN']


class SVHN(VisionDataset):
    """SVHN dataset from `"The Street View House Numbers (SVHN) Dataset"
    <http://ufldl.stanford.edu/housenumbers/>`_.

    Example::
        >>> from doctr.datasets import SVHN
        >>> train_set = SVHN(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """
    TRAIN = ('http://ufldl.stanford.edu/housenumbers/train.tar.gz',
             '4b17bb33b6cd8f963493168f80143da956f28ec406cc12f8e5745a9f91a51898',
             'svhn_train.tar')

    TEST = ('http://ufldl.stanford.edu/housenumbers/test.tar.gz',
            '57ac9ceb530e4aa85b55d991be8fc49c695b3d71c6f6a88afea86549efde7fb5',
            'svhn_test.tar')

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        url, sha256, name = self.TRAIN if train else self.TEST
        super().__init__(url=url, file_name=name, file_hash=sha256, extract_archive=True, **kwargs)
        self.sample_transforms = sample_transforms
        self.train = train
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float32

        tmp_root = os.path.join(self.root, 'train' if train else 'test')

        # Load mat data (matlab v7.3 - can not be loaded with scipy)
        with h5py.File(os.path.join(tmp_root, 'digitStruct.mat'), 'r') as f:
            names = f['digitStruct/name']
            for idx in tqdm(iterable=range(len(names)), desc='Load SVHN', total=len(names)):
                img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))

                # File existence check
                if not os.path.exists(os.path.join(tmp_root, img_name)):
                    raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_name)}")

                x, y, width, height, labels = _get_coords(f, idx)
                label_targets = [str(label) for label in labels]

                if rotated_bbox:
                    # x_center, y_center, w, h, alpha = 0
                    box_targets = [[x + width / 2, y + height / 2, width, height, 0]
                                   for x, y, width, height in zip(x, y, width, height)]
                else:
                    # x, y, width, height -> xmin, ymin, xmax, ymax
                    box_targets = [[x, y, x + width, y + height] for x, y, width, height in zip(x, y, width, height)]
                self.data.append((img_name, dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=label_targets)))

        print(len(self.data))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"


def _get_coords(f, idx: int) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    """Get coordinates of the bounding boxes and the labels of the image with index idx."""
    meta: Dict[str, List[int]] = {key: [] for key in ['height', 'left', 'top', 'width', 'label']}

    bboxes = f['digitStruct/bbox']
    box = f[bboxes[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))
    return (meta['left'], meta['top'], meta['width'], meta['height'], meta['label'])
