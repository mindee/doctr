# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from copy import deepcopy
from typing import Any, List, Tuple

import torch

from doctr.io import read_img_as_tensor

from .base import _AbstractDataset, _VisionDataset

__all__ = ['AbstractDataset', 'VisionDataset']


class AbstractDataset(_AbstractDataset):

    def _read_sample(self, index: int) -> Tuple[torch.Tensor, Any]:
        img_name, target = self.data[index]
        # Read image
        img = read_img_as_tensor(os.path.join(self.root, img_name), dtype=torch.float32)

        return img, deepcopy(target)

    @staticmethod
    def collate_fn(samples: List[Tuple[torch.Tensor, Any]]) -> Tuple[torch.Tensor, List[Any]]:

        images, targets = zip(*samples)
        images = torch.stack(images, dim=0)

        return images, list(targets)


class VisionDataset(AbstractDataset, _VisionDataset):
    pass
