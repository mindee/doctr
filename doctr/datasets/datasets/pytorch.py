# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import List, Any, Tuple
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .base import _AbstractDataset, _VisionDataset


__all__ = ['AbstractDataset', 'VisionDataset']


class AbstractDataset(_AbstractDataset):

    @staticmethod
    def _get_img_shape(img: Any) -> Tuple[int, int]:
        return img.shape[-2:]

    def _read_sample(self, index: int) -> Tuple[torch.Tensor, Any]:
        img_name, target = self.data[index]
        # Read image
        img = to_tensor(Image.open(os.path.join(self.root, img_name), mode='r').convert('RGB'))

        return img, target

    @staticmethod
    def collate_fn(samples: List[Tuple[torch.Tensor, Any]]) -> Tuple[torch.Tensor, List[Any]]:

        images, targets = zip(*samples)
        images = torch.stack(images, dim=0)

        return images, list(targets)


class VisionDataset(AbstractDataset, _VisionDataset):
    pass
