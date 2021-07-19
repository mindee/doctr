# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import List, Any, Tuple
import numpy as np
from PIL import Image
import torch
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
        pil_img = Image.open(os.path.join(self.root, img_name), mode='r').convert('RGB')
        if self.fp16:
            img = torch.from_numpy(
                np.array(pil_img, np.uint8, copy=True)
            )
            img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
            # put it from HWC to CHW format
            img = img.permute((2, 0, 1)).contiguous()
            # Switch to FP16
            img = img.to(dtype=torch.float16).div(255)
        else:
            img = to_tensor(pil_img)

        return img, target

    @staticmethod
    def collate_fn(samples: List[Tuple[torch.Tensor, Any]]) -> Tuple[torch.Tensor, List[Any]]:

        images, targets = zip(*samples)
        images = torch.stack(images, dim=0)

        return images, list(targets)


class VisionDataset(AbstractDataset, _VisionDataset):
    pass
