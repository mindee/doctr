# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import torch
from PIL.Image import Image
from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

__all__ = ['Resize', 'GaussianNoise', 'ChannelShuffle', 'RandomHorizontalFlip', 'RandomPerspective']


class Resize(T.Resize):
    def __init__(
        self,
        size: Tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]
        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio):
            return super().forward(img)
        else:
            # Resize
            if actual_ratio > target_ratio:
                tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
            else:
                tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])

            # Scale image
            img = F.resize(img, tmp_size, self.interpolation)
            # Pad (inverted in pytorch)
            _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
            if self.symmetric_pad:
                half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
            return pad(img, _pad)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"


class GaussianNoise(torch.nn.Module):
    """Adds Gaussian Noise to the input tensor

       Example::
           >>> from doctr.transforms import GaussianNoise
           >>> import torch
           >>> transfo = GaussianNoise(0., 1.)
           >>> out = transfo(torch.rand((3, 224, 224)))

       Args:
           mean : mean of the gaussian distribution
           std : std of the gaussian distribution
       """
    def __init__(self, mean: float = 0., std: float = 1.) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the distribution
        noise = self.mean + 2 * self.std * torch.rand(x.shape, device=x.device) - self.std
        if x.dtype == torch.uint8:
            return (x + 255 * noise).round().clamp(0, 255).to(dtype=torch.uint8)
        else:
            return (x + noise.to(dtype=x.dtype)).clamp(0, 1)

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class ChannelShuffle(torch.nn.Module):
    """Randomly shuffle channel order of a given image"""

    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Get a random order
        chan_order = torch.rand(img.shape[0]).argsort()
        return img[chan_order]


class RandomHorizontalFlip(T.RandomHorizontalFlip):

    def forward(
            self,
            img: Union[torch.Tensor, Image],
            target: Dict[str, Any]
    ) -> Tuple[Union[torch.Tensor, Image], Dict[str, Any]]:
        """
        Args:
            img: Image to be flipped.
            target: Dictionary with boxes (in relative coordinates of shape (N, 4)) and labels as keys
        Returns:
            Tuple of PIL Image or Tensor and target
        """
        if torch.rand(1) < self.p:
            _img = F.hflip(img)
            _target = target.copy()
            # Changing the relative bbox coordinates
            _target["boxes"][:, ::2] = 1 - target["boxes"][:, [2, 0]]
            return _img, _target
        return img, target


class RandomPerspective(T.RandomPerspective):

    def forward(
            self, img: Union[torch.Tensor, Image],
            target: Dict[str, Any]
    ) -> Tuple[Union[torch.Tensor, Image], Dict[str, Any]]:
        """
          Args:
            img: Image to be transformed.
            target: Dictionary with boxes (in relative coordinates of shape (N, 4)) and labels as keys
        Returns:
            Tuple of PIL Image or Tensor and target
         """

        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * self._get_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        if torch.rand(1) < self.p:
            cmap = [(0, 0, 0), (230, 230, 230), (195, 195, 195), (175, 175, 175), (250, 250, 250)]
            bbox = []
            label = []

            #  Preparing mask
            mask = np.zeros_like(img.permute(1, 2, 0))

            if isinstance(img, torch.Tensor):
                height, width = img.shape[-2:]
            elif isinstance(img, Image.Image):
                width, height = img.size

            #  Drawing bounding boxes on mask with respective color
            for ind, val in enumerate(target["boxes"]):
                cv2.rectangle(
                    mask, (int(val[0] * width), int(val[1] * height)),
                    (int(val[2] * width), int(val[3] * height)),
                    cmap[target["labels"][ind]], -1
                )
            mask = torch.from_numpy(mask).permute(2, 0, 1)
            ddic = {}
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)

            #  transformed mask
            transformed_mask = F.perspective(
                mask, startpoints, endpoints, self.interpolation, fill
            ).permute(1, 2, 0).numpy()

            new_img = F.perspective(img, startpoints, endpoints, self.interpolation, fill)

            #  filtering to separate bboxes of each label
            mask_qr = cv2.inRange(transformed_mask, np.array([220, 220, 220]), np.array([240, 240, 240]))
            mask_bar = cv2.inRange(transformed_mask, np.array([180, 180, 180]), np.array([200, 200, 200]))
            mask_logo = cv2.inRange(transformed_mask, np.array([160, 160, 160]), np.array([180, 180, 180]))
            mask_photo = cv2.inRange(transformed_mask, np.array([240, 240, 240]), np.array([255, 255, 255]))

            #  finding contours of the masks of each label
            contours_photo, _ = cv2.findContours(mask_photo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_bar, _ = cv2.findContours(mask_bar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_logo, _ = cv2.findContours(mask_logo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_qr, _ = cv2.findContours(mask_qr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_name = [0, contours_qr, contours_bar, contours_logo, contours_photo]

            #  new target
            for lab in range(1, 5):
                for i in range(np.count_nonzero(target["labels"] == lab)):
                    try:
                        bbox.append([
                            min(contour_name[lab][i][..., 0]) / width,
                            min(contour_name[lab][i][..., 1]) / height,
                            max(contour_name[lab][i][..., 0]) / width,
                            max(contour_name[lab][i][..., 1]) / height
                        ])
                        label.append(lab)
                    except IndexError:
                        pass
            ddic.update({"boxes": np.array(bbox, dtype=np.float32).reshape(-1, 4), "labels": np.array(label)})
            return new_img, ddic
        return img, target

    @staticmethod
    def _get_num_channels(img):
        if img.ndim == 2:
            return 1
        elif img.ndim > 2:
            return img.shape[-3]
