# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image
from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from ..functional.pytorch import random_shadow

__all__ = ["Resize", "GaussianNoise", "ChannelShuffle", "RandomHorizontalFlip", "RandomShadow"]


class Resize(T.Resize):
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError("size should be either a tuple, a list or an int")

    def forward(
        self,
        img: torch.Tensor,
        target: Optional[np.ndarray] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        if isinstance(self.size, int):
            target_ratio = img.shape[-2] / img.shape[-1]
        else:
            target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]

        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio and (isinstance(self.size, (tuple, list)))):
            # If we don't preserve the aspect ratio or the wanted aspect ratio is the same than the original one
            # We can use with the regular resize
            if target is not None:
                return super().forward(img), target
            return super().forward(img)
        else:
            # Resize
            if isinstance(self.size, (tuple, list)):
                if actual_ratio > target_ratio:
                    tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
                else:
                    tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])
            elif isinstance(self.size, int):  # self.size is the longest side, infer the other
                if img.shape[-2] <= img.shape[-1]:
                    tmp_size = (max(int(self.size * actual_ratio), 1), self.size)
                else:
                    tmp_size = (self.size, max(int(self.size / actual_ratio), 1))

            # Scale image
            img = F.resize(img, tmp_size, self.interpolation)
            raw_shape = img.shape[-2:]
            if isinstance(self.size, (tuple, list)):
                # Pad (inverted in pytorch)
                _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
                if self.symmetric_pad:
                    half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                    _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
                img = pad(img, _pad)

            # In case boxes are provided, resize boxes if needed (for detection task if preserve aspect ratio)
            if target is not None:
                if self.preserve_aspect_ratio:
                    # Get absolute coords
                    if target.shape[1:] == (4,):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            if np.max(target) <= 1:
                                offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]
                            target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[:, [0, 2]] *= raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] *= raw_shape[-2] / img.shape[-2]
                    elif target.shape[1:] == (4, 2):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            if np.max(target) <= 1:
                                offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]
                            target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / img.shape[-1]
                            target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[..., 0] *= raw_shape[-1] / img.shape[-1]
                            target[..., 1] *= raw_shape[-2] / img.shape[-2]
                    else:
                        raise AssertionError
                return img, target

            return img

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"


class GaussianNoise(torch.nn.Module):
    """Adds Gaussian Noise to the input tensor

    >>> import torch
    >>> from doctr.transforms import GaussianNoise
    >>> transfo = GaussianNoise(0., 1.)
    >>> out = transfo(torch.rand((3, 224, 224)))

    Args:
        mean : mean of the gaussian distribution
        std : std of the gaussian distribution
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
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
        self, img: Union[torch.Tensor, Image], target: Dict[str, Any]
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


class RandomShadow(torch.nn.Module):
    """Adds random shade to the input image

    >>> import torch
    >>> from doctr.transforms import RandomShadow
    >>> transfo = RandomShadow((0., 1.))
    >>> out = transfo(torch.rand((3, 64, 64)))

    Args:
        opacity_range : minimum and maximum opacity of the shade
    """

    def __init__(self, opacity_range: Optional[Tuple[float, float]] = None) -> None:
        super().__init__()
        self.opacity_range = opacity_range if isinstance(opacity_range, tuple) else (0.2, 0.8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the distribution
        try:
            if x.dtype == torch.uint8:
                return (
                    (
                        255
                        * random_shadow(
                            x.to(dtype=torch.float32) / 255,
                            self.opacity_range,
                        )
                    )
                    .round()
                    .clip(0, 255)
                    .to(dtype=torch.uint8)
                )
            else:
                return random_shadow(x, self.opacity_range).clip(0, 1)
        except ValueError:
            return x

    def extra_repr(self) -> str:
        return f"opacity_range={self.opacity_range}"
