# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math

import numpy as np
import torch
from PIL.Image import Image
from scipy.ndimage import gaussian_filter
from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from ..functional.pytorch import random_shadow

__all__ = [
    "Resize",
    "GaussianNoise",
    "ChannelShuffle",
    "RandomHorizontalFlip",
    "RandomShadow",
    "RandomResize",
    "GaussianBlur",
]


class Resize(T.Resize):
    """Resize the input image to the given size"""

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation, antialias=True)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError("size should be either a tuple, a list or an int")

    def forward(
        self,
        img: torch.Tensor,
        target: np.ndarray | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
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
            img = F.resize(img, tmp_size, self.interpolation, antialias=True)
            raw_shape = img.shape[-2:]
            if isinstance(self.size, (tuple, list)):
                # Pad (inverted in pytorch)
                _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
                if self.symmetric_pad:
                    half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                    _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
                # Pad image
                img = pad(img, _pad)

            # In case boxes are provided, resize boxes if needed (for detection task if preserve aspect ratio)
            if target is not None:
                if self.symmetric_pad:
                    offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]

                if self.preserve_aspect_ratio:
                    # Get absolute coords
                    if target.shape[1:] == (4,):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[:, [0, 2]] *= raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] *= raw_shape[-2] / img.shape[-2]
                    elif target.shape[1:] == (4, 2):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / img.shape[-1]
                            target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[..., 0] *= raw_shape[-1] / img.shape[-1]
                            target[..., 1] *= raw_shape[-2] / img.shape[-2]
                    else:
                        raise AssertionError("Boxes should be in the format (n_boxes, 4, 2) or (n_boxes, 4)")

                return img, np.clip(target, 0, 1)

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
            return (x + 255 * noise).round().clamp(0, 255).to(dtype=torch.uint8)  # type: ignore[attr-defined]
        else:
            return (x + noise.to(dtype=x.dtype)).clamp(0, 1)  # type: ignore[attr-defined]

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class GaussianBlur(torch.nn.Module):
    """Apply Gaussian Blur to the input tensor

    >>> import torch
    >>> from doctr.transforms import GaussianBlur
    >>> transfo = GaussianBlur(sigma=(0.0, 1.0))

    Args:
        sigma : standard deviation range for the gaussian kernel
    """

    def __init__(self, sigma: tuple[float, float]) -> None:
        super().__init__()
        self.sigma_range = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample a random sigma value within the specified range
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()

        # Apply Gaussian blur along spatial dimensions only
        blurred = torch.tensor(
            gaussian_filter(
                x.numpy(),
                sigma=sigma,
                mode="reflect",
                truncate=4.0,
            ),
            dtype=x.dtype,
            device=x.device,
        )
        return blurred


class ChannelShuffle(torch.nn.Module):
    """Randomly shuffle channel order of a given image"""

    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Get a random order
        chan_order = torch.rand(img.shape[0]).argsort()
        return img[chan_order]


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """Randomly flip the input image horizontally"""

    def forward(self, img: torch.Tensor | Image, target: np.ndarray) -> tuple[torch.Tensor | Image, np.ndarray]:
        if torch.rand(1) < self.p:
            _img = F.hflip(img)
            _target = target.copy()
            # Changing the relative bbox coordinates
            if target.shape[1:] == (4,):
                _target[:, ::2] = 1 - target[:, [2, 0]]
            else:
                _target[..., 0] = 1 - target[..., 0]
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

    def __init__(self, opacity_range: tuple[float, float] | None = None) -> None:
        super().__init__()
        self.opacity_range = opacity_range if isinstance(opacity_range, tuple) else (0.2, 0.8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the distribution
        try:
            if x.dtype == torch.uint8:
                return (
                    (  # type: ignore[attr-defined]
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


class RandomResize(torch.nn.Module):
    """Randomly resize the input image and align corresponding targets

    >>> import torch
    >>> from doctr.transforms import RandomResize
    >>> transfo = RandomResize((0.3, 0.9), preserve_aspect_ratio=True, symmetric_pad=True, p=0.5)
    >>> out = transfo(torch.rand((3, 64, 64)))

    Args:
        scale_range: range of the resizing factor for width and height (independently)
        preserve_aspect_ratio: whether to preserve the aspect ratio of the image,
        given a float value, the aspect ratio will be preserved with this probability
        symmetric_pad: whether to symmetrically pad the image,
        given a float value, the symmetric padding will be applied with this probability
        p: probability to apply the transformation
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.3, 0.9),
        preserve_aspect_ratio: bool | float = False,
        symmetric_pad: bool | float = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale_range = scale_range
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.p = p
        self._resize = Resize

    def forward(self, img: torch.Tensor, target: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        if torch.rand(1) < self.p:
            scale_h = np.random.uniform(*self.scale_range)
            scale_w = np.random.uniform(*self.scale_range)
            new_size = (int(img.shape[-2] * scale_h), int(img.shape[-1] * scale_w))

            _img, _target = self._resize(
                new_size,
                preserve_aspect_ratio=self.preserve_aspect_ratio
                if isinstance(self.preserve_aspect_ratio, bool)
                else bool(torch.rand(1) <= self.symmetric_pad),
                symmetric_pad=self.symmetric_pad
                if isinstance(self.symmetric_pad, bool)
                else bool(torch.rand(1) <= self.symmetric_pad),
            )(img, target)

            return _img, _target
        return img, target

    def extra_repr(self) -> str:
        return f"scale_range={self.scale_range}, preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}, p={self.p}"  # noqa: E501
