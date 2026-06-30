# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from doctr.utils import Sample

from ..functional import random_shadow

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
    """Resize the input image to the given size

    >>> import torch
    >>> from doctr.transforms import Resize
    >>> from doctr.utils import Sample
    >>> transfo = Resize((64, 64), preserve_aspect_ratio=True, symmetric_pad=True)
    >>> out = transfo(Sample(image=torch.rand((3, 64, 64))))

    Args:
        size: output size in pixels, either a tuple (height, width) or a single integer for square images
        interpolation: interpolation mode to use for resizing, default is bilinear
        preserve_aspect_ratio: whether to preserve the aspect ratio of the image,
            if True, the image will be resized to fit within the target size while maintaining its aspect ratio
        symmetric_pad: whether to symmetrically pad the image to the target size,
            if True, the image will be padded equally on both sides to fit the target size
        return_padding_mask: whether to return a padding mask indicating the padded areas of the image
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
        return_padding_mask: bool = False,
    ) -> None:
        super().__init__(size if isinstance(size, (list, tuple)) else (size, size), interpolation, antialias=True)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.return_padding_mask = return_padding_mask

    def _resize_target(
        self,
        target: np.ndarray,
        raw_shape: Sequence[int],
        final_shape: Sequence[int],
        symmetric_pad: bool = False,
        offset: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Resize the target boxes according to the resizing of the image and the padding if needed"""
        target = target.copy()

        if target.shape[1:] == (4,):
            if symmetric_pad:
                target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / final_shape[-1]
                target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / final_shape[-2]
            else:
                target[:, [0, 2]] *= raw_shape[-1] / final_shape[-1]
                target[:, [1, 3]] *= raw_shape[-2] / final_shape[-2]

        elif target.shape[1:] == (4, 2):
            if symmetric_pad:
                target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / final_shape[-1]
                target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / final_shape[-2]
            else:
                target[..., 0] *= raw_shape[-1] / final_shape[-1]
                target[..., 1] *= raw_shape[-2] / final_shape[-2]

        else:
            raise AssertionError("Boxes should be in the format (n_boxes, 4, 2) or (n_boxes, 4)")

        return np.clip(target, 0, 1)

    def _resize_mask(self, mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.resize(
            mask,
            size,
            interpolation=F.InterpolationMode.NEAREST,
            antialias=False,
        ).squeeze(0)

    def _build_return_sample(
        self,
        sample: Sample,
        img: torch.Tensor,
        mask: torch.Tensor | None,
        target: np.ndarray | dict | str | None,
        padding_mask: torch.Tensor | None,
        resize_mask: bool,
    ) -> Sample:
        if target is not None:
            if self.return_padding_mask:
                return sample.replace(image=img, target=target, mask=mask if resize_mask else padding_mask)
            return sample.replace(image=img, target=target, mask=mask if resize_mask else sample.mask)
        if self.return_padding_mask:
            return sample.replace(image=img, mask=mask if resize_mask else padding_mask)
        return sample.replace(image=img, mask=mask if resize_mask else sample.mask)

    def _resize_targets(
        self,
        target: np.ndarray | dict | str | None,
        raw_shape: tuple[int, int],
        final_shape: tuple[int, int],
        half_pad: tuple[int, int] | None,
    ) -> np.ndarray | dict | str | None:
        if target is None:
            return target

        if self.symmetric_pad:
            offset = (
                half_pad[0] / final_shape[-1],
                half_pad[1] / final_shape[-2],
            )
        else:
            offset = (0, 0)

        if isinstance(target, str) or (isinstance(target, np.ndarray) and target.shape == (1,)):
            return target
        elif isinstance(target, dict):
            return {
                cls_name: self._resize_target(
                    arr,
                    raw_shape,
                    final_shape,
                    symmetric_pad=self.symmetric_pad,
                    offset=offset,
                )
                for cls_name, arr in target.items()
            }
        else:
            return self._resize_target(
                target,
                raw_shape,
                final_shape,
                symmetric_pad=self.symmetric_pad,
                offset=offset,
            )

    def _resize_preserve_aspect_ratio(
        self,
        sample: Sample,
        img: torch.Tensor,
        mask: torch.Tensor | None,
        target: np.ndarray | dict | str | None,
        resize_mask: bool,
    ) -> Sample:
        target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]

        if actual_ratio > target_ratio:
            tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
        else:
            tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])

        img = F.resize(img, tmp_size, self.interpolation, antialias=True)

        if resize_mask:
            mask = self._resize_mask(mask, tmp_size)

        raw_shape = img.shape[-2:]
        padding_mask = None
        half_pad = (0, 0)

        if isinstance(self.size, (tuple, list)):
            _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])

            if self.symmetric_pad:
                half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])

            img = pad(img, _pad)

            if resize_mask and mask is not None:
                mask = pad(mask, _pad)

            if self.return_padding_mask:
                h, w = self.size
                padding_mask = torch.zeros((h, w), dtype=torch.bool, device=img.device)
                left, right, top, bottom = _pad
                padding_mask[top : h - bottom, left : w - right] = True

        target = self._resize_targets(target, raw_shape, img.shape[-2:], half_pad)

        return self._build_return_sample(sample, img, mask, target, padding_mask, resize_mask)

    def forward(
        self,
        sample: Sample,
    ) -> Sample:
        img = sample.image
        target = sample.target
        mask = sample.mask

        resize_mask = mask is not None
        if resize_mask and mask.ndim == 2:
            mask = mask.unsqueeze(0)

        target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]

        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio):
            img = super().forward(img)

            if resize_mask:
                mask = self._resize_mask(mask, self.size)

            padding_mask = (
                torch.zeros(self.size, dtype=torch.bool, device=img.device)
                if self.return_padding_mask
                else None
            )

            return self._build_return_sample(sample, img, mask, target, padding_mask, resize_mask)

        return self._resize_preserve_aspect_ratio(sample, img, mask, target, resize_mask)

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
    >>> from doctr.utils import Sample
    >>> transfo = GaussianNoise(0., 1.)
    >>> out = transfo(Sample(image=torch.rand((3, 224, 224))))

    Args:
        mean : mean of the gaussian distribution
        std : std of the gaussian distribution
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, sample: Sample) -> Sample:
        x = sample.image
        noise = self.mean + 2 * self.std * torch.rand(x.shape, device=x.device) - self.std
        if x.dtype == torch.uint8:
            image = (x + 255 * noise).round().clamp(0, 255).to(dtype=torch.uint8)
        else:
            image = (x + noise.to(dtype=x.dtype)).clamp(0, 1)
        return sample.replace(image=image)

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class GaussianBlur(torch.nn.Module):
    """Apply Gaussian Blur to the input tensor

    >>> import torch
    >>> from doctr.transforms import GaussianBlur
    >>> from doctr.utils import Sample
    >>> transfo = GaussianBlur(sigma=(0.0, 1.0))
    >>> out = transfo(Sample(image=torch.rand((3, 224, 224))))

    Args:
        sigma : standard deviation range for the gaussian kernel
    """

    def __init__(self, sigma: tuple[float, float]) -> None:
        super().__init__()
        self.sigma_range = sigma

    def forward(self, sample: Sample) -> Sample:
        # Sample a random sigma value within the specified range
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()

        # Apply Gaussian blur along spatial dimensions only
        blurred = torch.tensor(
            gaussian_filter(
                sample.image.numpy(),
                sigma=sigma,
                mode="reflect",
                truncate=4.0,
            ),
            dtype=sample.image.dtype,
            device=sample.image.device,
        )
        return sample.replace(image=blurred)


class ChannelShuffle(torch.nn.Module):
    """Randomly shuffle channel order of a given image

    >>> import torch
    >>> from doctr.transforms import ChannelShuffle
    >>> from doctr.utils import Sample
    >>> transfo = ChannelShuffle()
    >>> out = transfo(Sample(image=torch.rand((3, 224, 224))))
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample: Sample) -> Sample:
        # Get a random order
        chan_order = torch.rand(sample.image.shape[0]).argsort()
        return sample.replace(image=sample.image[chan_order])


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """Randomly flip the input image horizontally

    >>> import torch
    >>> from doctr.transforms import RandomHorizontalFlip
    >>> from doctr.utils import Sample
    >>> transfo = RandomHorizontalFlip(p=1.0)
    >>> out = transfo(Sample(image=torch.rand((3, 224, 224)), target=np.array([[0.1, 0.2, 0.3, 0.4]])))
    """

    def _flip_array(self, target):
        _target = target.copy()
        # Changing the relative bbox coordinates
        if target.shape[1:] == (4,):
            _target[:, ::2] = 1 - target[:, [2, 0]]
        else:
            _target[..., 0] = 1 - target[..., 0]
        return _target

    def forward(self, sample: Sample) -> Sample:
        if torch.rand(1) < self.p:
            img = F.hflip(sample.image)
            mask = F.hflip(sample.mask) if sample.mask is not None else None

            target = sample.target
            if target is not None:
                if isinstance(target, dict):
                    target = {k: self._flip_array(v) for k, v in target.items()}
                else:
                    target = self._flip_array(target)

            return sample.replace(image=img, mask=mask, target=target)

        return sample


class RandomShadow(torch.nn.Module):
    """Adds random shade to the input image

    >>> import torch
    >>> from doctr.transforms import RandomShadow
    >>> from doctr.utils import Sample
    >>> transfo = RandomShadow((0., 1.))
    >>> out = transfo(Sample(image=torch.rand((3, 64, 64))))

    Args:
        opacity_range : minimum and maximum opacity of the shade
    """

    def __init__(self, opacity_range: tuple[float, float] | None = None) -> None:
        super().__init__()
        self.opacity_range = opacity_range if isinstance(opacity_range, tuple) else (0.2, 0.8)

    def __call__(self, sample: Sample) -> Sample:
        # Reshape the distribution
        try:
            if sample.image.dtype == torch.uint8:
                shadowed_image = (
                    (
                        255
                        * random_shadow(
                            sample.image.to(dtype=torch.float32) / 255,
                            self.opacity_range,
                        )
                    )
                    .round()
                    .clip(0, 255)
                    .to(dtype=torch.uint8)
                )
                return sample.replace(image=shadowed_image)
            else:
                shadowed_image = random_shadow(sample.image, self.opacity_range).clip(0, 1)
                return sample.replace(image=shadowed_image)
        except ValueError:  # pragma: no cover
            return sample

    def extra_repr(self) -> str:
        return f"opacity_range={self.opacity_range}"


class RandomResize(torch.nn.Module):
    """Randomly resize the input image and align corresponding targets

    >>> import torch
    >>> from doctr.transforms import RandomResize
    >>> from doctr.utils import Sample
    >>> transfo = RandomResize((0.3, 0.9), preserve_aspect_ratio=True, symmetric_pad=True, p=0.5)
    >>> out = transfo(Sample(image=torch.rand((3, 64, 64))))

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

    def forward(
        self,
        sample: Sample,
    ) -> Sample:
        if torch.rand(1) < self.p:
            scale_h = np.random.uniform(*self.scale_range)
            scale_w = np.random.uniform(*self.scale_range)
            new_size = (int(sample.image.shape[-2] * scale_h), int(sample.image.shape[-1] * scale_w))

            res = self._resize(
                new_size,
                preserve_aspect_ratio=self.preserve_aspect_ratio
                if isinstance(self.preserve_aspect_ratio, bool)
                else bool(torch.rand(1) <= self.preserve_aspect_ratio),
                symmetric_pad=self.symmetric_pad
                if isinstance(self.symmetric_pad, bool)
                else bool(torch.rand(1) <= self.symmetric_pad),
            )(sample)
            return res
        return sample

    def extra_repr(self) -> str:
        return f"scale_range={self.scale_range}, preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}, p={self.p}"  # noqa: E501
