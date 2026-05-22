# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np

from doctr.utils.common_types import Sample
from doctr.utils.repr import NestedObject

from .. import functional as F

__all__ = [
    "SampleCompose",
    "ImageTransform",
    "ColorInversion",
    "OneOf",
    "RandomApply",
    "RandomRotate",
    "RandomCrop",
    "ImageTorchvisionTransform",
]


class SampleCompose(NestedObject):
    """Implements a wrapper that will apply transformations sequentially on both image and target

    >>> import numpy as np
    >>> import torch
    >>> from doctr.transforms import SampleCompose, ImageTransform, ColorInversion, RandomRotate
    >>> from doctr.utils import Sample
    >>> transfos = SampleCompose([ImageTransform(ColorInversion((32, 32))), RandomRotate(30)])
    >>> out, out_boxes = transfos(Sample(image=torch.rand(8, 64, 64, 3), target=np.zeros((2, 4))))

    Args:
        transforms: list of transformation modules
    """

    _children_names: list[str] = ["sample_transforms"]

    def __init__(self, transforms: list[Callable[[Sample], Sample]]) -> None:
        self.sample_transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.sample_transforms:
            sample = t(sample)
        return sample


class ImageTransform(NestedObject):
    """Implements a transform wrapper to turn an image-only transformation into an image+target transform

    >>> import torch
    >>> from doctr.transforms import ImageTransform, ColorInversion
    >>> from doctr.utils import Sample
    >>> transfo = ImageTransform(ColorInversion((32, 32)))
    >>> out = transfo(Sample(image=torch.rand(8, 64, 64, 3)))

    Args:
        transform: the image transformation module to wrap
    """

    _children_names: list[str] = ["img_transform"]

    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.img_transform = transform

    def __call__(self, sample: Sample) -> Sample:
        img = self.img_transform(sample)
        return sample.replace(image=img)


class ImageTorchvisionTransform(NestedObject):
    """Implements a transform wrapper to turn a torchvision image-only transformation into an image+target transform

    >>> import torch
    >>> from torchvision import transforms
    >>> from doctr.transforms import ImageTorchvisionTransform
    >>> from doctr.utils import Sample
    >>> transfo = ImageTorchvisionTransform(transforms.ColorJitter(brightness=0.5))
    >>> out, _ = transfo(Sample(image=torch.rand(8, 64, 64, 3)))

    Args:
        transform: the torchvision image transformation module to wrap
    """

    _children_names: list[str] = ["img_transform"]

    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.img_transform = transform

    def __call__(self, sample: Sample) -> Sample:
        img = self.img_transform(sample.image)
        return sample.replace(image=img)


class ColorInversion(NestedObject):
    """Applies the following tranformation to a tensor (image or batch of images):
    convert to grayscale, colorize (shift 0-values randomly), and then invert colors

    >>> import torch
    >>> from doctr.transforms import ColorInversion
    >>> from doctr.utils import Sample
    >>> transfo = ColorInversion(min_val=0.6)
    >>> out = transfo(Sample(image=torch.rand(8, 64, 64, 3)))

    Args:
        min_val: range [min_val, 1] to colorize RGB pixels
    """

    def __init__(self, min_val: float = 0.5) -> None:
        self.min_val = min_val

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}"

    def __call__(self, sample: Sample) -> Sample:
        out = F.invert_colors(sample.image, self.min_val)
        return sample.replace(image=out)


class OneOf(NestedObject):
    """Randomly apply one of the input transformations

    >>> import torch
    >>> from doctr.transforms import OneOf, JpegQuality, Gamma
    >>> from doctr.utils import Sample
    >>> transfo = OneOf([JpegQuality(), Gamma()])
    >>> out = transfo(Sample(image=torch.rand(1, 64, 64, 3)))

    Args:
        transforms: list of transformations, one only will be picked
    """

    _children_names: list[str] = ["transforms"]

    def __init__(self, transforms: list[Callable[[Sample], Sample]]) -> None:
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        transfo = self.transforms[int(random.random() * len(self.transforms))]
        return transfo(sample)


class RandomApply(NestedObject):
    """Apply with a probability p the input transformation

    >>> import torch
    >>> from doctr.transforms import RandomApply, Gamma
    >>> from doctr.utils import Sample
    >>> transfo = RandomApply(Gamma(), p=.5)
    >>> out = transfo(Sample(image=torch.rand(1, 64, 64, 3), target=np.array([[0.1, 0.1, 0.9, 0.9]]), mask=None))

    Args:
        transform: transformation to apply
        p: probability to apply
    """

    def __init__(self, transform: Callable[[Sample], Sample], p: float = 0.5) -> None:
        self.transform = transform
        self.p = p

    def extra_repr(self) -> str:
        return f"transform={self.transform}, p={self.p}"

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            return self.transform(sample)
        return sample


class RandomRotate(NestedObject):
    """Randomly rotate a tensor image and its boxes

    .. image:: https://doctr-static.mindee.com/models?id=v0.4.0/rotation_illustration.png&src=0
        :align: center

    >>> import torch
    >>> from doctr.transforms import RandomRotate
    >>> transfo = RandomRotate(max_angle=30, expand=True)
    >>> out = transfo(Sample(image=torch.rand(1, 64, 64, 3), target=np.array([[0.1, 0.1, 0.9, 0.9]]), mask=None))

    Args:
        max_angle: maximum angle for rotation, in degrees. Angles will be uniformly picked in [-max_angle, max_angle]
        expand: whether the image should be padded before the rotation
    """

    def __init__(self, max_angle: float = 5.0, expand: bool = False) -> None:
        self.max_angle = max_angle
        self.expand = expand

    def extra_repr(self) -> str:
        return f"max_angle={self.max_angle}, expand={self.expand}"

    def _rotate_array(self, img: Any, target: np.ndarray, angle: float):
        is_polygon = target.shape[1:] == (4, 2)

        r_img, r_polys = F.rotate_sample(img, target, angle, self.expand)

        is_kept = (r_polys.max(1) > r_polys.min(1)).sum(1) == 2
        r_polys = r_polys[is_kept]

        # convert back if input was boxes
        if not is_polygon:
            x1y1 = r_polys.min(axis=1)
            x2y2 = r_polys.max(axis=1)
            return r_img, np.concatenate([x1y1, x2y2], axis=1)

        return r_img, r_polys

    def __call__(self, sample: Sample) -> Sample:
        angle = random.uniform(-self.max_angle, self.max_angle)

        img = sample.image
        target = sample.target
        mask = sample.mask

        r_mask = None
        if mask is not None:
            r_mask, _ = F.rotate_sample(
                mask.unsqueeze(0), np.array([[0, 0, 1, 1]], dtype=np.float32), angle, self.expand
            )

        if target is None:
            r_img, _ = F.rotate_sample(img, np.array([[0, 0, 1, 1]], dtype=np.float32), angle, self.expand)
            return sample.replace(image=r_img, mask=r_mask)

        if isinstance(target, dict):
            rotated_targets = {}
            rotated_img = None

            for cls_name, arr in target.items():
                if len(arr) == 0:
                    rotated_targets[cls_name] = arr.copy()
                    continue

                r_img, r_arr = self._rotate_array(img, arr, angle)
                if rotated_img is None:
                    rotated_img = r_img
                rotated_targets[cls_name] = r_arr

            final_img = rotated_img if rotated_img is not None else img
            return sample.replace(image=final_img, mask=r_mask, target=rotated_targets)

        r_img, r_target = self._rotate_array(img, target, angle)
        return sample.replace(image=r_img, mask=r_mask, target=r_target)


class RandomCrop(NestedObject):
    """Randomly crop a tensor image and its boxes

    >>> import torch
    >>> from doctr.transforms import RandomCrop
    >>> from doctr.utils import Sample
    >>> transfo = RandomCrop(scale=(0.5, 1.0), ratio=(0.75, 1.33))
    >>> out = transfo(Sample(image=torch.rand(1, 64, 64, 3), target=np.array([[0.1, 0.1, 0.9, 0.9]]), mask=None))

    Args:
        scale: tuple of floats, relative (min_area, max_area) of the crop
        ratio: tuple of float, relative (min_ratio, max_ratio) where ratio = h/w
    """

    def __init__(self, scale: tuple[float, float] = (0.08, 1.0), ratio: tuple[float, float] = (0.75, 1.33)) -> None:
        self.scale = scale
        self.ratio = ratio

    def extra_repr(self) -> str:
        return f"scale={self.scale}, ratio={self.ratio}"

    def _crop_array(self, img: Any, target: np.ndarray, crop_box):
        is_polygon = target.shape[1:] == (4, 2)

        if is_polygon:
            cropped_img, _ = F.crop_detection(
                img,
                np.concatenate((np.min(target, axis=1), np.max(target, axis=1)), axis=1),
                crop_box,
            )

            cropped_polys = target.copy()

            # pixel-space crop box for coordinate transform
            x0, y0, x1, y1 = (
                int(crop_box[0] * img.shape[-1]),
                int(crop_box[1] * img.shape[-2]),
                int(crop_box[2] * img.shape[-1]),
                int(crop_box[3] * img.shape[-2]),
            )

            crop_w = x1 - x0
            crop_h = y1 - y0

            # shift polygons into cropped pixel frame
            cropped_polys[..., 0] -= x0
            cropped_polys[..., 1] -= y0

            # visibility check in pixel space
            poly_min = np.min(cropped_polys, axis=1)
            poly_max = np.max(cropped_polys, axis=1)

            is_kept = (
                (poly_max[:, 0] > 0) & (poly_min[:, 0] < crop_w) & (poly_max[:, 1] > 0) & (poly_min[:, 1] < crop_h)
            )

            cropped_polys = cropped_polys[is_kept]

            if cropped_polys.shape[0] == 0:
                return img, target

            # final clipping in pixel space
            cropped_polys[..., 0] = np.clip(cropped_polys[..., 0], 0, crop_w)
            cropped_polys[..., 1] = np.clip(cropped_polys[..., 1], 0, crop_h)

            return cropped_img, cropped_polys

    def __call__(self, sample: Sample) -> Sample:
        scale = random.uniform(self.scale[0], self.scale[1])
        ratio = random.uniform(self.ratio[0], self.ratio[1])

        img = sample.image
        target = sample.target
        mask = sample.mask

        h, w = img.shape[-2:]

        crop_area = scale * w * h
        aspect_ratio = ratio * (w / h)

        crop_w = int(round(math.sqrt(crop_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(crop_area / aspect_ratio)))

        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)

        crop_box = (
            x / w,
            y / h,
            (x + crop_w) / w,
            (y + crop_h) / h,
        )

        r_mask = None
        if mask is not None:
            r_mask, _ = self._crop_array(mask, np.zeros((0, 4)), crop_box)

        if target is None:
            r_img, _ = self._crop_array(img, np.zeros((0, 4)), crop_box)
            return sample.replace(image=r_img, mask=r_mask)

        if isinstance(target, dict):
            cropped_targets = {}
            cropped_img = None

            for cls_name, arr in target.items():
                if len(arr) == 0:
                    cropped_targets[cls_name] = arr.copy()
                    continue

                c_img, c_arr = self._crop_array(img, arr, crop_box)

                if cropped_img is None:
                    cropped_img = c_img

                cropped_targets[cls_name] = c_arr

            final_img = cropped_img if cropped_img is not None else img
            return sample.replace(image=final_img, mask=r_mask, target=cropped_targets)

        c_img, c_target = self._crop_array(img, target, crop_box)
        return sample.replace(image=c_img, mask=r_mask, target=c_target)
