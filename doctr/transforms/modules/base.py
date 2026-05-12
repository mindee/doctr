# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np

from doctr.utils.repr import NestedObject

from .. import functional as F

__all__ = ["SampleCompose", "ImageTransform", "ColorInversion", "OneOf", "RandomApply", "RandomRotate", "RandomCrop"]


class SampleCompose(NestedObject):
    """Implements a wrapper that will apply transformations sequentially on both image and target

    .. code:: python

        >>> import numpy as np
        >>> import torch
        >>> from doctr.transforms import SampleCompose, ImageTransform, ColorInversion, RandomRotate
        >>> transfos = SampleCompose([ImageTransform(ColorInversion((32, 32))), RandomRotate(30)])
        >>> out, out_boxes = transfos(torch.rand(8, 64, 64, 3), np.zeros((2, 4)))

    Args:
        transforms: list of transformation modules
    """

    _children_names: list[str] = ["sample_transforms"]

    def __init__(self, transforms: list[Callable[[Any, Any], tuple[Any, Any]]]) -> None:
        self.sample_transforms = transforms

    def __call__(self, x: Any, target: Any) -> tuple[Any, Any]:
        for t in self.sample_transforms:
            x, target = t(x, target)

        return x, target


class ImageTransform(NestedObject):
    """Implements a transform wrapper to turn an image-only transformation into an image+target transform

    .. code:: python

        >>> import torch
        >>> from doctr.transforms import ImageTransform, ColorInversion
        >>> transfo = ImageTransform(ColorInversion((32, 32)))
        >>> out, _ = transfo(torch.rand(8, 64, 64, 3), None)

    Args:
        transform: the image transformation module to wrap
    """

    _children_names: list[str] = ["img_transform"]

    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.img_transform = transform

    def __call__(self, img: Any, target: Any) -> tuple[Any, Any]:
        img = self.img_transform(img)
        return img, target


class ColorInversion(NestedObject):
    """Applies the following tranformation to a tensor (image or batch of images):
    convert to grayscale, colorize (shift 0-values randomly), and then invert colors

    .. code:: python

        >>> import torch
        >>> from doctr.transforms import ColorInversion
        >>> transfo = ColorInversion(min_val=0.6)
        >>> out = transfo(torch.rand(8, 64, 64, 3))

    Args:
        min_val: range [min_val, 1] to colorize RGB pixels
    """

    def __init__(self, min_val: float = 0.5) -> None:
        self.min_val = min_val

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}"

    def __call__(self, img: Any) -> Any:
        return F.invert_colors(img, self.min_val)


class OneOf(NestedObject):
    """Randomly apply one of the input transformations

    .. code:: python

        >>> import torch
        >>> from doctr.transforms import OneOf
        >>> transfo = OneOf([JpegQuality(), Gamma()])
        >>> out = transfo(torch.rand(1, 64, 64, 3))

    Args:
        transforms: list of transformations, one only will be picked
    """

    _children_names: list[str] = ["transforms"]

    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(
        self, img: Any, target: np.ndarray | dict[str, np.ndarray] | None = None
    ) -> Any | tuple[Any, np.ndarray | dict[str, np.ndarray]]:
        # Pick transformation
        transfo = self.transforms[int(random.random() * len(self.transforms))]
        # Apply
        return transfo(img) if target is None else transfo(img, target)  # type: ignore[call-arg]


class RandomApply(NestedObject):
    """Apply with a probability p the input transformation

    .. code:: python

        >>> import torch
        >>> from doctr.transforms import RandomApply
        >>> transfo = RandomApply(Gamma(), p=.5)
        >>> out = transfo(torch.rand(1, 64, 64, 3))

    Args:
        transform: transformation to apply
        p: probability to apply
    """

    def __init__(self, transform: Callable[[Any], Any], p: float = 0.5) -> None:
        self.transform = transform
        self.p = p

    def extra_repr(self) -> str:
        return f"transform={self.transform}, p={self.p}"

    def __call__(
        self,
        img: Any,
        target: np.ndarray | dict[str, np.ndarray] | None = None,
    ) -> Any | tuple[Any, np.ndarray | dict[str, np.ndarray]]:
        if random.random() < self.p:
            return self.transform(img) if target is None else self.transform(img, target)  # type: ignore[call-arg]
        return img if target is None else (img, target)


class RandomRotate(NestedObject):
    """Randomly rotate a tensor image and its boxes

    .. image:: https://doctr-static.mindee.com/models?id=v0.4.0/rotation_illustration.png&src=0
        :align: center

    Args:
        max_angle: maximum angle for rotation, in degrees. Angles will be uniformly picked in [-max_angle, max_angle]
        expand: whether the image should be padded before the rotation
    """

    def __init__(self, max_angle: float = 5.0, expand: bool = False) -> None:
        self.max_angle = max_angle
        self.expand = expand

    def extra_repr(self) -> str:
        return f"max_angle={self.max_angle}, expand={self.expand}"

    def _rotate_array(self, img: Any, target: np.ndarray, angle: float) -> tuple[Any, np.ndarray]:
        """Rotate the image and the target, and keep only boxes with at least partial visibility after rotation"""
        is_polygon = target.shape[1:] == (4, 2)

        r_img, r_polys = F.rotate_sample(img, target, angle, self.expand)

        is_kept = (r_polys.max(1) > r_polys.min(1)).sum(1) == 2
        r_polys = r_polys[is_kept]

        # convert back if input was boxes
        if not is_polygon:
            # (N, 4, 2) -> (N, 4)
            x1y1 = r_polys.min(axis=1)
            x2y2 = r_polys.max(axis=1)
            r_boxes = np.concatenate([x1y1, x2y2], axis=1)
            return r_img, r_boxes

        return r_img, r_polys

    def __call__(
        self, img: Any, target: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[Any, np.ndarray | dict[str, np.ndarray]]:
        angle = random.uniform(-self.max_angle, self.max_angle)

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
            return rotated_img if rotated_img is not None else img, rotated_targets

        return self._rotate_array(img, target, angle)


class RandomCrop(NestedObject):
    """Randomly crop a tensor image and its boxes

    Args:
        scale: tuple of floats, relative (min_area, max_area) of the crop
        ratio: tuple of float, relative (min_ratio, max_ratio) where ratio = h/w
    """

    def __init__(self, scale: tuple[float, float] = (0.08, 1.0), ratio: tuple[float, float] = (0.75, 1.33)) -> None:
        self.scale = scale
        self.ratio = ratio

    def extra_repr(self) -> str:
        return f"scale={self.scale}, ratio={self.ratio}"

    def _crop_array(
        self,
        img: Any,
        target: np.ndarray,
        crop_box: tuple[float, float, float, float],
    ) -> tuple[Any, np.ndarray]:
        is_polygon = target.shape[1:] == (4, 2)
        # For polygons, we need to reproject the coordinates into the cropped frame,
        # and keep only those with at least partial visibility
        if is_polygon:
            cropped_img, _ = F.crop_detection(
                img,
                np.concatenate(
                    (
                        np.min(target, axis=1),
                        np.max(target, axis=1),
                    ),
                    axis=1,
                ),
                crop_box,
            )

            cropped_polys = target.copy()

            crop_w = crop_box[2] - crop_box[0]
            crop_h = crop_box[3] - crop_box[1]

            # Reproject coordinates into cropped frame
            cropped_polys[..., 0] = (cropped_polys[..., 0] - crop_box[0]) / crop_w
            cropped_polys[..., 1] = (cropped_polys[..., 1] - crop_box[1]) / crop_h

            # Keep polygons with at least partial visibility
            poly_min = np.min(cropped_polys, axis=1)
            poly_max = np.max(cropped_polys, axis=1)
            is_kept = (poly_max[:, 0] > 0) & (poly_min[:, 0] < 1) & (poly_max[:, 1] > 0) & (poly_min[:, 1] < 1)
            cropped_polys = cropped_polys[is_kept]

            if cropped_polys.shape[0] == 0:
                return img, target

            return cropped_img, np.clip(cropped_polys, 0, 1)

        # For detection boxes, we can directly crop and clip them
        cropped_img, crop_boxes = F.crop_detection(img, target, crop_box)

        if crop_boxes.shape[0] == 0:
            return img, target

        return cropped_img, np.clip(crop_boxes, 0, 1)

    def __call__(
        self,
        img: Any,
        target: np.ndarray | dict[str, np.ndarray],
    ) -> tuple[Any, np.ndarray | dict[str, np.ndarray]]:
        scale = random.uniform(self.scale[0], self.scale[1])
        ratio = random.uniform(self.ratio[0], self.ratio[1])

        height, width = img.shape[-2:]

        # Calculate crop size
        crop_area = scale * width * height
        aspect_ratio = ratio * (width / height)
        crop_width = int(round(math.sqrt(crop_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(crop_area / aspect_ratio)))

        # Ensure crop size does not exceed image dimensions
        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)

        # Randomly select crop position
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        crop_box = (
            x / width,
            y / height,
            (x + crop_width) / width,
            (y + crop_height) / height,
        )

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

            return cropped_img if cropped_img is not None else img, cropped_targets

        return self._crop_array(img, target, crop_box)
