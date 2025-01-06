# Copyright (C) 2021-2025, Mindee.

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

    .. tabs::

        .. tab:: PyTorch

            .. code:: python

                >>> import numpy as np
                >>> import torch
                >>> from doctr.transforms import SampleCompose, ImageTransform, ColorInversion, RandomRotate
                >>> transfos = SampleCompose([ImageTransform(ColorInversion((32, 32))), RandomRotate(30)])
                >>> out, out_boxes = transfos(torch.rand(8, 64, 64, 3), np.zeros((2, 4)))

        .. tab:: TensorFlow

            .. code:: python

                >>> import numpy as np
                >>> import tensorflow as tf
                >>> from doctr.transforms import SampleCompose, ImageTransform, ColorInversion, RandomRotate
                >>> transfo = SampleCompose([ImageTransform(ColorInversion((32, 32))), RandomRotate(30)])
                >>> out, out_boxes = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1), np.zeros((2, 4)))

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

    .. tabs::

        .. tab:: PyTorch

            .. code:: python

                >>> import torch
                >>> from doctr.transforms import ImageTransform, ColorInversion
                >>> transfo = ImageTransform(ColorInversion((32, 32)))
                >>> out, _ = transfo(torch.rand(8, 64, 64, 3), None)

        .. tab:: TensorFlow

            .. code:: python

                >>> import tensorflow as tf
                >>> from doctr.transforms import ImageTransform, ColorInversion
                >>> transfo = ImageTransform(ColorInversion((32, 32)))
                >>> out, _ = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1), None)

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

    .. tabs::

        .. tab:: PyTorch

            .. code:: python

                >>> import torch
                >>> from doctr.transforms import ColorInversion
                >>> transfo = ColorInversion(min_val=0.6)
                >>> out = transfo(torch.rand(8, 64, 64, 3))

        .. tab:: TensorFlow

            .. code:: python

                >>> import tensorflow as tf
                >>> from doctr.transforms import ColorInversion
                >>> transfo = ColorInversion(min_val=0.6)
                >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

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

    .. tabs::

        .. tab:: PyTorch

            .. code:: python

                >>> import torch
                >>> from doctr.transforms import OneOf
                >>> transfo = OneOf([JpegQuality(), Gamma()])
                >>> out = transfo(torch.rand(1, 64, 64, 3))

        .. tab:: TensorFlow

            .. code:: python

                >>> import tensorflow as tf
                >>> from doctr.transforms import OneOf
                >>> transfo = OneOf([JpegQuality(), Gamma()])
                >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transforms: list of transformations, one only will be picked
    """

    _children_names: list[str] = ["transforms"]

    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, img: Any, target: np.ndarray | None = None) -> Any | tuple[Any, np.ndarray]:
        # Pick transformation
        transfo = self.transforms[int(random.random() * len(self.transforms))]
        # Apply
        return transfo(img) if target is None else transfo(img, target)  # type: ignore[call-arg]


class RandomApply(NestedObject):
    """Apply with a probability p the input transformation

    .. tabs::

        .. tab:: PyTorch

            .. code:: python

                >>> import torch
                >>> from doctr.transforms import RandomApply
                >>> transfo = RandomApply(Gamma(), p=.5)
                >>> out = transfo(torch.rand(1, 64, 64, 3))

        .. tab:: TensorFlow

            .. code:: python

                >>> import tensorflow as tf
                >>> from doctr.transforms import RandomApply
                >>> transfo = RandomApply(Gamma(), p=.5)
                >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transform: transformation to apply
        p: probability to apply
    """

    def __init__(self, transform: Callable[[Any], Any], p: float = 0.5) -> None:
        self.transform = transform
        self.p = p

    def extra_repr(self) -> str:
        return f"transform={self.transform}, p={self.p}"

    def __call__(self, img: Any, target: np.ndarray | None = None) -> Any | tuple[Any, np.ndarray]:
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

    def __call__(self, img: Any, target: np.ndarray) -> tuple[Any, np.ndarray]:
        angle = random.uniform(-self.max_angle, self.max_angle)
        r_img, r_polys = F.rotate_sample(img, target, angle, self.expand)
        # Removes deleted boxes
        is_kept = (r_polys.max(1) > r_polys.min(1)).sum(1) == 2
        return r_img, r_polys[is_kept]


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

    def __call__(self, img: Any, target: np.ndarray) -> tuple[Any, np.ndarray]:
        scale = random.uniform(self.scale[0], self.scale[1])
        ratio = random.uniform(self.ratio[0], self.ratio[1])

        height, width = img.shape[:2]

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

        # relative crop box
        crop_box = (x / width, y / height, (x + crop_width) / width, (y + crop_height) / height)
        if target.shape[1:] == (4, 2):
            min_xy = np.min(target, axis=1)
            max_xy = np.max(target, axis=1)
            _target = np.concatenate((min_xy, max_xy), axis=1)
        else:
            _target = target

        # Crop image and targets
        croped_img, crop_boxes = F.crop_detection(img, _target, crop_box)
        # hard fallback if no box is kept
        if crop_boxes.shape[0] == 0:
            return img, target
        # clip boxes
        return croped_img, np.clip(crop_boxes, 0, 1)
