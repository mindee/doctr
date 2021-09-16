# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import random
import math
from typing import List, Any, Callable, Dict, Tuple
import numpy as np

from doctr.utils.repr import NestedObject
from .. import functional as F


__all__ = ['ColorInversion', 'OneOf', 'RandomApply', 'RandomRotate', 'RandomCrop']


class ColorInversion(NestedObject):
    """Applies the following tranformation to a tensor (image or batch of images):
    convert to grayscale, colorize (shift 0-values randomly), and then invert colors

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
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

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = OneOf([JpegQuality(), Gamma()])
        >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transforms: list of transformations, one only will be picked
    """

    _children_names: List[str] = ['transforms']

    def __init__(self, transforms: List[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, img: Any) -> Any:
        # Pick transformation
        transfo = self.transforms[int(random.random() * len(self.transforms))]
        # Apply
        return transfo(img)


class RandomApply(NestedObject):
    """Apply with a probability p the input transformation

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = RandomApply(Gamma(), p=.5)
        >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transform: transformation to apply
        p: probability to apply
    """
    def __init__(self, transform: Callable[[Any], Any], p: float = .5) -> None:
        self.transform = transform
        self.p = p

    def extra_repr(self) -> str:
        return f"transform={self.transform}, p={self.p}"

    def __call__(self, img: Any) -> Any:
        if random.random() < self.p:
            return self.transform(img)
        return img


class RandomRotate(NestedObject):
    """Randomly rotate a tensor image and its boxes

    Args:
        max_angle: maximum angle for rotation, in degrees. Angles will be uniformly picked in
            [-max_angle, max_angle]
        expand: whether the image should be padded before the rotation
    """
    def __init__(self, max_angle: float = 5., expand: bool = False) -> None:
        self.max_angle = max_angle
        self.expand = expand

    def extra_repr(self) -> str:
        return f"max_angle={self.max_angle}, expand={self.expand}"

    def __call__(self, img: Any, target: Dict[str, np.ndarray]) -> Tuple[Any, Dict[str, np.ndarray]]:
        angle = random.uniform(-self.max_angle, self.max_angle)
        r_img, r_boxes = F.rotate(img, target["boxes"], angle, self.expand)
        return r_img, dict(boxes=r_boxes)


class RandomCrop(NestedObject):
    """Randomly crop a tensor image and its boxes

    Args:
        scale: tuple of floats, relative (min_area, max_area) of the crop
        ratio: tuple of float, relative (min_ratio, max_ratio) where ratio = h/w
    """
    def __init__(self, scale: Tuple[float, float] = (0.08, 1.), ratio: Tuple[float, float] = (0.75, 1.33)) -> None:
        self.scale = scale
        self.ratio = ratio

    def extra_repr(self) -> str:
        return f"scale={self.scale}, ratio={self.ratio}"

    def __call__(self, img: Any, target: Dict[str, np.ndarray]) -> Tuple[Any, Dict[str, np.ndarray]]:
        h, w = img.shape[:2]
        scale = random.uniform(self.scale[0], self.scale[1])
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        crop_h = math.sqrt(scale * ratio)
        crop_w = math.sqrt(scale / ratio)
        start_x, start_y = random.uniform(0, 1 - crop_w), random.uniform(0, 1 - crop_h)
        crop_box = (
            max(0, int(round(start_x * w))),
            max(0, int(round(start_y * h))),
            min(int(round((start_x + crop_w) * w)), w - 1),
            min(int(round((start_y + crop_h) * h)), h - 1)
        )
        croped_img, crop_boxes = F.crop_detection(img, target["boxes"], crop_box)
        return croped_img, dict(boxes=crop_boxes)
