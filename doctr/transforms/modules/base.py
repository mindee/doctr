# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import random
from typing import List, Any, Callable

from doctr.utils.repr import NestedObject
from .. import functional as F


__all__ = ['ColorInversion', 'OneOf', 'RandomApply']


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
