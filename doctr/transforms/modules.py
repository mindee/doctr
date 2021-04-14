# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from typing import List, Any, Tuple

from doctr.utils.repr import NestedObject


__all__ = ['Compose', 'Resize']


class Compose(NestedObject):
    """Implements a wrapper that will apply transformations sequentially

    Example::
        >>> from doctr.transforms import Compose, Resize
        >>> import tensorflow as tf
        >>> transfos = Compose([Resize((32, 32))])
        >>> out = transfos(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transforms: list of transformation modules
    """

    _children_names: List[str] = ['transforms']

    def __init__(self, transforms: List[NestedObject]) -> None:
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)

        return x


class Resize(NestedObject):
    """Resizes a tensor to a target size

    Example::
        >>> from doctr.transforms import Resize
        >>> import tensorflow as tf
        >>> transfo = Resize((32, 32))
        >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        output_size: expected output size
        method: interpolation method
    """
    def __init__(self, output_size: Tuple[int, int], method: str = 'bilinear') -> None:
        self.output_size = output_size
        self.method = method

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}, method='{self.method}'"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        img = tf.image.resize(img, self.output_size, method=self.method)
        return img