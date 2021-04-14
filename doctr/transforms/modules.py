# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from typing import List, Any, Tuple, Callable

from doctr.utils.repr import NestedObject


__all__ = ['Compose', 'Resize', 'Normalize', 'LambdaTransformation']


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
        preserve_aspect_ratio: if `True`, preserve aspect ratio and pad the rest with zeros
    """
    def __init__(
        self,
        output_size: Tuple[int, int],
        method: str = 'bilinear',
        preserve_aspect_ratio: bool = False,
    ) -> None:
        self.output_size = output_size
        self.method = method
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}, method='{self.method}'"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        img = tf.image.resize(img, self.output_size, self.method, self.preserve_aspect_ratio)
        if self.preserve_aspect_ratio:
            img = tf.image.pad_to_bounding_box(img, 0, 0, *self.output_size)
        return img


class Normalize(NestedObject):
    """Normalize a tensor to a Gaussian distribution for each channel

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        mean: average value per channel
        std: standard deviation per channel
    """
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)

    def extra_repr(self) -> str:
        return f"mean={self.mean.numpy().tolist()}, std={self.std.numpy().tolist()}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        img -= self.mean
        img /= self.std
        return img


class LambdaTransformation(NestedObject):
    """Normalize a tensor to a Gaussian distribution for each channel

    Example::
        >>> from doctr.transforms import LambdaTransformation
        >>> import tensorflow as tf
        >>> transfo = LambdaTransformation(lambda x: x/ 255.)
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        fn: the function to be applied to the input tensor
    """
    def __init__(self, fn: Callable[[tf.Tensor], tf.Tensor]) -> None:
        self.fn = fn

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return self.fn(img)
