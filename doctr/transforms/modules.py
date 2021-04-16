# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from typing import List, Any, Tuple, Callable

from doctr.utils.repr import NestedObject


__all__ = ['Compose', 'Resize', 'Normalize', 'LambdaTransformation', 'ToGray', 'InvertColorize',
           'Brightness', 'Contrast', 'Saturation', 'Hue', 'Gamma', 'JpegQuality']


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


class ToGray(NestedObject):
    """Convert a RGB tensor (batch of images or image) to a 3-channels grayscale tensor

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = ToGray()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:

    """
    def __init__(self) -> None:
        pass

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        grey = tf.image.rgb_to_grayscale(img)
        # Retrieve last dimension
        grey = tf.concat([grey, grey, grey], axis=-1)
        return grey


class InvertColorize(NestedObject):
    """Applies the following tranformation to a tensor (image or batch of images):
    convert to grayscale, colorize (shift 0-values randomly), and then invert colors

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = InvertColorize(r_min=0.8, g_min=0.8, b_min=0.8)
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        r_min: range [r_min, 1] to colorize pixels to red (0 convert all original white pixels to red)
        g_min: range [g_min, 1] to colorize pixels to green (0 convert all original white pixels to green)
        b_min: range [b_min, 1] to colorize pixels to blue (0 convert all original white pixels to blue)
    """
    def __init__(self, r_min: float = 0.6, g_min: float = 0.6, b_min: float = 0.6) -> None:
        self.r_min = r_min
        self.g_min = g_min
        self.b_min = b_min
        self.togray = ToGray()

    def extra_repr(self) -> str:
        return f"r_min={self.r_min}, g_max={self.g_min}, b_min={self.b_min}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        gray = self.togray(img)  # Convert to gray
        # Random RGB shifts
        if len(img.shape) == 4:
            batch_size = img.shape[0]
            r_shift = tf.random.uniform(shape=[batch_size, 1, 1], minval=self.r_min, maxval=1)
            g_shift = tf.random.uniform(shape=[batch_size, 1, 1], minval=self.g_min, maxval=1)
            b_shift = tf.random.uniform(shape=[batch_size, 1, 1], minval=self.b_min, maxval=1)
        else:  # No batch dim
            r_shift = tf.random.uniform(shape=[1, 1], minval=self.r_min, maxval=1)
            g_shift = tf.random.uniform(shape=[1, 1], minval=self.g_min, maxval=1)
            b_shift = tf.random.uniform(shape=[1, 1], minval=self.b_min, maxval=1)
        rgb_shift = tf.stack([r_shift, g_shift, b_shift], axis=-1)
        colorized = tf.multiply(gray, rgb_shift)
        # Invert values
        inverted = tf.ones_like(colorized) - colorized
        return inverted


class Brightness(NestedObject):
    """Adjust brightness of a tensor (batch of images or image) by adding delta
    to all pixels

    Example:
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Brightness()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        delta: offset to add to each value. Can be negative to darken pictures.
    """
    def __init__(self, delta: float = 0.3) -> None:
        self.delta = delta

    def extra_repr(self) -> str:
        return f"delta={self.delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.adjust_brightness(img, delta=self.delta)


class Contrast(NestedObject):
    """Adjust contrast of a tensor (batch of images or image) by adjusting
    each pixel: (img - mean) * contrast_factor + mean.

    Example:
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Contrast()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        contrast_factor: multiplicative factor to use to augment (if > 1) or reduce (if < 1) contrast
    """
    def __init__(self, contrast_factor: float = 1.3) -> None:
        self.contrast_factor = contrast_factor

    def extra_repr(self) -> str:
        return f"contrast_factor={self.contrast_factor}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.adjust_contrast(img, contrast_factor=self.contrast_factor)


class Saturation(NestedObject):
    """Adjust saturation of a tensor (batch of images or image) by converting to HSV and
    increasing saturation by a factor.

    Example:
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Saturation()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        saturation_factor: multiplicative factor to use to augment (if > 1) or reduce (if < 1) saturation
    """
    def __init__(self, saturation_factor: float = 1.5) -> None:
        self.saturation_factor = saturation_factor

    def extra_repr(self) -> str:
        return f"saturation_factor={self.saturation_factor}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.adjust_saturation(img, saturation_factor=self.saturation_factor)


class Hue(NestedObject):
    """Adjust hue of a tensor (batch of images or image) by converting to HSV and adding delta

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Hue()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        delta: offset to add to each value. Can be negative to darken pictures.
    """
    def __init__(self, delta: float = 0.3) -> None:
        self.delta = delta

    def extra_repr(self) -> str:
        return f"delta={self.delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.adjust_hue(img, delta=self.delta)


class Gamma(NestedObject):
    """Performs gamma correction for a tensor (batch of images or image)

    Example:
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Gamma()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        gamma: non-negative real number
        gain: constant multiplier
    """
    def __init__(self, gamma: float = 0.8, gain: float = 1.5) -> None:
        self.gamma = gamma
        self.gain = gain

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, gain={self.gain}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.adjust_gamma(img, gamma=self.gamma, gain=self.gain)


class JpegQuality(NestedObject):
    """Adjust jpeg quality of a 3 dimensional RGB image

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = JpegQuality()
        >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        quality: int between [0, 100], 100 = perfect, 0 = very degraded
    """
    def __init__(self, quality: int = 60) -> None:
        self.quality = quality

    def extra_repr(self) -> str:
        return f"quality={self.quality}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.adjust_jpeg_quality(img, jpeg_quality=self.quality)
