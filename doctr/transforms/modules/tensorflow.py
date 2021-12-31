# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import random
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from doctr.utils.repr import NestedObject

__all__ = ['Compose', 'Resize', 'Normalize', 'LambdaTransformation', 'ToGray', 'RandomBrightness',
           'RandomContrast', 'RandomSaturation', 'RandomHue', 'RandomGamma', 'RandomJpegQuality', 'GaussianBlur',
           'ChannelShuffle', 'GaussianNoise', 'RandomHorizontalFlip']


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

    def __init__(self, transforms: List[Callable[[Any], Any]]) -> None:
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
        symmetric_pad: if `True` while preserving aspect ratio, the padding will be done symmetrically
    """
    def __init__(
        self,
        output_size: Tuple[int, int],
        method: str = 'bilinear',
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        self.output_size = output_size
        self.method = method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

    def extra_repr(self) -> str:
        _repr = f"output_size={self.output_size}, method='{self.method}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return _repr

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        input_dtype = img.dtype
        img = tf.image.resize(img, self.output_size, self.method, self.preserve_aspect_ratio)
        if self.preserve_aspect_ratio:
            # pad width
            if not self.symmetric_pad:
                offset = (0, 0)
            elif self.output_size[0] == img.shape[0]:
                offset = (0, int((self.output_size[1] - img.shape[1]) / 2))
            else:
                offset = (int((self.output_size[0] - img.shape[0]) / 2), 0)
            img = tf.image.pad_to_bounding_box(img, *offset, *self.output_size)
        return tf.cast(img, dtype=input_dtype)


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
        self.mean = tf.constant(mean)
        self.std = tf.constant(std)

    def extra_repr(self) -> str:
        return f"mean={self.mean.numpy().tolist()}, std={self.std.numpy().tolist()}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        img -= tf.cast(self.mean, dtype=img.dtype)
        img /= tf.cast(self.std, dtype=img.dtype)
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
    """
    def __init__(self, num_output_channels: int = 1):
        self.num_output_channels = num_output_channels

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        img = tf.image.rgb_to_grayscale(img)
        return img if self.num_output_channels == 1 else tf.repeat(img, self.num_output_channels, axis=-1)


class RandomBrightness(NestedObject):
    """Randomly adjust brightness of a tensor (batch of images or image) by adding a delta
    to all pixels

    Example:
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Brightness()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        max_delta: offset to add to each pixel is randomly picked in [-max_delta, max_delta]
        p: probability to apply transformation
    """
    def __init__(self, max_delta: float = 0.3) -> None:
        self.max_delta = max_delta

    def extra_repr(self) -> str:
        return f"max_delta={self.max_delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_brightness(img, max_delta=self.max_delta)


class RandomContrast(NestedObject):
    """Randomly adjust contrast of a tensor (batch of images or image) by adjusting
    each pixel: (img - mean) * contrast_factor + mean.

    Example:
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Contrast()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        delta: multiplicative factor is picked in [1-delta, 1+delta] (reduce contrast if factor<1)
    """
    def __init__(self, delta: float = .3) -> None:
        self.delta = delta

    def extra_repr(self) -> str:
        return f"delta={self.delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_contrast(img, lower=1 - self.delta, upper=1 / (1 - self.delta))


class RandomSaturation(NestedObject):
    """Randomly adjust saturation of a tensor (batch of images or image) by converting to HSV and
    increasing saturation by a factor.

    Example:
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Saturation()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        delta: multiplicative factor is picked in [1-delta, 1+delta] (reduce saturation if factor<1)
    """
    def __init__(self, delta: float = .5) -> None:
        self.delta = delta

    def extra_repr(self) -> str:
        return f"delta={self.delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_saturation(img, lower=1 - self.delta, upper=1 + self.delta)


class RandomHue(NestedObject):
    """Randomly adjust hue of a tensor (batch of images or image) by converting to HSV and adding a delta

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Hue()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        max_delta: offset to add to each pixel is randomly picked in [-max_delta, max_delta]
    """
    def __init__(self, max_delta: float = 0.3) -> None:
        self.max_delta = max_delta

    def extra_repr(self) -> str:
        return f"max_delta={self.max_delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_hue(img, max_delta=self.max_delta)


class RandomGamma(NestedObject):
    """randomly performs gamma correction for a tensor (batch of images or image)

    Example:
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = Gamma()
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        min_gamma: non-negative real number, lower bound for gamma param
        max_gamma: non-negative real number, upper bound for gamma
        min_gain: lower bound for constant multiplier
        max_gain: upper bound for constant multiplier
    """
    def __init__(
        self,
        min_gamma: float = 0.5,
        max_gamma: float = 1.5,
        min_gain: float = 0.8,
        max_gain: float = 1.2,
    ) -> None:
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.min_gain = min_gain
        self.max_gain = max_gain

    def extra_repr(self) -> str:
        return f"""gamma_range=({self.min_gamma}, {self.max_gamma}),
                 gain_range=({self.min_gain}, {self.max_gain})"""

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        gain = random.uniform(self.min_gain, self.max_gain)
        return tf.image.adjust_gamma(img, gamma=gamma, gain=gain)


class RandomJpegQuality(NestedObject):
    """Randomly adjust jpeg quality of a 3 dimensional RGB image

    Example::
        >>> from doctr.transforms import Normalize
        >>> import tensorflow as tf
        >>> transfo = JpegQuality()
        >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        min_quality: int between [0, 100]
        max_quality: int between [0, 100]
    """
    def __init__(self, min_quality: int = 60, max_quality: int = 100) -> None:
        self.min_quality = min_quality
        self.max_quality = max_quality

    def extra_repr(self) -> str:
        return f"min_quality={self.min_quality}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_jpeg_quality(
            img, min_jpeg_quality=self.min_quality, max_jpeg_quality=self.max_quality
        )


class GaussianBlur(NestedObject):
    """Randomly adjust jpeg quality of a 3 dimensional RGB image

    Example::
        >>> from doctr.transforms import GaussianBlur
        >>> import tensorflow as tf
        >>> transfo = GaussianBlur(3, (.1, 5))
        >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        kernel_shape: size of the blurring kernel
        std: min and max value of the standard deviation
    """
    def __init__(self, kernel_shape: Union[int, Iterable[int]], std: Tuple[float, float]) -> None:
        self.kernel_shape = kernel_shape
        self.std = std

    def extra_repr(self) -> str:
        return f"kernel_shape={self.kernel_shape}, std={self.std}"

    @tf.function
    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        sigma = random.uniform(self.std[0], self.std[1])
        return tfa.image.gaussian_filter2d(
            img, filter_shape=self.kernel_shape, sigma=sigma,
        )


class ChannelShuffle(NestedObject):
    """Randomly shuffle channel order of a given image"""

    def __init__(self):
        pass

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.transpose(tf.random.shuffle(tf.transpose(img, perm=[2, 0, 1])), perm=[1, 2, 0])


class GaussianNoise(NestedObject):
    """Adds Gaussian Noise to the input tensor

       Example::
           >>> from doctr.transforms import GaussianNoise
           >>> import tensorflow as tf
           >>> transfo = GaussianNoise(0., 1.)
           >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

       Args:
           mean : mean of the gaussian distribution
           std : std of the gaussian distribution
       """
    def __init__(self, mean: float = 0., std: float = 1.) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Reshape the distribution
        noise = self.mean + 2 * self.std * tf.random.uniform(x.shape) - self.std
        if x.dtype == tf.uint8:
            return tf.cast(
                tf.clip_by_value(tf.math.round(tf.cast(x, dtype=tf.float32) + 255 * noise), 0, 255),
                dtype=tf.uint8
            )
        else:
            return tf.cast(tf.clip_by_value(x + noise, 0, 1), dtype=x.dtype)

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class RandomHorizontalFlip(NestedObject):
    """Adds random horizontal flip to the input tensor/np.ndarray

    Example::
           >>> from doctr.transforms import RandomHorizontalFlip
           >>> import tensorflow as tf
           >>> transfo = RandomHorizontalFlip(p=0.5)
           >>> image = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1)
           >>> target = {
            "boxes": np.array([[0.1, 0.1, 0.4, 0.5] ], dtype= np.float32),
            "labels": np.ones(1, dtype= np.int64)
            }
           >>> out = transfo(image, target)

       Args:
           p : probability of Horizontal Flip"""
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def __call__(
            self,
            img: Union[tf.Tensor, np.ndarray],
            target: Dict[str, Any]
    ) -> Tuple[Union[tf.Tensor, np.ndarray], Dict[str, Any]]:
        """
        Args:
            img: Image to be flipped.
            target: Dictionary with boxes (in relative coordinates of shape (N, 4)) and labels as keys
        Returns:
            Tuple of numpy nd-array or Tensor and target
        """
        if np.random.rand(1) <= self.p:
            _img = tf.image.flip_left_right(img)
            _target = target.copy()
            # Changing the relative bbox coordinates
            _target["boxes"][:, ::2] = 1 - target["boxes"][:, [2, 0]]
            return _img, _target
        return img, target
