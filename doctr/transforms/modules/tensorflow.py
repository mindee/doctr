# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import tensorflow as tf

from doctr.utils.repr import NestedObject

from ..functional.tensorflow import _gaussian_filter, random_shadow

__all__ = [
    "Compose",
    "Resize",
    "Normalize",
    "LambdaTransformation",
    "ToGray",
    "RandomBrightness",
    "RandomContrast",
    "RandomSaturation",
    "RandomHue",
    "RandomGamma",
    "RandomJpegQuality",
    "GaussianBlur",
    "ChannelShuffle",
    "GaussianNoise",
    "RandomHorizontalFlip",
    "RandomShadow",
    "RandomResize",
]


class Compose(NestedObject):
    """Implements a wrapper that will apply transformations sequentially

    >>> import tensorflow as tf
    >>> from doctr.transforms import Compose, Resize
    >>> transfos = Compose([Resize((32, 32))])
    >>> out = transfos(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transforms: list of transformation modules
    """

    _children_names: list[str] = ["transforms"]

    def __init__(self, transforms: list[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)

        return x


class Resize(NestedObject):
    """Resizes a tensor to a target size

    >>> import tensorflow as tf
    >>> from doctr.transforms import Resize
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
        output_size: int | tuple[int, int],
        method: str = "bilinear",
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        self.output_size = output_size
        self.method = method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.antialias = True

        if isinstance(self.output_size, int):
            self.wanted_size = (self.output_size, self.output_size)
        elif isinstance(self.output_size, (tuple, list)):
            self.wanted_size = self.output_size
        else:
            raise AssertionError("Output size should be either a list, a tuple or an int")

    def extra_repr(self) -> str:
        _repr = f"output_size={self.output_size}, method='{self.method}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return _repr

    def __call__(
        self,
        img: tf.Tensor,
        target: np.ndarray | None = None,
    ) -> tf.Tensor | tuple[tf.Tensor, np.ndarray]:
        input_dtype = img.dtype
        self.output_size = (
            (self.output_size, self.output_size) if isinstance(self.output_size, int) else self.output_size
        )

        img = tf.image.resize(img, self.wanted_size, self.method, self.preserve_aspect_ratio, self.antialias)
        # It will produce an un-padded resized image, with a side shorter than wanted if we preserve aspect ratio
        raw_shape = img.shape[:2]
        if self.symmetric_pad:
            half_pad = (int((self.output_size[0] - img.shape[0]) / 2), 0)
        if self.preserve_aspect_ratio:
            if isinstance(self.output_size, (tuple, list)):
                # In that case we need to pad because we want to enforce both width and height
                if not self.symmetric_pad:
                    half_pad = (0, 0)
                elif self.output_size[0] == img.shape[0]:
                    half_pad = (0, int((self.output_size[1] - img.shape[1]) / 2))
                # Pad image
                img = tf.image.pad_to_bounding_box(img, *half_pad, *self.output_size)

        # In case boxes are provided, resize boxes if needed (for detection task if preserve aspect ratio)
        if target is not None:
            if self.symmetric_pad:
                offset = half_pad[0] / img.shape[0], half_pad[1] / img.shape[1]

            if self.preserve_aspect_ratio:
                # Get absolute coords
                if target.shape[1:] == (4,):
                    if isinstance(self.output_size, (tuple, list)) and self.symmetric_pad:
                        target[:, [0, 2]] = offset[1] + target[:, [0, 2]] * raw_shape[1] / img.shape[1]
                        target[:, [1, 3]] = offset[0] + target[:, [1, 3]] * raw_shape[0] / img.shape[0]
                    else:
                        target[:, [0, 2]] *= raw_shape[1] / img.shape[1]
                        target[:, [1, 3]] *= raw_shape[0] / img.shape[0]
                elif target.shape[1:] == (4, 2):
                    if isinstance(self.output_size, (tuple, list)) and self.symmetric_pad:
                        target[..., 0] = offset[1] + target[..., 0] * raw_shape[1] / img.shape[1]
                        target[..., 1] = offset[0] + target[..., 1] * raw_shape[0] / img.shape[0]
                    else:
                        target[..., 0] *= raw_shape[1] / img.shape[1]
                        target[..., 1] *= raw_shape[0] / img.shape[0]
                else:
                    raise AssertionError("Boxes should be in the format (n_boxes, 4, 2) or (n_boxes, 4)")

            return tf.cast(img, dtype=input_dtype), np.clip(target, 0, 1)

        return tf.cast(img, dtype=input_dtype)


class Normalize(NestedObject):
    """Normalize a tensor to a Gaussian distribution for each channel

    >>> import tensorflow as tf
    >>> from doctr.transforms import Normalize
    >>> transfo = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        mean: average value per channel
        std: standard deviation per channel
    """

    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
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

    >>> import tensorflow as tf
    >>> from doctr.transforms import LambdaTransformation
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

    >>> import tensorflow as tf
    >>> from doctr.transforms import ToGray
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

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomBrightness
    >>> transfo = RandomBrightness()
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

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomContrast
    >>> transfo = RandomContrast()
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        delta: multiplicative factor is picked in [1-delta, 1+delta] (reduce contrast if factor<1)
    """

    def __init__(self, delta: float = 0.3) -> None:
        self.delta = delta

    def extra_repr(self) -> str:
        return f"delta={self.delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_contrast(img, lower=1 - self.delta, upper=1 / (1 - self.delta))


class RandomSaturation(NestedObject):
    """Randomly adjust saturation of a tensor (batch of images or image) by converting to HSV and
    increasing saturation by a factor.

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomSaturation
    >>> transfo = RandomSaturation()
    >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        delta: multiplicative factor is picked in [1-delta, 1+delta] (reduce saturation if factor<1)
    """

    def __init__(self, delta: float = 0.5) -> None:
        self.delta = delta

    def extra_repr(self) -> str:
        return f"delta={self.delta}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_saturation(img, lower=1 - self.delta, upper=1 + self.delta)


class RandomHue(NestedObject):
    """Randomly adjust hue of a tensor (batch of images or image) by converting to HSV and adding a delta

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomHue
    >>> transfo = RandomHue()
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

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomGamma
    >>> transfo = RandomGamma()
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

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomJpegQuality
    >>> transfo = RandomJpegQuality()
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
        return tf.image.random_jpeg_quality(img, min_jpeg_quality=self.min_quality, max_jpeg_quality=self.max_quality)


class GaussianBlur(NestedObject):
    """Randomly adjust jpeg quality of a 3 dimensional RGB image

    >>> import tensorflow as tf
    >>> from doctr.transforms import GaussianBlur
    >>> transfo = GaussianBlur(3, (.1, 5))
    >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        kernel_shape: size of the blurring kernel
        std: min and max value of the standard deviation
    """

    def __init__(self, kernel_shape: int | Iterable[int], std: tuple[float, float]) -> None:
        self.kernel_shape = kernel_shape
        self.std = std

    def extra_repr(self) -> str:
        return f"kernel_shape={self.kernel_shape}, std={self.std}"

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(
            _gaussian_filter(
                img[tf.newaxis, ...],
                kernel_size=self.kernel_shape,
                sigma=random.uniform(self.std[0], self.std[1]),
                mode="REFLECT",
            ),
            axis=0,
        )


class ChannelShuffle(NestedObject):
    """Randomly shuffle channel order of a given image"""

    def __init__(self):
        pass

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return tf.transpose(tf.random.shuffle(tf.transpose(img, perm=[2, 0, 1])), perm=[1, 2, 0])


class GaussianNoise(NestedObject):
    """Adds Gaussian Noise to the input tensor

    >>> import tensorflow as tf
    >>> from doctr.transforms import GaussianNoise
    >>> transfo = GaussianNoise(0., 1.)
    >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        mean : mean of the gaussian distribution
        std : std of the gaussian distribution
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Reshape the distribution
        noise = self.mean + 2 * self.std * tf.random.uniform(x.shape) - self.std
        if x.dtype == tf.uint8:
            return tf.cast(
                tf.clip_by_value(tf.math.round(tf.cast(x, dtype=tf.float32) + 255 * noise), 0, 255), dtype=tf.uint8
            )
        else:
            return tf.cast(tf.clip_by_value(x + noise, 0, 1), dtype=x.dtype)

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class RandomHorizontalFlip(NestedObject):
    """Adds random horizontal flip to the input tensor/np.ndarray

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomHorizontalFlip
    >>> transfo = RandomHorizontalFlip(p=0.5)
    >>> image = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1)
    >>> target = np.array([[0.1, 0.1, 0.4, 0.5] ], dtype= np.float32)
    >>> out = transfo(image, target)

    Args:
        p : probability of Horizontal Flip
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def __call__(self, img: tf.Tensor | np.ndarray, target: np.ndarray) -> tuple[tf.Tensor, np.ndarray]:
        if np.random.rand(1) <= self.p:
            _img = tf.image.flip_left_right(img)
            _target = target.copy()
            # Changing the relative bbox coordinates
            if target.shape[1:] == (4,):
                _target[:, ::2] = 1 - target[:, [2, 0]]
            else:
                _target[..., 0] = 1 - target[..., 0]
            return _img, _target
        return img, target


class RandomShadow(NestedObject):
    """Adds random shade to the input image

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomShadow
    >>> transfo = RandomShadow(0., 1.)
    >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        opacity_range : minimum and maximum opacity of the shade
    """

    def __init__(self, opacity_range: tuple[float, float] | None = None) -> None:
        super().__init__()
        self.opacity_range = opacity_range if isinstance(opacity_range, tuple) else (0.2, 0.8)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Reshape the distribution
        if x.dtype == tf.uint8:
            return tf.cast(
                tf.clip_by_value(
                    tf.math.round(255 * random_shadow(tf.cast(x, dtype=tf.float32) / 255, self.opacity_range)),
                    0,
                    255,
                ),
                dtype=tf.uint8,
            )
        else:
            return tf.clip_by_value(random_shadow(x, self.opacity_range), 0, 1)

    def extra_repr(self) -> str:
        return f"opacity_range={self.opacity_range}"


class RandomResize(NestedObject):
    """Randomly resize the input image and align corresponding targets

    >>> import tensorflow as tf
    >>> from doctr.transforms import RandomResize
    >>> transfo = RandomResize((0.3, 0.9), preserve_aspect_ratio=True, symmetric_pad=True, p=0.5)
    >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

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
    ):
        super().__init__()
        self.scale_range = scale_range
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.p = p
        self._resize = Resize

    def __call__(self, img: tf.Tensor, target: np.ndarray) -> tuple[tf.Tensor, np.ndarray]:
        if np.random.rand(1) <= self.p:
            scale_h = random.uniform(*self.scale_range)
            scale_w = random.uniform(*self.scale_range)
            new_size = (int(img.shape[-3] * scale_h), int(img.shape[-2] * scale_w))

            _img, _target = self._resize(
                new_size,
                preserve_aspect_ratio=self.preserve_aspect_ratio
                if isinstance(self.preserve_aspect_ratio, bool)
                else bool(np.random.rand(1) <= self.symmetric_pad),
                symmetric_pad=self.symmetric_pad
                if isinstance(self.symmetric_pad, bool)
                else bool(np.random.rand(1) <= self.symmetric_pad),
            )(img, target)

            return _img, _target
        return img, target

    def extra_repr(self) -> str:
        return f"scale_range={self.scale_range}, preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}, p={self.p}"  # noqa: E501
