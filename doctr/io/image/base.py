# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from doctr.utils.common_types import AbstractFile

__all__ = ["read_img_as_numpy"]


def _pil_to_rgb_array(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB array, with its EXIF orientation applied (like `cv2.imdecode` does)
    and transparent pixels placed on a white background

    Args:
        pil_img: a PIL image

    Returns:
        the image as a writable numpy array of shape H x W x 3 in np.uint8
    """
    has_transparency = pil_img.has_transparency_data
    ImageOps.exif_transpose(pil_img, in_place=True)
    if not has_transparency:
        return np.array(pil_img.convert("RGB"))
    rgba = np.asarray(pil_img if pil_img.mode == "RGBA" else pil_img.convert("RGBA"))
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    alpha = rgba[..., 3]
    # Fully opaque images (e.g. screenshots) need no compositing
    if alpha.min() == 255:
        return rgb
    # Dropping the alpha channel would reveal the color stored under the transparent pixels (often black),
    # so blend with white in place: 255 - (255 - rgb) * alpha / 255
    cv2.bitwise_not(rgb, dst=rgb)
    cv2.multiply(rgb, cv2.cvtColor(np.ascontiguousarray(alpha), cv2.COLOR_GRAY2RGB), dst=rgb, scale=1 / 255)
    return cv2.bitwise_not(rgb, dst=rgb)


def _decode_transparent_img(content: bytes) -> np.ndarray | None:
    """Decode an image with transparency over a white background, in RGB order (None if it has no transparency)"""
    try:
        with Image.open(BytesIO(content)) as pil_img:
            if not pil_img.has_transparency_data:
                return None
            return _pil_to_rgb_array(pil_img)
    except (OSError, ValueError, Image.DecompressionBombError):  # left to OpenCV
        return None


def read_img_as_numpy(
    file: AbstractFile,
    output_size: tuple[int, int] | None = None,
    rgb_output: bool = True,
) -> np.ndarray:
    """Read an image file into numpy format

    >>> from doctr.io import read_img_as_numpy
    >>> page = read_img_as_numpy("path/to/your/doc.jpg")

    Args:
        file: the path to the image file
        output_size: the expected output size of each page in format H x W
        rgb_output: whether the output ndarray channel order should be RGB instead of BGR.

    Returns:
        the page decoded as numpy ndarray of shape H x W x 3, transparent images are placed on a white background
    """
    if isinstance(file, (str, Path)):
        if not Path(file).is_file():
            raise FileNotFoundError(f"unable to access {file}")
        content = Path(file).read_bytes()
    elif isinstance(file, bytes):
        content = file
    else:
        raise TypeError("unsupported object type for argument 'file'")

    img = _decode_transparent_img(content)
    is_rgb = img is not None
    if img is None:
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    # Validity check
    if img is None:
        raise ValueError("unable to read file.")
    # Resizing
    if isinstance(output_size, tuple):
        img = cv2.resize(img, output_size[::-1], interpolation=cv2.INTER_LINEAR)
    # Switch the channel order (OpenCV decodes as BGR, Pillow as RGB)
    if rgb_output != is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if rgb_output else cv2.COLOR_RGB2BGR)
    return img
