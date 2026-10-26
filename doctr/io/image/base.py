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


def _decode_transparent_img(content: bytes) -> np.ndarray | None:
    """Decode an image with transparency over a white background, in BGR order (None if it has no transparency)"""
    try:
        with Image.open(BytesIO(content)) as pil_img:
            if not pil_img.has_transparency_data:
                return None
            # Apply the EXIF orientation, like `cv2.imdecode` does
            rgba = ImageOps.exif_transpose(pil_img).convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError):  # left to OpenCV
        return None
    # Dropping the alpha channel would reveal the color stored under the transparent pixels (often black)
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return cv2.cvtColor(np.asarray(Image.alpha_composite(background, rgba).convert("RGB")), cv2.COLOR_RGB2BGR)


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
    if img is None:
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    # Validity check
    if img is None:
        raise ValueError("unable to read file.")
    # Resizing
    if isinstance(output_size, tuple):
        img = cv2.resize(img, output_size[::-1], interpolation=cv2.INTER_LINEAR)
    # Switch the channel order
    if rgb_output:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
