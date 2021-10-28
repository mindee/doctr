# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from doctr.utils.common_types import AbstractFile

__all__ = ['read_img_as_numpy']


def read_img_as_numpy(
    file: AbstractFile,
    output_width: Optional[int] = None,
    output_height: Optional[int] = None,
    rgb_output: bool = True,
) -> np.ndarray:
    """Read an image file into numpy format

    Example::
        >>> from doctr.documents import read_img
        >>> page = read_img("path/to/your/doc.jpg")

    Args:
        file: the path to the image file
        output_width: the expected output width of each page keep ratio to height
        output_height: the expected output height of each page keep ratio to width
        rgb_output: whether the output ndarray channel order should be RGB instead of BGR.
    Returns:
        the page decoded as numpy ndarray of shape H x W x 3
    """

    if isinstance(file, (str, Path)):
        if not Path(file).is_file():
            raise FileNotFoundError(f"unable to access {file}")
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
    elif isinstance(file, bytes):
        file = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(file, cv2.IMREAD_COLOR)
    else:
        raise TypeError("unsupported object type for argument 'file'")

    # Validity check
    if img is None:
        raise ValueError("unable to read file.")
    # Resizing
    if isinstance(output_width, int) or isinstance(output_height, int):
        img = _resize_image(img, output_width, output_height)
    # Switch the channel order
    if rgb_output:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _resize_image(image, width: Optional[int] = None, height: Optional[int] = None):
    """
    resize image and keep ratio
    """
    if width and height:
        dim = (width, height)
    elif height:
        (h, w) = image.shape[:2]
        r = height / float(h)
        dim = (int(w * r), height)
    elif width:
        (h, w) = image.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
    inter = cv2.INTER_AREA if dim[0] < 1200 else cv2.INTER_CUBIC
    image = cv2.resize(image, dim, interpolation=inter)

    return image
