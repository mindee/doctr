# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from pathlib import Path

import cv2
import numpy as np

from doctr.utils.common_types import AbstractFile

__all__ = ["read_img_as_numpy"]


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
        the page decoded as numpy ndarray of shape H x W x 3
    """
    if isinstance(file, (str, Path)):
        if not Path(file).is_file():
            raise FileNotFoundError(f"unable to access {file}")
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
    elif isinstance(file, bytes):
        _file: np.ndarray = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(_file, cv2.IMREAD_COLOR)
    else:
        raise TypeError("unsupported object type for argument 'file'")

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
