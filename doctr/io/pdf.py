# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from pathlib import Path
from typing import List

import pdf2image
import numpy as np

from doctr.utils.common_types import AbstractFile

__all__ = ['read_pdf_as_numpy']


def read_pdf_as_numpy(file: AbstractFile) -> List[np.ndarray]:
    """Read a PDF file and convert it into an image in numpy format

    Example::
        >>> from doctr.documents import read_pdf
        >>> doc = read_pdf_as_numpy("path/to/your/doc.pdf")

    Args:
        file: the path to the PDF file
    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x 3
    """

    if isinstance(file, (str, Path)) and not Path(file).is_file():
        raise FileNotFoundError(f"unable to access {file}")

    if isinstance(file, (str, Path)):
        pil_images = pdf2image.convert_from_path(
            file,
            dpi=144,  # To keep the same behaviour than before with fitz
        )
    elif isinstance(file, bytes):
        pil_images = pdf2image.convert_from_bytes(
            file,
            dpi=144,
        )
    else:
        raise TypeError("unsupported object type for argument 'file'")

    # Convert pages to numpy ndarrays
    return [np.array(pil_img, np.uint8, copy=True) for pil_img in pil_images]
