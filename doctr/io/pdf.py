# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from pathlib import Path
from typing import Any, List

import pypdfium2 as pdfium
import numpy as np

from doctr.utils.common_types import AbstractFile

__all__ = ['read_pdf', 'PDF']


def read_pdf(file: AbstractFile, default_scale: float = 2, **kwargs: Any) -> List[np.ndarray]:
    """Read a PDF file and convert it into an image in numpy format

    Example::
        >>> from doctr.documents import read_pdf
        >>> doc = read_pdf("path/to/your/doc.pdf")

    Args:
        file: the path to the PDF file
    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x 3
    """

    if isinstance(file, (str, Path)) and not Path(file).is_file():
        raise FileNotFoundError(f"unable to access {file}")

    # Read pages with fitz and convert them to numpy ndarrays
    return [np.asarray(img) for img, _ in pdfium.render_pdf("/home/fg/Downloads/sample.pdf", scale=default_scale)]
