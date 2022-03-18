# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os.path
from pathlib import Path
from typing import Any, List

import numpy as np
import pypdfium2 as pdfium

from doctr.utils.common_types import AbstractFile

__all__ = ['read_pdf']


def read_pdf(file: AbstractFile, scale: float = 2, **kwargs: Any) -> List[np.ndarray]:
    """Read a PDF file and convert it into an image in numpy format

    >>> from doctr.documents import read_pdf
    >>> doc = read_pdf("path/to/your/doc.pdf")

    Args:
        file: the path to the PDF file
        scale: rendering scale (1 corresponds to 72dpi)
        kwargs: additional parameters to :func:`pypdfium2._helpers.pdf_renderer.render_pdf_topil`

    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x C
    """

    if isinstance(file, Path):
        file = str(file)
    if not isinstance(file, (str, bytes)):
        raise TypeError("unsupported object type for argument 'file'")

    if isinstance(file, str) and not os.path.isfile(file):
        raise FileNotFoundError(f"unable to access {file}")

    # Rasterise pages to PIL images with pypdfium2 and convert to numpy ndarrays
    return [np.asarray(img) for img, _ in pdfium.render_pdf_topil(file, scale=scale, **kwargs)]
