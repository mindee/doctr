# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pypdfium2 as pdfium

from doctr.utils.common_types import AbstractFile

__all__ = ["read_pdf"]


def read_pdf(
    file: AbstractFile,
    scale: float = 2,
    rgb_mode: bool = True,
    password: Optional[str] = None,
    **kwargs: Any,
) -> List[np.ndarray]:
    """Read a PDF file and convert it into an image in numpy format

    >>> from doctr.documents import read_pdf
    >>> doc = read_pdf("path/to/your/doc.pdf")

    Args:
        file: the path to the PDF file
        scale: rendering scale (1 corresponds to 72dpi)
        rgb_mode: if True, the output will be RGB, otherwise BGR
        password: a password to unlock the document, if encrypted
        kwargs: additional parameters to :meth:`pypdfium2.PdfDocument.render_to`

    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x C
    """

    if isinstance(file, Path):
        file = str(file)

    # Rasterise pages to numpy ndarrays with pypdfium2
    pdf = pdfium.PdfDocument(file, password=password)
    renderer = pdf.render_to(pdfium.BitmapConv.numpy_ndarray, scale=scale, rev_byteorder=rgb_mode, **kwargs)
    return [img for img, _ in renderer]
