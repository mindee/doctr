# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pypdfium2 as pdfium

from doctr.utils.common_types import AbstractFile

__all__ = ['read_pdf']


def read_pdf(
    file: AbstractFile,
    scale: float = 2,
    password: Optional[str] = None,
    **kwargs: Any,
) -> List[np.ndarray]:
    """Read a PDF file and convert it into an image in numpy format

    >>> from doctr.documents import read_pdf
    >>> doc = read_pdf("path/to/your/doc.pdf")

    Args:
        file: the path to the PDF file
        scale: rendering scale (1 corresponds to 72dpi)
        password: a password to unlock the document, if encrypted
        kwargs: additional parameters to :meth:`pypdfium2.PdfDocument.render_topil`

    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x C
    """

    if isinstance(file, Path):
        file = str(file)
    if not isinstance(file, (str, bytes)):
        raise TypeError("unsupported object type for argument 'file'")

    # Rasterise pages to PIL images with pypdfium2 and convert to numpy ndarrays
    with pdfium.PdfDocument(file, password=password) as pdf:
        return [np.asarray(img) for img in pdf.render_topil(scale=scale, **kwargs)]
