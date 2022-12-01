# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Optional, List, Generator
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

from doctr.utils.common_types import AbstractFile

__all__ = ["PdfRenderer"]


class PdfRenderer:

    # iterable with known length

    def __init__(
        self,
        file: AbstractFile,
        scale: float = 2,
        page_indices: Optional[List[int]] = None,
        password: Optional[str] = None,
        rgb_mode: bool = True,
        **kwargs: Any,
    ):
        """
        Read a PDF file and convert it to images in numpy format.
        This class behaves like an iterator with known length.

        >>> from doctr.documents import PdfRenderer
        >>> doc = PdfRenderer("path/to/your/doc.pdf")
        >>> n_pages = len(doc)
        >>> first_page = next(doc)
        >>> for further_page in doc:
        >>>     do_something(further_page)

        Args:
            file: the path to the PDF file
            scale: rendering scale (1 corresponds to 72dpi)
            rgb_mode: if True, the output will be RGB, otherwise BGR
            password: a password to unlock the document, if encrypted
            kwargs: additional parameters to :meth:`pypdfium2.PdfDocument.render_to`
        """

        if isinstance(file, Path):  # v3 compat
            file = str(file)

        pdf = pdfium.PdfDocument(file, password=password)

        if page_indices:
            self._len = len(page_indices)
        else:
            self._len = len(pdf)

        render_kwargs = dict(scale=scale, page_indices=page_indices, rev_byteorder=rgb_mode, **kwargs)
        if hasattr(pdf, "render_to"):  # v3 compat
            self._generator = (p for p, _ in pdf.render_to(pdfium.BitmapConv.numpy_ndarray, **render_kwargs))
        else:  # upcoming v4
            self._generator = pdf.render(pdfium.PdfBitmap.to_numpy, **render_kwargs)

    def __len__(self) -> int:
        return self._len

    def __next__(self) -> np.ndarray:
        return next(self._generator)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        yield from self._generator
