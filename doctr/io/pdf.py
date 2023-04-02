# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Optional, List, Generator

import numpy as np
import pypdfium2 as pdfium

from doctr.utils.common_types import AbstractFile

__all__ = ["PdfRenderer"]


class PdfRenderer:

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

        >>> from doctr.io.pdf import PdfRenderer
        >>> doc = PdfRenderer("path/to/your/doc.pdf")
        >>> n_pages = len(doc)
        >>> first_page = next(doc)
        >>> for further_page in doc:
        >>>     do_something(further_page)

        Args:
            file: the path to the PDF file
            scale: rendering scale (1 corresponds to 72dpi)
            page_indices: indices of the pages to include
            rgb_mode: if True, the output will be RGB, otherwise BGR
            password: a password to unlock the document, if encrypted
            kwargs: additional parameters to :meth:`pypdfium2.PdfDocument.render`
        """

        pdf = pdfium.PdfDocument(file, password=password)
        self._len = len(page_indices) if page_indices else len(pdf)
        self._generator = pdf.render(
            pdfium.PdfBitmap.to_numpy,
            scale=scale, page_indices=page_indices, rev_byteorder=rgb_mode, **kwargs
        )

    def __len__(self) -> int:
        return self._len

    def __next__(self) -> np.ndarray:
        self._len -= 1
        return next(self._generator)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while True:
            try:
                yield next(self)
            except StopIteration:
                break
