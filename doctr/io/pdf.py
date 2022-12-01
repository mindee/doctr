# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Optional, List
from pathlib import Path

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

    def __next__(self):
        return next(self._generator)

    def __iter__(self):
        yield from self._generator
