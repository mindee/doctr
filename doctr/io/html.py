# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from weasyprint import HTML
from typing import Any

__all__ = ['read_html']


def read_html(url: str, **kwargs: Any) -> bytes:
    """Read a PDF file and convert it into an image in numpy format

    Example::
        >>> from doctr.documents import read_html
        >>> doc = read_html("https://www.yoursite.com")

    Args:
        url: URL of the target web page
    Returns:
        decoded PDF file as a bytes stream
    """

    return HTML(url, **kwargs).write_pdf()
