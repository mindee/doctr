# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from weasyprint import HTML

__all__ = ["read_html"]


def read_html(url: str, **kwargs: Any) -> bytes:
    """Read a PDF file and convert it into an image in numpy format

    >>> from doctr.documents import read_html
    >>> doc = read_html("https://www.yoursite.com")

    Args:
        url: URL of the target web page

    Returns:
        decoded PDF file as a bytes stream
    """

    return HTML(url, **kwargs).write_pdf()
