# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

__all__ = ["read_html"]


def read_html(url: str, **kwargs: Any) -> bytes:
    """Read a PDF file and convert it into an image in numpy format

    >>> from doctr.io import read_html
    >>> doc = read_html("https://www.yoursite.com")

    Args:
        url: URL of the target web page
        **kwargs: keyword arguments from `weasyprint.HTML`

    Returns:
        decoded PDF file as a bytes stream
    """
    from weasyprint import HTML

    return HTML(url, **kwargs).write_pdf()
