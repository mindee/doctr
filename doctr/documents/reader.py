# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
from pathlib import Path
import fitz
from typing import List, Tuple, Optional, Any

__all__ = ['read_pdf', 'read_pdf_from_stream', 'read_img']


def read_img(
    file_path: str,
    output_size: Optional[Tuple[int, int]] = None,
    rgb_output: bool = True,
) -> np.ndarray:
    """Read an image file into numpy format

    Example::
        >>> from doctr.documents import read_img
        >>> page = read_img("path/to/your/doc.jpg")

    Args:
        file_path: the path to the image file
        output_size: the expected output size of each page in format H x W
        rgb_output: whether the output ndarray channel order should be RGB instead of BGR.
    Returns:
        the page decoded as numpy ndarray of shape H x W x 3
    """

    if not Path(file_path).is_file():
        raise FileNotFoundError(f"unable to access {file_path}")

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # Validity check
    if img is None:
        raise ValueError("unable to read file.")
    # Resizing
    if isinstance(output_size, tuple):
        img = cv2.resize(img, output_size[::-1], interpolation=cv2.INTER_LINEAR)
    # Switch the channel order
    if rgb_output:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_pdf(file_path: str, **kwargs: Any) -> List[np.ndarray]:
    """Read a PDF file and convert it into an image in numpy format

    Example::
        >>> from doctr.documents import read_pdf
        >>> doc = read_pdf("path/to/your/doc.pdf")

    Args:
        file_path: the path to the PDF file
    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x 3
    """

    # Read pages with fitz and convert them to numpy ndarrays
    return [convert_page_to_numpy(page, **kwargs) for page in fitz.open(file_path, filetype="pdf")]


def read_pdf_from_stream(stream: bytes, **kwargs: Any) -> List[np.ndarray]:
    """Read a PDF stream and convert it into an image in numpy format

    Example::
        >>> from doctr.documents import read_pdf_from_stream
        >>> with open("path/to/your/doc.pdf", 'rb') as f: doc = read_pdf_from_stream(f.read())

    Args:
        stream: serialized stream of the PDF content
    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x 3
    """

    # Read pages with fitz and convert them to numpy ndarrays
    return [convert_page_to_numpy(page, **kwargs) for page in fitz.open(stream=stream, filetype="pdf")]


def convert_page_to_numpy(
    page: fitz.fitz.Page,
    output_size: Optional[Tuple[int, int]] = None,
    rgb_output: bool = True,
) -> np.ndarray:
    """Convert a fitz page to a numpy-formatted image

    Args:
        page: the page of a file read with PyMuPDF
        output_size: the expected output size of each page in format H x W
        rgb_output: whether the output ndarray channel order should be RGB instead of BGR.

    Returns:
        the rendered image in numpy format
    """

    transform_matrix = None

    # If no output size is specified, keep the origin one
    if output_size is not None:
        scales = (output_size[1] / page.MediaBox[2], output_size[0] / page.MediaBox[3])
        transform_matrix = fitz.Matrix(*scales)

    # Generate the pixel map using the transformation matrix
    stream = page.getPixmap(matrix=transform_matrix).getImageData()
    # Decode it into a numpy
    img = cv2.imdecode(np.frombuffer(stream, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    # Switch the channel order
    if rgb_output:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
