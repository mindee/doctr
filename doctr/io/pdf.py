# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
from pathlib import Path
import fitz
from typing import List, Tuple, Optional, Any, Dict

from doctr.utils.common_types import AbstractFile, Bbox

__all__ = ['read_pdf', 'PDF']


def read_pdf(file: AbstractFile, **kwargs: Any) -> fitz.Document:
    """Read a PDF file and convert it into an image in numpy format

    Example::
        >>> from doctr.documents import read_pdf
        >>> doc = read_pdf("path/to/your/doc.pdf")

    Args:
        file: the path to the PDF file
    Returns:
        the list of pages decoded as numpy ndarray of shape H x W x 3
    """

    if isinstance(file, (str, Path)) and not Path(file).is_file():
        raise FileNotFoundError(f"unable to access {file}")

    fitz_args: Dict[str, AbstractFile] = {}

    if isinstance(file, (str, Path)):
        fitz_args['filename'] = file
    elif isinstance(file, bytes):
        fitz_args['stream'] = file
    else:
        raise TypeError("unsupported object type for argument 'file'")

    # Read pages with fitz and convert them to numpy ndarrays
    return fitz.open(**fitz_args, filetype="pdf", **kwargs)


def convert_page_to_numpy(
    page: fitz.fitz.Page,
    output_size: Optional[Tuple[int, int]] = None,
    bgr_output: bool = False,
    default_scales: Tuple[float, float] = (2, 2),
) -> np.ndarray:
    """Convert a fitz page to a numpy-formatted image

    Args:
        page: the page of a file read with PyMuPDF
        output_size: the expected output size of each page in format H x W. Default goes to 840 x 595 for A4 pdf,
        if you want to increase the resolution while preserving the original A4 aspect ratio can pass (1024, 726)
        rgb_output: whether the output ndarray channel order should be RGB instead of BGR.
        default_scales: spatial scaling to be applied when output_size is not specified where (1, 1)
            corresponds to 72 dpi rendering.

    Returns:
        the rendered image in numpy format
    """

    # If no output size is specified, keep the origin one
    if output_size is not None:
        scales = (output_size[1] / page.MediaBox[2], output_size[0] / page.MediaBox[3])
    else:
        # Default 72 DPI (scales of (1, 1)) is unnecessarily low
        scales = default_scales

    transform_matrix = fitz.Matrix(*scales)

    # Generate the pixel map using the transformation matrix
    pixmap = page.getPixmap(matrix=transform_matrix)
    # Decode it into a numpy
    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)

    # Switch the channel order
    if bgr_output:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


class PDF:
    """PDF document template

    Args:
        doc: input PDF document
    """
    def __init__(self, doc: fitz.Document) -> None:
        self.doc = doc

    def as_images(self, **kwargs) -> List[np.ndarray]:
        """Convert all document pages to images

        Example::
            >>> from doctr.documents import DocumentFile
            >>> pages = DocumentFile.from_pdf("path/to/your/doc.pdf").as_images()

        Args:
            kwargs: keyword arguments of `convert_page_to_numpy`
        Returns:
            the list of pages decoded as numpy ndarray of shape H x W x 3
        """
        return [convert_page_to_numpy(page, **kwargs) for page in self.doc]

    def get_page_words(self, idx, **kwargs) -> List[Tuple[Bbox, str]]:
        """Get the annotations for all words of a given page"""

        # xmin, ymin, xmax, ymax, value, block_idx, line_idx, word_idx
        return [(info[:4], info[4]) for info in self.doc[idx].getTextWords(**kwargs)]

    def get_words(self, **kwargs) -> List[List[Tuple[Bbox, str]]]:
        """Get the annotations for all words in the document

        Example::
            >>> from doctr.documents import DocumentFile
            >>> words = DocumentFile.from_pdf("path/to/your/doc.pdf").get_words()

        Args:
            kwargs: keyword arguments of `fitz.Page.getTextWords`
        Returns:
            the list of pages annotations, represented as a list of tuple (bounding box, value)
        """
        return [self.get_page_words(idx, **kwargs) for idx in range(len(self.doc))]

    def get_page_artefacts(self, idx) -> List[Tuple[float, float, float, float]]:
        return [tuple(self.doc[idx].getImageBbox(artefact))  # type: ignore[misc]
                for artefact in self.doc[idx].get_images(full=True)]

    def get_artefacts(self) -> List[List[Tuple[float, float, float, float]]]:
        """Get the artefacts for the entire document

        Example::
            >>> from doctr.documents import DocumentFile
            >>> artefacts = DocumentFile.from_pdf("path/to/your/doc.pdf").get_artefacts()

        Returns:
            the list of pages artefacts, represented as a list of bounding boxes
        """

        return [self.get_page_artefacts(idx) for idx in range(len(self.doc))]
