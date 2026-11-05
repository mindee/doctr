# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, overload

import numpy as np

from doctr.file_utils import requires_package
from doctr.utils.common_types import AbstractFile

from .html import read_html
from .sources import (
    _PAGE_ERRORS,
    IMAGE_EXTENSIONS,
    ConcatSource,
    ImageSource,
    PageSource,
    PDFSource,
    _file_name,
    _is_pdf,
    _list_files,
    _UnreadablePage,
)

__all__ = ["DocumentFile"]

logger = logging.getLogger(__name__)

# What to do with a file that cannot be read (missing, not an image, a corrupted PDF...): raise the error, skip it
# with a warning, or skip it and call a function with the file and the error
OnFileError: TypeAlias = Literal["raise", "skip"] | Callable[[AbstractFile, Exception], Any]


def _skip_file(on_error: OnFileError, file: AbstractFile, error: Exception) -> None:
    """Report a file that cannot be read, and is left out of the document"""
    if callable(on_error):
        on_error(file, error)
    else:
        logger.warning(f"Skipping {_file_name(file)}, which cannot be read: {error}")


def _file_at(doc: ConcatSource, idx: int) -> AbstractFile:
    """The input file of a page of a concatenation built by `DocumentFile.from_files`"""
    source_idx, page_idx = doc.locate(idx)
    source = doc.sources[source_idx]
    return source.files[page_idx] if isinstance(source, ImageSource) else source.file  # type: ignore[attr-defined]


def _materialize(source: PageSource) -> list[np.ndarray]:
    """Decode every page of a source into memory"""
    try:
        return list(source)
    finally:
        source.close()


class DocumentFile:
    """Read a document from multiple extensions

    Every reader decodes all the pages into a list of numpy arrays by default. Pass `lazy=True` to get a
    :class:`~doctr.io.PageSource` instead: it behaves like that list, but a page is only decoded when it is
    accessed. Predictors process page sources chunk by chunk, so that very long documents (1000+ pages) or
    huge collections of images can be passed in a single call with a bounded memory footprint.
    """

    @overload
    @classmethod
    def from_pdf(cls, file: AbstractFile, lazy: Literal[False] = ..., **kwargs: Any) -> list[np.ndarray]: ...

    @overload
    @classmethod
    def from_pdf(cls, file: AbstractFile, lazy: Literal[True], **kwargs: Any) -> PDFSource: ...

    @overload
    @classmethod
    def from_pdf(cls, file: AbstractFile, lazy: bool, **kwargs: Any) -> list[np.ndarray] | PDFSource: ...

    @classmethod
    def from_pdf(cls, file: AbstractFile, lazy: bool = False, **kwargs: Any) -> list[np.ndarray] | PDFSource:
        """Read a PDF file

        >>> from doctr.io import DocumentFile
        >>> doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
        >>> # Pages are rendered on demand: suited for very long documents
        >>> doc = DocumentFile.from_pdf("path/to/your/long_doc.pdf", lazy=True)

        Args:
            file: the path to the PDF file or a binary stream
            lazy: if True, return a :class:`~doctr.io.PDFSource` rendering the pages on demand
            **kwargs: additional parameters to :class:`~doctr.io.PDFSource` (`scale`, `rgb_mode`, `password`)
                and :meth:`pypdfium2.PdfPage.render`

        Returns:
            the list of pages decoded as numpy ndarray of shape H x W x 3, or a lazy source if `lazy` is True
        """
        source = PDFSource(file, **kwargs)
        return source if lazy else _materialize(source)

    @overload
    @classmethod
    def from_url(cls, url: str, lazy: Literal[False] = ..., **kwargs: Any) -> list[np.ndarray]: ...

    @overload
    @classmethod
    def from_url(cls, url: str, lazy: Literal[True], **kwargs: Any) -> PDFSource: ...

    @overload
    @classmethod
    def from_url(cls, url: str, lazy: bool, **kwargs: Any) -> list[np.ndarray] | PDFSource: ...

    @classmethod
    def from_url(cls, url: str, lazy: bool = False, **kwargs: Any) -> list[np.ndarray] | PDFSource:
        """Interpret a web page as a PDF document

        >>> from doctr.io import DocumentFile
        >>> doc = DocumentFile.from_url("https://www.yoursite.com")

        Args:
            url: the URL of the target web page
            lazy: if True, return a :class:`~doctr.io.PDFSource` rendering the pages on demand
            **kwargs: additional parameters to :class:`~doctr.io.PDFSource` and :meth:`pypdfium2.PdfPage.render`

        Returns:
            the list of pages decoded as numpy ndarray of shape H x W x 3, or a lazy source if `lazy` is True
        """
        requires_package(
            "weasyprint",
            "`.from_url` requires weasyprint installed.\n"
            + "Installation instructions: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation",
        )
        pdf_stream = read_html(url)
        source = PDFSource(pdf_stream, **kwargs)
        return source if lazy else _materialize(source)

    @overload
    @classmethod
    def from_images(
        cls, files: Sequence[AbstractFile] | AbstractFile, lazy: Literal[False] = ..., **kwargs: Any
    ) -> list[np.ndarray]: ...

    @overload
    @classmethod
    def from_images(
        cls, files: Sequence[AbstractFile] | AbstractFile, lazy: Literal[True], **kwargs: Any
    ) -> ImageSource: ...

    @overload
    @classmethod
    def from_images(
        cls, files: Sequence[AbstractFile] | AbstractFile, lazy: bool, **kwargs: Any
    ) -> list[np.ndarray] | ImageSource: ...

    @classmethod
    def from_images(
        cls, files: Sequence[AbstractFile] | AbstractFile, lazy: bool = False, **kwargs: Any
    ) -> list[np.ndarray] | ImageSource:
        """Read an image file (or a collection of image files) and convert it into an image in numpy format

        >>> from doctr.io import DocumentFile
        >>> pages = DocumentFile.from_images(["path/to/your/page1.png", "path/to/your/page2.png"])
        >>> # Every image of a folder (sorted by name), decoded on demand
        >>> pages = DocumentFile.from_images("path/to/your/folder", lazy=True)

        Args:
            files: the path to an image file, to a directory of images, or an image as bytes, or a collection of those
            lazy: if True, return a :class:`~doctr.io.ImageSource` decoding the images on demand
            **kwargs: additional parameters to :meth:`doctr.io.image.read_img_as_numpy`

        Returns:
            the list of pages decoded as numpy ndarray of shape H x W x 3, or a lazy source if `lazy` is True
        """
        source = ImageSource(files, **kwargs)
        return source if lazy else _materialize(source)

    @overload
    @classmethod
    def from_files(
        cls,
        files: Sequence[AbstractFile] | AbstractFile,
        lazy: Literal[False] = ...,
        pdf_kwargs: dict[str, Any] | None = ...,
        image_kwargs: dict[str, Any] | None = ...,
        on_error: OnFileError = ...,
    ) -> list[np.ndarray]: ...

    @overload
    @classmethod
    def from_files(
        cls,
        files: Sequence[AbstractFile] | AbstractFile,
        lazy: Literal[True],
        pdf_kwargs: dict[str, Any] | None = ...,
        image_kwargs: dict[str, Any] | None = ...,
        on_error: OnFileError = ...,
    ) -> ConcatSource: ...

    @overload
    @classmethod
    def from_files(
        cls,
        files: Sequence[AbstractFile] | AbstractFile,
        lazy: bool,
        pdf_kwargs: dict[str, Any] | None = ...,
        image_kwargs: dict[str, Any] | None = ...,
        on_error: OnFileError = ...,
    ) -> list[np.ndarray] | ConcatSource: ...

    @classmethod
    def from_files(
        cls,
        files: Sequence[AbstractFile] | AbstractFile,
        lazy: bool = False,
        pdf_kwargs: dict[str, Any] | None = None,
        image_kwargs: dict[str, Any] | None = None,
        on_error: OnFileError = "raise",
    ) -> list[np.ndarray] | ConcatSource:
        """Read any mix of PDFs, images, directories and URLs as a single document, in the given order

        >>> from doctr.io import DocumentFile
        >>> doc = DocumentFile.from_files(["cover.png", "report.pdf", "path/to/scans/"], lazy=True)
        >>> source_idx, page_idx = doc.locate(42)  # the source (in `doc.sources`) of the 43rd page

        Args:
            files: a path (to a PDF, an image or a directory), a URL, a PDF or an image as bytes,
                or a collection of those. Directories are expanded (non-recursively) to the PDFs and images they
                contain, sorted by name.
            lazy: if True, return a :class:`~doctr.io.ConcatSource` decoding the pages on demand
            pdf_kwargs: keyword arguments of :class:`~doctr.io.PDFSource`, applied to every PDF and URL
            image_kwargs: keyword arguments of :meth:`doctr.io.image.read_img_as_numpy`, applied to every image
            on_error: what to do with a file that cannot be read (missing, not an image, a corrupted PDF...): "raise"
                (default) the error, "skip" the file with a warning, or a callable to skip it and call with the file
                and the error. Images are checked from their header: a corrupted content is only detected when the
                image is decoded, i.e. here if `lazy` is False, and by the predictor otherwise (cf. the `on_error`
                argument of :meth:`~doctr.models.predictor.OCRPredictor.stream`).

        Returns:
            the list of pages decoded as numpy ndarray of shape H x W x 3, or a lazy source if `lazy` is True
        """
        if not (on_error in ("raise", "skip") or callable(on_error)):
            raise ValueError(f"`on_error` must be 'raise', 'skip' or a callable, got {on_error!r}")
        pdf_kwargs = pdf_kwargs or {}
        image_kwargs = image_kwargs or {}

        def _readable(file: AbstractFile, open_file: Callable[[], Any]) -> Any:
            # The opened file, or None if it cannot be read and errors are skipped
            try:
                return open_file()
            except _PAGE_ERRORS as e:
                if on_error == "raise":
                    raise
                _skip_file(on_error, file, e)
                return None

        inputs: list[AbstractFile] = []
        for file in [files] if isinstance(files, (str, Path, bytes)) else files:
            if isinstance(file, (str, Path)) and Path(file).is_dir():
                inputs.extend(_list_files(Path(file), (".pdf", *IMAGE_EXTENSIONS)))
            else:
                inputs.append(file)

        sources: list[PageSource] = []
        images: list[AbstractFile] = []

        def _flush_images() -> None:
            # Consecutive images are grouped into a single source
            if images:
                sources.append(ImageSource(list(images), **image_kwargs))
                images.clear()

        for file in inputs:
            if isinstance(file, str) and file.lower().startswith(("http://", "https://")):
                _flush_images()
                sources.append(cls.from_url(file, lazy=True, **pdf_kwargs))
            elif isinstance(file, (str, Path, bytes)) and _is_pdf(file):
                _flush_images()
                pdf = _readable(file, lambda: PDFSource(file, **pdf_kwargs))
                if pdf is not None:
                    sources.append(pdf)
            elif on_error == "raise" or _readable(file, lambda: ImageSource([file], **image_kwargs)) is not None:
                # When skipping errors, every image is checked on its own, so that one bad file only skips itself
                images.append(file)
        _flush_images()

        source = ConcatSource(sources)
        if lazy:
            return source
        if on_error == "raise":
            return _materialize(source)
        # Decode every page, skipping those that cannot be read
        pages: list[np.ndarray] = []
        try:
            for idx, page in enumerate(source._iter_indices(range(len(source)), skip_errors=True)):
                if isinstance(page, _UnreadablePage):
                    _skip_file(on_error, _file_at(source, idx), page.error)
                else:
                    pages.append(page)
        finally:
            source.close()
        return pages
