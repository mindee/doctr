# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import atexit
import threading
import weakref
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from itertools import accumulate
from pathlib import Path
from typing import Any, overload

import cv2
import numpy as np
import pypdfium2 as pdfium

from doctr.utils.common_types import AbstractFile

from .image import read_img_as_numpy

__all__ = ["PageSource", "PDFSource", "ImageSource", "ConcatSource", "IMAGE_EXTENSIONS"]

# File extensions recognized as images when a directory is expanded
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".jp2",
    ".pbm",
    ".pgm",
    ".ppm",
)

# pdfium is not thread-safe, not even across different documents: every call into it is serialized,
# which makes it safe to render pages in a background thread (cf. `OCRPredictor.stream`)
_PDFIUM_LOCK = threading.RLock()
# Sources holding an open PDF: they are closed at exit, before pypdfium2 reports the documents left open
_OPEN_PDF_SOURCES: "weakref.WeakSet[PDFSource]" = weakref.WeakSet()


@atexit.register
def _close_pdf_sources() -> None:  # pragma: no cover
    for source in list(_OPEN_PDF_SOURCES):
        source.close()


class PageSource(Sequence[np.ndarray]):
    """Abstract lazy collection of pages: a page is only decoded when it is accessed.

    A page source behaves like a read-only list of H x W x 3 numpy arrays (`len`, indexing, slicing, iteration),
    but only keeps the encoded inputs in memory. It can be passed to the predictors as-is: they process it in
    chunks, so that documents of any length are handled with a bounded memory footprint.

    >>> from doctr.io import DocumentFile
    >>> doc = DocumentFile.from_pdf("path/to/your/doc.pdf", lazy=True)
    >>> len(doc), doc[0].shape
    >>> for batch in doc.iter_batches(16):
    ...     ...
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def _load(self, idx: int) -> np.ndarray:
        """Decode the page at a given (non-negative, in-range) index"""
        raise NotImplementedError

    @overload
    def __getitem__(self, idx: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, idx: slice) -> "PageSource": ...

    def __getitem__(self, idx: int | slice) -> "np.ndarray | PageSource":
        if isinstance(idx, slice):
            return _PageView(self, range(len(self))[idx])
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"page indices must be integers or slices, not {type(idx).__name__}")
        num_pages = len(self)
        _idx = int(idx) + num_pages if idx < 0 else int(idx)
        if not 0 <= _idx < num_pages:
            raise IndexError(f"page index {idx} is out of range for a document of {num_pages} page(s)")
        return self._load(_idx)

    def __iter__(self) -> Iterator[np.ndarray]:
        for idx in range(len(self)):
            yield self._load(idx)

    def __add__(self, other: Sequence[np.ndarray]) -> "ConcatSource":
        if not isinstance(other, Sequence):
            return NotImplemented
        return ConcatSource([self, other])

    def __radd__(self, other: Sequence[np.ndarray]) -> "ConcatSource":
        if not isinstance(other, Sequence):
            return NotImplemented
        return ConcatSource([other, self])

    def iter_batches(self, batch_size: int) -> Iterator[list[np.ndarray]]:
        """Iterate over the pages by batches, only `batch_size` pages being decoded at a time

        Args:
            batch_size: the number of pages per batch

        Returns:
            an iterator of lists of pages
        """
        if batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer")
        batch: list[np.ndarray] = []
        for page in self:
            batch.append(page)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def close(self) -> None:
        """Release the resources held by the source (e.g. open PDF handles). The source can still be used
        afterwards: the resources are acquired again on demand."""

    def __enter__(self) -> "PageSource":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_pages={len(self)})"


class PDFSource(PageSource):
    """Lazy PDF document: pages are rasterized on access, so a 1000+ page PDF only costs the memory of
    the pages being processed.

    >>> from doctr.io import PDFSource
    >>> doc = PDFSource("path/to/your/doc.pdf", scale=2)
    >>> first_page = doc[0]

    Args:
        file: the path to the PDF file or its content as bytes
        scale: rendering scale (1 corresponds to 72dpi)
        rgb_mode: if True, the output will be RGB, otherwise BGR
        password: a password to unlock the document, if encrypted
        **kwargs: additional parameters to :meth:`pypdfium2.PdfPage.render`
    """

    def __init__(
        self,
        file: AbstractFile,
        scale: float = 2,
        rgb_mode: bool = True,
        password: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.file = file
        self.scale = scale
        self.rgb_mode = rgb_mode
        self.password = password
        self.render_kwargs = kwargs
        self._pdf: pdfium.PdfDocument | None = None
        # Open the document eagerly: invalid inputs fail here rather than in the middle of a prediction
        with _PDFIUM_LOCK:
            pdf = self._open()
            self._num_pages = len(pdf)
            # Do not hold a file handle per document: this matters when thousands of PDFs are concatenated
            self.close()

    def _open(self) -> pdfium.PdfDocument:
        if self._pdf is None:
            file = str(self.file) if isinstance(self.file, Path) else self.file
            self._pdf = pdfium.PdfDocument(file, password=self.password)
            _OPEN_PDF_SOURCES.add(self)
        return self._pdf

    def __len__(self) -> int:
        return self._num_pages

    def _load(self, idx: int) -> np.ndarray:
        with _PDFIUM_LOCK:
            page = self._open()[idx]
            try:
                bitmap = page.render(scale=self.scale, rev_byteorder=self.rgb_mode, **self.render_kwargs)
                # The bitmap buffer is allocated by Python: the array safely outlives the pdfium page & document
                return bitmap.to_numpy()
            finally:
                page.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        was_open = self._pdf is not None
        try:
            for idx in range(len(self)):
                yield self._load(idx)
        finally:
            # Release the file handle once the whole document has been read
            if not was_open:
                self.close()

    def close(self) -> None:
        with _PDFIUM_LOCK:
            if self._pdf is not None:
                self._pdf.close()
                self._pdf = None
            _OPEN_PDF_SOURCES.discard(self)

    def __getstate__(self) -> dict[str, Any]:
        # pdfium handles cannot be pickled: the document is re-opened on demand
        state = self.__dict__.copy()
        state["_pdf"] = None
        return state

    def __repr__(self) -> str:
        name = self.file if isinstance(self.file, (str, Path)) else "<bytes>"
        return f"{self.__class__.__name__}({name!s}, num_pages={len(self)})"


class ImageSource(PageSource):
    """Lazy collection of images: each image is only decoded when it is accessed, so that tens of thousands
    of files can be processed in a single call.

    >>> from doctr.io import ImageSource
    >>> doc = ImageSource(["path/to/page1.png", "path/to/page2.png"])
    >>> doc = ImageSource("path/to/a/folder")  # every image of the folder, sorted by name

    Args:
        files: the path to an image file, to a directory of images, or an image as bytes, or a collection of those
        **kwargs: additional parameters to :meth:`doctr.io.image.read_img_as_numpy`
    """

    def __init__(self, files: Sequence[AbstractFile] | AbstractFile, **kwargs: Any) -> None:
        self.files: list[AbstractFile] = []
        for file in [files] if isinstance(files, (str, Path, bytes)) else files:
            if isinstance(file, (str, Path)):
                path = Path(file)
                if path.is_dir():
                    self.files.extend(_list_files(path, IMAGE_EXTENSIONS))
                    continue
                if not path.is_file():
                    raise FileNotFoundError(f"unable to access {file}")
            elif not isinstance(file, bytes):
                raise TypeError(f"unsupported object type for an image file: {type(file).__name__}")
            self.files.append(file)
        self.read_kwargs = kwargs

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, idx: int) -> np.ndarray:
        file = self.files[idx]
        try:
            return read_img_as_numpy(file, **self.read_kwargs)
        except ValueError as e:
            # Images are decoded lazily: point to the faulty file among possibly thousands of them
            name = file if isinstance(file, (str, Path)) else f"<bytes #{idx}>"
            raise ValueError(f"unable to read the image {name!s}: {e}") from e


class ConcatSource(PageSource):
    """Concatenation of several page sources (or in-memory lists of pages) into a single lazy document.

    >>> from doctr.io import ConcatSource, PDFSource, ImageSource
    >>> doc = ConcatSource([PDFSource("report.pdf"), ImageSource("scans/")])
    >>> source_idx, page_idx = doc.locate(42)  # where does the 43rd page come from?

    Args:
        sources: the sequences of pages to concatenate
    """

    def __init__(self, sources: Iterable[Sequence[np.ndarray]]) -> None:
        self.sources: list[Sequence[np.ndarray]] = []
        for source in sources:
            # Flatten nested concatenations
            self.sources.extend(source.sources if isinstance(source, ConcatSource) else [source])
        self._offsets = [0, *accumulate(len(source) for source in self.sources)]
        self._active: Sequence[np.ndarray] | None = None

    def __len__(self) -> int:
        return self._offsets[-1]

    def locate(self, idx: int) -> tuple[int, int]:
        """Map a page index of the concatenation to the index of its source and its page index in that source

        Args:
            idx: the page index in the concatenation

        Returns:
            a tuple (index of the source in `sources`, index of the page in that source)
        """
        _idx = idx + len(self) if idx < 0 else idx
        if not 0 <= _idx < len(self):
            raise IndexError(f"page index {idx} is out of range for a document of {len(self)} page(s)")
        source_idx = bisect_right(self._offsets, _idx) - 1
        return source_idx, _idx - self._offsets[source_idx]

    def _switch_to(self, source: Sequence[np.ndarray]) -> None:
        # Keep at most one underlying document open at a time
        if self._active is not None and self._active is not source and isinstance(self._active, PageSource):
            self._active.close()
        self._active = source

    def _load(self, idx: int) -> np.ndarray:
        source_idx, page_idx = self.locate(idx)
        source = self.sources[source_idx]
        self._switch_to(source)
        return source[page_idx]

    def __iter__(self) -> Iterator[np.ndarray]:
        for source in self.sources:
            self._switch_to(source)
            yield from source

    def close(self) -> None:
        for source in self.sources:
            if isinstance(source, PageSource):
                source.close()
        self._active = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_sources={len(self.sources)}, num_pages={len(self)})"


class _PageView(PageSource):
    """Lazy view on a subset of the pages of another source (returned when slicing a source)"""

    def __init__(self, source: PageSource, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def _load(self, idx: int) -> np.ndarray:
        return self.source[self.indices[idx]]

    def close(self) -> None:
        self.source.close()


def _list_files(folder: Path, extensions: Sequence[str]) -> list[Path]:
    """List the files of a directory (non-recursively) with one of the given extensions, sorted by name"""
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.name,
    )


def _is_pdf(file: AbstractFile) -> bool:
    """Whether a file (path or bytes) is a PDF"""
    if isinstance(file, bytes):
        return file.lstrip()[:5] == b"%PDF-"
    return Path(file).suffix.lower() == ".pdf"


class _SourcePage:
    """Picklable reference to a page of a source, decoded on demand (used to keep predictions lightweight)"""

    __slots__ = ("source", "idx")

    def __init__(self, source: PageSource, idx: int) -> None:
        self.source = source
        self.idx = idx

    def __call__(self) -> np.ndarray:
        return self.source[self.idx]


class _EncodedImage:
    """Losslessly compressed in-memory image, decoded on demand (used for pages that cannot be re-read from their
    source as-is, e.g. straightened pages)"""

    __slots__ = ("data", "shape")

    def __init__(self, image: np.ndarray) -> None:
        self.shape = image.shape
        # Channel order is irrelevant here: the image is decoded exactly as it was encoded
        ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:  # pragma: no cover
            raise ValueError("unable to encode the page image")
        self.data = buffer.tobytes()

    def __call__(self) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(self.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:  # pragma: no cover
            raise ValueError("unable to decode the page image")
        return image.reshape(self.shape)
