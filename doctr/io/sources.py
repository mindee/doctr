# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import atexit
import threading
import weakref
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from itertools import accumulate, groupby
from pathlib import Path
from typing import Any, overload

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from doctr.utils.common_types import AbstractFile

from .image import read_img_as_numpy

__all__ = ["PageSource", "PDFSource", "ImageSource", "ConcatSource", "PageDecodeError", "IMAGE_EXTENSIONS"]

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


class PageDecodeError(ValueError):
    """Raised when a page of a source cannot be decoded (e.g. an unreadable or corrupted image file).

    It subclasses `ValueError`, so that code catching the errors of the eager readers keeps working.
    """


def _check_image_header(path: Path) -> None:
    """Fail fast on files that are not images: only their header is read (a few microseconds per file), so that
    an invalid file is reported when the source is created rather than after hours of processing. A file with a
    valid header but a corrupted body is still only detected when it is decoded.
    """
    if cv2.haveImageReader(str(path)):
        return
    # Formats read with Pillow by `read_img_as_numpy` (e.g. images with transparency)
    try:
        with Image.open(path):
            return
    except (OSError, ValueError, Image.DecompressionBombError):
        pass
    raise PageDecodeError(f"unable to read the image {path}: unknown or invalid image format")


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

    def _session(self) -> AbstractContextManager[Any]:
        """Context during which the resources needed to decode the pages (e.g. an open PDF) are kept.

        Iterations run within one, so that the resources are acquired once for the whole pass. Outside of it, an
        access to a single page (`source[idx]`) acquires them for that page only, and releases them right after.
        """
        return nullcontext()

    def _iter_indices(self, indices: Iterable[int]) -> Iterator[np.ndarray]:
        """Decode the pages at the given (non-negative, in-range) indices, in this order"""
        with self._session():
            for idx in indices:
                yield self._load(idx)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self._iter_indices(range(len(self)))

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
        afterwards: the resources are acquired again on demand.

        Resources are only held during an iteration or a `with` block: accessing a single page outside of those
        (e.g. `source[idx]`, or `page.page` on a prediction made from this source) releases them right after.
        """

    def __enter__(self) -> "PageSource":
        # Keep the resources for the whole block, e.g. for many random accesses to the pages of a PDF
        session = self._session()
        session.__enter__()
        self.__dict__.setdefault("_entered", []).append(session)
        return self

    def __exit__(self, *args: Any) -> None:
        entered = self.__dict__.get("_entered")
        if entered:
            entered.pop().__exit__(*args)
        if not entered:
            # Leaving the outermost block
            self._release()

    def _release(self) -> None:
        """Release the resources once the last `with` block on the source is left"""
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        # The resources held by a `with` block are not part of the source
        state = self.__dict__.copy()
        state.pop("_entered", None)
        return state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_pages={len(self)})"


class PDFSource(PageSource):
    """Lazy PDF document: pages are rasterized on access, so a 1000+ page PDF only costs the memory of
    the pages being processed.

    >>> from doctr.io import PDFSource
    >>> doc = PDFSource("path/to/your/doc.pdf", scale=2)
    >>> first_page = doc[0]
    >>> with doc:  # keep the document open for many random accesses
    ...     pages = [doc[idx] for idx in (0, 10, 500)]

    No file handle is held between accesses: an iteration opens the document once for the pass, and accessing a
    single page outside of an iteration or a `with` block opens it for that page only.

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
        # Number of sessions (iterations, `with` blocks) keeping the document open
        self._holders = 0
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

    @contextmanager
    def _session(self) -> Iterator[None]:
        with _PDFIUM_LOCK:
            self._holders += 1
        try:
            yield
        finally:
            with _PDFIUM_LOCK:
                self._holders -= 1
                # Release the file handle once nobody reads the document anymore
                if self._holders == 0:
                    self.close()

    def _load(self, idx: int) -> np.ndarray:
        with _PDFIUM_LOCK:
            # Outside of any session (e.g. `source[idx]`, or the page image of a prediction accessed later on), the
            # document is only opened for this page: no file handle is left behind. Within a session, it is kept open
            # (and opened again if it was closed in the meantime).
            transient = self._pdf is None and self._holders == 0
            pdf = self._open()
            try:
                page = pdf[idx]
                try:
                    bitmap = page.render(scale=self.scale, rev_byteorder=self.rgb_mode, **self.render_kwargs)
                    # The bitmap buffer is allocated by Python: the array safely outlives the pdfium page & document
                    return bitmap.to_numpy()
                finally:
                    page.close()
            finally:
                if transient:
                    self.close()

    def close(self) -> None:
        with _PDFIUM_LOCK:
            if self._pdf is not None:
                self._pdf.close()
                self._pdf = None
            _OPEN_PDF_SOURCES.discard(self)

    def _release(self) -> None:
        # The document is closed by the last session releasing it: an iteration still running in another thread
        # keeps it open
        pass

    def __getstate__(self) -> dict[str, Any]:
        # pdfium handles cannot be pickled: the document is re-opened on demand
        state = super().__getstate__()
        state["_pdf"] = None
        state["_holders"] = 0
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

    Missing files and files that are not images (checked from their header only) are reported when the source is
    created. An image that still cannot be decoded raises a :class:`~doctr.io.PageDecodeError` naming the file
    when it is accessed.

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
                    for image_path in _list_files(path, IMAGE_EXTENSIONS):
                        _check_image_header(image_path)
                        self.files.append(image_path)
                    continue
                if not path.is_file():
                    raise FileNotFoundError(f"unable to access {file}")
                _check_image_header(path)
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
            raise PageDecodeError(f"unable to read the image {name!s}: {e}") from e


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

    def _load(self, idx: int) -> np.ndarray:
        # No state is shared between accesses, so that pages can be read from several threads at once (e.g. the
        # page images of predictions while the predictor still iterates over the document in the background)
        source_idx, page_idx = self.locate(idx)
        return self.sources[source_idx][page_idx]

    def _iter_indices(self, indices: Iterable[int]) -> Iterator[np.ndarray]:
        # Consecutive pages of the same input are read within one session of that input: a PDF is opened once for
        # them and closed before moving on, so that at most one underlying document is open at a time
        for source_idx, run in groupby(indices, key=lambda idx: bisect_right(self._offsets, idx) - 1):
            source, offset = self.sources[source_idx], self._offsets[source_idx]
            if isinstance(source, PageSource):
                yield from source._iter_indices(idx - offset for idx in run)
            else:
                yield from (source[idx - offset] for idx in run)

    def close(self) -> None:
        for source in self.sources:
            if isinstance(source, PageSource):
                source.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_sources={len(self.sources)}, num_pages={len(self)})"


class _PageView(PageSource):
    """Lazy view on a subset of the pages of another source (returned when slicing a source)

    A view holds no resources of its own: iterating over it holds those of its source for the pass only, and
    closing it leaves its source (and the other views on it) untouched.
    """

    def __init__(self, source: PageSource, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def _load(self, idx: int) -> np.ndarray:
        return self.source[self.indices[idx]]

    def _session(self) -> AbstractContextManager[Any]:
        # `with view:` keeps the resources of the source for the block
        return self.source._session()

    def _iter_indices(self, indices: Iterable[int]) -> Iterator[np.ndarray]:
        return self.source._iter_indices(self.indices[idx] for idx in indices)


def _list_files(folder: Path, extensions: Sequence[str]) -> list[Path]:
    """List the files of a directory (non-recursively) with one of the given extensions, sorted by name"""
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.name,
    )


def _is_pdf(file: AbstractFile) -> bool:
    """Whether a file (path or bytes) is a PDF"""
    if isinstance(file, bytes):
        # Only look at the beginning of the content: stripping a whole document would copy it
        return file[:1024].lstrip()[:5] == b"%PDF-"
    return Path(file).suffix.lower() == ".pdf"


class _SourcePage:
    """Picklable reference to a page of a source, decoded on demand (used to keep predictions lightweight)"""

    __slots__ = ("source", "idx")

    def __init__(self, source: PageSource, idx: int) -> None:
        self.source = source
        self.idx = idx

    def __call__(self) -> np.ndarray:
        return self.source[self.idx]
