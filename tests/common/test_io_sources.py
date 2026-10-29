import pickle
import threading
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from doctr import io
from doctr.io import sources
from doctr.io.elements import Page
from doctr.utils.multithreading import chunked, prefetch


@pytest.fixture(scope="module")
def image_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("scans")
    # Written in a non-sorted order, with a non-image file that must be ignored
    for name, value in [("page_2.png", 2), ("page_0.png", 0), ("page_1.jpg", 1)]:
        cv2.imwrite(str(folder / name), np.full((20 + value, 30, 3), 40 * value, dtype=np.uint8))
    (folder / "notes.txt").write_text("not an image")
    return folder


# Lazy sources


def test_pdf_source(mock_pdf):
    eager = io.read_pdf(mock_pdf)
    source = io.PDFSource(mock_pdf)

    assert isinstance(source, io.PageSource)
    assert len(source) == 2
    assert repr(source).startswith("PDFSource(")
    # pages are rendered on access, and identical to the eager reader
    assert all(np.array_equal(page, ref) for page, ref in zip(source, eager))
    assert np.array_equal(source[-1], eager[-1])
    assert source[0].dtype == np.uint8 and source[0].ndim == 3
    # Path & bytes inputs
    assert len(io.PDFSource(Path(mock_pdf))) == 2
    with open(mock_pdf, "rb") as f:
        assert np.array_equal(io.PDFSource(f.read())[1], eager[1])
    # Rendering options
    assert io.PDFSource(mock_pdf, scale=1)[0].shape[0] < eager[0].shape[0]

    with pytest.raises(IndexError):
        source[2]
    with pytest.raises(IndexError):
        source[-3]
    with pytest.raises(TypeError):
        source["0"]


def test_pdf_source_fails_eagerly(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.PDFSource("my_imaginary_file.pdf")
    with pytest.raises(TypeError):
        io.PDFSource(123)
    corrupted = tmp_path / "corrupted.pdf"
    corrupted.write_text("not a real pdf")
    with pytest.raises(Exception):
        io.PDFSource(corrupted)


def test_pdf_source_handles(mock_pdf, monkeypatch):
    opened = []
    _open = io.PDFSource._open

    def _counting_open(self):
        if self._pdf is None:
            opened.append(self)
        return _open(self)

    monkeypatch.setattr(io.PDFSource, "_open", _counting_open)
    source = io.PDFSource(mock_pdf)
    opened.clear()

    # no file handle is held outside of an iteration or a `with` block: random access opens the document for the
    # page only
    assert source._pdf is None
    assert source[0].ndim == 3 and source[1].ndim == 3
    assert source._pdf is None
    assert len(opened) == 2
    assert not sources._OPEN_PDF_SOURCES

    # a `with` block keeps it open for all the accesses of the block
    opened.clear()
    with source as doc:
        assert doc[0].ndim == 3 and doc[1].ndim == 3
        assert source._pdf is not None
        # nested blocks
        with source:
            assert doc[1].ndim == 3
        assert source._pdf is not None
    assert source._pdf is None
    assert len(opened) == 1

    # an iteration opens it once, and releases it at the end
    opened.clear()
    iterator = iter(source)
    next(iterator)
    assert source._pdf is not None
    assert len(list(iterator)) == 1
    assert source._pdf is None
    assert len(opened) == 1
    # also when stopped early
    iterator = iter(source)
    next(iterator)
    iterator.close()
    assert source._pdf is None

    # closing it within a `with` block or an iteration is not an issue: it is opened again on demand
    with source:
        source.close()
        assert source[0].ndim == 3
        assert source._pdf is not None
    assert source._pdf is None


def test_pdf_source_handle_released_on_errors(mock_pdf, monkeypatch):
    source = io.PDFSource(mock_pdf)

    def _failing_render(*args, **kwargs):
        raise RuntimeError("render failure")

    # a page that cannot be loaded
    with pytest.raises(Exception):
        source._load(5)
    assert source._pdf is None
    # a page that cannot be rendered
    monkeypatch.setattr(pdfium.PdfPage, "render", _failing_render)
    with pytest.raises(RuntimeError, match="render failure"):
        source[0]
    assert source._pdf is None
    with pytest.raises(RuntimeError, match="render failure"):
        list(source)
    assert source._pdf is None
    assert not sources._OPEN_PDF_SOURCES


def test_pdf_source_pickle(mock_pdf):
    source = io.PDFSource(mock_pdf)
    _ = source[0]
    restored = pickle.loads(pickle.dumps(source))
    assert restored._pdf is None
    assert np.array_equal(restored[0], source[0])


def test_pdf_source_thread_safety(mock_pdf):
    source = io.PDFSource(mock_pdf, scale=1)
    expected = [page.copy() for page in source]
    errors = []

    def _render():
        try:
            for idx in range(20):
                assert np.array_equal(source[idx % 2], expected[idx % 2])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=_render) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors


def test_image_source(image_folder):
    files = sorted(str(path) for path in image_folder.glob("page_*"))
    source = io.ImageSource(files)
    assert len(source) == 3
    assert [page.shape[0] for page in source] == [20, 21, 22]

    # a directory expands to its images, sorted by name
    folder_source = io.ImageSource(image_folder)
    assert [Path(file).name for file in folder_source.files] == ["page_0.png", "page_1.jpg", "page_2.png"]
    assert [page.shape[0] for page in folder_source] == [20, 21, 22]

    # single file, bytes & reader options
    assert len(io.ImageSource(files[0])) == 1
    raw = (image_folder / "page_0.png").read_bytes()
    assert io.ImageSource([raw, raw])[1].shape == (20, 30, 3)
    assert io.ImageSource(files, output_size=(8, 8))[2].shape == (8, 8, 3)

    # missing files and unsupported types fail when the source is built, not when the page is processed
    with pytest.raises(FileNotFoundError):
        io.ImageSource([files[0], "my_imaginary_file.png"])
    with pytest.raises(TypeError):
        io.ImageSource([123])


def test_image_source_invalid_image_names_the_file(image_folder, tmp_path):
    # files that are not images fail when the source is built, from their header only
    with pytest.raises(io.PageDecodeError, match="notes.txt"):
        io.ImageSource([image_folder / "page_0.png", image_folder / "notes.txt"])
    # including the files found in a directory
    (tmp_path / "page_0.png").write_bytes((image_folder / "page_0.png").read_bytes())
    (tmp_path / "page_1.png").write_text("not an image")
    with pytest.raises(io.PageDecodeError, match="page_1.png"):
        io.ImageSource(tmp_path)
    # still a `ValueError`, as raised by the eager readers
    with pytest.raises(ValueError):
        io.DocumentFile.from_images([image_folder / "notes.txt"])

    # a valid header with corrupted content is only detected when the image is decoded
    truncated = tmp_path / "page_1.png"
    truncated.write_bytes((image_folder / "page_0.png").read_bytes()[:60])
    source = io.ImageSource([image_folder / "page_0.png", truncated])
    assert source[0].ndim == 3
    with pytest.raises(io.PageDecodeError, match="page_1.png"):
        source[1]
    # bytes are not checked upfront
    with pytest.raises(io.PageDecodeError, match="<bytes #0>"):
        io.ImageSource([b"not an image"])[0]


def test_image_source_transparent_images(tmp_path):
    """Lazy sources decode with `read_img_as_numpy`: transparent pixels land on white, exactly as when eager"""
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)  # fully transparent black background
    rgba[5:15, 5:25] = (200, 10, 10, 255)  # opaque red box
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a_rgba.png")
    Image.fromarray(rgba, "RGBA").save(tmp_path / "b_rgba.webp", lossless=True)
    Image.fromarray(rgba, "RGBA").convert("P").save(tmp_path / "c_palette.png", transparency=0)

    lazy = io.DocumentFile.from_images(tmp_path, lazy=True)
    eager = io.DocumentFile.from_images(sorted(tmp_path.iterdir()))
    assert len(lazy) == 3
    for page, ref in zip(lazy, eager):
        assert np.array_equal(page, ref)
        assert page[0, 0].tolist() == [255, 255, 255]
        assert page[10, 10].tolist() == [200, 10, 10]

    # bytes inputs, mixed documents and decoding options go through the same path
    content = (tmp_path / "a_rgba.png").read_bytes()
    assert io.DocumentFile.from_files([content], lazy=True)[0][0, 0].tolist() == [255, 255, 255]
    assert io.ImageSource([content], rgb_output=False)[0][10, 10].tolist() == [10, 10, 200]


def test_slicing(mock_pdf):
    source = io.PDFSource(mock_pdf)
    view = source[1:]
    assert isinstance(view, io.PageSource)
    assert len(view) == 1
    assert np.array_equal(view[0], source[1])
    assert len(source[::-1]) == 2
    assert np.array_equal(source[::-1][0], source[1])
    assert len(source[5:]) == 0

    # a view holds nothing of its own: closing it leaves its source and the other views untouched
    with source:
        _ = source[0]
        view.close()
        assert source._pdf is not None
        assert np.array_equal(source[:1][0], source[0])
    assert source._pdf is None
    # iterating over a view opens its source for the pass only, `with view:` for the block
    assert len(list(source[::-1])) == 2
    assert source._pdf is None
    with view:
        _ = view[0]
        assert source._pdf is not None
    assert source._pdf is None
    # views of views
    assert np.array_equal(source[::-1][1:][0], source[0])
    # a source used in a `with` block can still be pickled
    with source:
        assert pickle.loads(pickle.dumps(source))._pdf is None
        assert len(pickle.loads(pickle.dumps(view))) == 1


def test_iter_batches(image_folder):
    source = io.ImageSource(image_folder)
    assert [len(batch) for batch in source.iter_batches(2)] == [2, 1]
    assert [len(batch) for batch in source.iter_batches(10)] == [3]
    with pytest.raises(ValueError):
        next(source.iter_batches(0))


def test_concat_source(mock_pdf, image_folder):
    pdf, images = io.PDFSource(mock_pdf), io.ImageSource(image_folder)
    in_memory = [np.zeros((5, 5, 3), dtype=np.uint8)]
    source = io.ConcatSource([pdf, images, in_memory])

    assert len(source) == 6
    assert repr(source) == "ConcatSource(num_sources=3, num_pages=6)"
    assert [page.shape for page in source][-1] == (5, 5, 3)
    assert source.locate(0) == (0, 0)
    assert source.locate(2) == (1, 0)
    assert source.locate(-1) == (2, 0)
    assert np.array_equal(source[3], images[1])
    with pytest.raises(IndexError):
        source.locate(6)

    # nested concatenations are flattened, `+` concatenates
    assert len(io.ConcatSource([source, pdf]).sources) == 4
    assert len(pdf + images) == 5
    assert len(in_memory + pdf) == 3
    assert isinstance(pdf + images, io.ConcatSource)
    assert len(io.ConcatSource([])) == 0
    assert list(io.ConcatSource([])) == []


def test_concat_source_keeps_one_pdf_open(mock_pdf):
    pdfs = [io.PDFSource(mock_pdf) for _ in range(3)]
    source = io.ConcatSource([pdfs[0], [np.zeros((4, 4, 3), dtype=np.uint8)], pdfs[1], pdfs[2]])
    # random access holds no handle
    for idx in range(len(source)):
        assert source[idx].ndim == 3
        assert all(pdf._pdf is None for pdf in pdfs)
    # an iteration holds at most one, and none once it is over
    for _ in source:
        assert sum(pdf._pdf is not None for pdf in pdfs) <= 1
    assert all(pdf._pdf is None for pdf in pdfs)
    # same for the views on a concatenation, in any order
    pages = list(source)
    view = source[::-1]
    for page, ref in zip(view, pages[::-1]):
        assert np.array_equal(page, ref)
        assert sum(pdf._pdf is not None for pdf in pdfs) <= 1
    assert all(pdf._pdf is None for pdf in pdfs)


def test_concat_source_concurrent_access(mock_pdf, monkeypatch):
    """Pages can be read (e.g. through the page images of predictions) while another thread iterates over the
    concatenation: no state is shared, and the document being iterated is not closed under the iteration"""
    pdfs = [io.PDFSource(mock_pdf) for _ in range(4)]
    source = io.ConcatSource(pdfs)
    expected = list(source)
    # documents closed while an iteration was still reading them
    closed_under_iteration = []
    _close = io.PDFSource.close

    def _recording_close(self):
        if self._holders > 0 and self._pdf is not None:
            closed_under_iteration.append(self)
        return _close(self)

    monkeypatch.setattr(io.PDFSource, "close", _recording_close)

    errors = []

    def _read_randomly():
        try:
            for _ in range(5):
                for idx in range(len(source)):
                    assert np.array_equal(source[idx], expected[idx])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    reader = threading.Thread(target=_read_randomly)
    reader.start()
    iterated = list(source)
    reader.join()
    assert not errors
    assert all(np.array_equal(page, ref) for page, ref in zip(iterated, expected))
    assert not closed_under_iteration
    assert all(pdf._pdf is None for pdf in pdfs)
    assert not sources._OPEN_PDF_SOURCES


# DocumentFile


def test_document_file_lazy(mock_pdf, image_folder):
    assert isinstance(io.DocumentFile.from_pdf(mock_pdf), list)
    assert isinstance(io.DocumentFile.from_pdf(mock_pdf, lazy=True), io.PDFSource)
    assert isinstance(io.DocumentFile.from_images(image_folder), list)
    lazy_images = io.DocumentFile.from_images(image_folder, lazy=True)
    assert isinstance(lazy_images, io.ImageSource)
    assert len(lazy_images) == 3
    # options are forwarded to the lazy source
    assert io.DocumentFile.from_pdf(mock_pdf, lazy=True, scale=1).scale == 1


def test_document_file_from_files(mock_pdf, image_folder, tmp_path):
    pdf_bytes = Path(mock_pdf).read_bytes()
    image = str(image_folder / "page_0.png")

    doc = io.DocumentFile.from_files([image, mock_pdf, pdf_bytes, image, image], lazy=True)
    assert isinstance(doc, io.ConcatSource)
    assert len(doc) == 7
    # consecutive images are grouped into a single source
    assert [type(source).__name__ for source in doc.sources] == ["ImageSource", "PDFSource", "PDFSource", "ImageSource"]
    assert doc.locate(6) == (3, 1)

    eager = io.DocumentFile.from_files([image, mock_pdf])
    assert isinstance(eager, list) and len(eager) == 3

    # directories expand to their PDFs and images, sorted by name
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "b.pdf").write_bytes(pdf_bytes)
    (folder / "a.png").write_bytes((image_folder / "page_0.png").read_bytes())
    (folder / "c.txt").write_text("ignored")
    doc = io.DocumentFile.from_files(folder, lazy=True, pdf_kwargs={"scale": 1}, image_kwargs={"output_size": (8, 8)})
    assert len(doc) == 3
    assert doc[0].shape == (8, 8, 3)
    assert doc.sources[1].scale == 1


def test_is_pdf(mock_pdf):
    pdf_bytes = Path(mock_pdf).read_bytes()
    assert sources._is_pdf(pdf_bytes)
    assert sources._is_pdf(b"\n \r\n" + pdf_bytes)
    assert not sources._is_pdf(b"\x89PNG\r\n")
    assert not sources._is_pdf(b"")
    # only the beginning of the content is looked at
    assert not sources._is_pdf(b" " * 2048 + pdf_bytes)
    assert sources._is_pdf(Path("doc.PDF")) and sources._is_pdf("doc.pdf")
    assert not sources._is_pdf("doc.png")


# Lazy page images


def test_page_image_on_demand(mock_pdf):
    source = io.PDFSource(mock_pdf)
    page = Page(sources._SourcePage(source, 1), [], 1, source[1].shape[:2])
    assert np.array_equal(page.page, source[1])
    # accessing the image of a prediction leaves no file handle behind
    assert source._pdf is None
    assert not sources._OPEN_PDF_SOURCES
    # assigning an array still works
    page.page = np.zeros((2, 2, 3), dtype=np.uint8)
    assert page.page.shape == (2, 2, 3)
    # picklable, so that results can be sent across processes
    restored = pickle.loads(pickle.dumps(sources._SourcePage(source, 0)))
    assert np.array_equal(restored(), source[0])


# Iteration helpers


def test_chunked():
    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(chunked([], 2)) == []
    # lazily consumed
    assert next(chunked(iter(range(10**9)), 3)) == [0, 1, 2]
    with pytest.raises(ValueError):
        next(chunked(range(3), 0))


@pytest.mark.parametrize("enabled", [True, False])
def test_prefetch(enabled):
    assert list(prefetch(range(5), enabled=enabled)) == list(range(5))
    assert list(prefetch([], enabled=enabled)) == []


def test_prefetch_runs_ahead_in_background():
    main_thread = threading.get_ident()
    produced = []

    def _gen():
        for idx in range(3):
            produced.append((idx, threading.get_ident() != main_thread))
            yield idx

    iterator = prefetch(_gen())
    assert next(iterator) == 0
    # the next element is already being produced in another thread
    assert next(iterator) == 1
    assert all(in_background for _, in_background in produced)
    assert list(iterator) == [2]


def test_prefetch_disabled_by_env(monkeypatch):
    monkeypatch.setenv("DOCTR_MULTIPROCESSING_DISABLE", "TRUE")
    main_thread = threading.get_ident()
    threads = []

    def _gen():
        for idx in range(2):
            threads.append(threading.get_ident())
            yield idx

    assert list(prefetch(_gen())) == [0, 1]
    assert set(threads) == {main_thread}


def test_prefetch_propagates_errors_and_closes():
    def _failing():
        yield 0
        raise RuntimeError("decoding failed")

    iterator = prefetch(_failing())
    assert next(iterator) == 0
    with pytest.raises(RuntimeError, match="decoding failed"):
        next(iterator)

    closed = []

    def _gen():
        try:
            yield from range(100)
        finally:
            closed.append(True)

    for item in prefetch(_gen()):
        if item == 2:
            break
    assert closed == [True]
