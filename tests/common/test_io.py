from io import BytesIO

import numpy as np
import pytest
import requests

from doctr import io


def _check_doc_content(doc_tensors, num_pages):
    i = 0
    for page in doc_tensors:
        assert isinstance(page, np.ndarray)
        assert page.dtype == np.uint8
        i += 1
    assert i == num_pages == len(doc_tensors)


def test_read_pdf(mock_pdf):
    doc = io.PdfRenderer(mock_pdf)

    # length can be accessed without actually rendering the pdf
    assert len(doc) == 2

    _check_doc_content(doc, 2)

    with open(mock_pdf, "rb") as f:
        doc = io.PdfRenderer(f.read())
    _check_doc_content(doc, 2)

    # Wrong input type
    with pytest.raises(TypeError):
        _ = next(io.PdfRenderer(123))

    # Wrong path
    with pytest.raises(FileNotFoundError):
        _ = next(io.PdfRenderer("my_imaginary_file.pdf"))


def test_read_img_as_numpy(tmpdir_factory, mock_pdf):

    # Wrong input type
    with pytest.raises(TypeError):
        _ = io.read_img_as_numpy(123)

    # Non-existing file
    with pytest.raises(FileNotFoundError):
        io.read_img_as_numpy("my_imaginary_file.jpg")

    # Invalid image
    with pytest.raises(ValueError):
        io.read_img_as_numpy(str(mock_pdf))

    # From path
    url = "https://doctr-static.mindee.com/models?id=v0.2.1/Grace_Hopper.jpg&src=0"
    file = BytesIO(requests.get(url).content)
    tmp_path = str(tmpdir_factory.mktemp("data").join("mock_img_file.jpg"))
    with open(tmp_path, "wb") as f:
        f.write(file.getbuffer())

    # Path & stream
    with open(tmp_path, "rb") as f:
        page_stream = io.read_img_as_numpy(f.read())

    for page in (io.read_img_as_numpy(tmp_path), page_stream):
        # Data type
        assert isinstance(page, np.ndarray)
        assert page.dtype == np.uint8
        # Shape
        assert page.shape == (606, 517, 3)

    # RGB
    bgr_page = io.read_img_as_numpy(tmp_path, rgb_output=False)
    assert np.all(page == bgr_page[..., ::-1])

    # Resize
    target_size = (200, 150)
    resized_page = io.read_img_as_numpy(tmp_path, target_size)
    assert resized_page.shape[:2] == target_size


def test_read_html():
    url = "https://www.google.com"
    pdf_stream = io.read_html(url)
    assert isinstance(pdf_stream, bytes)


def test_document_file(mock_pdf, mock_image_stream):
    pages = io.DocumentFile.from_images(mock_image_stream)
    _check_doc_content(pages, 1)

    assert isinstance(io.DocumentFile.from_pdf(mock_pdf), io.PdfRenderer)
    assert isinstance(io.DocumentFile.from_url("https://www.google.com"), io.PdfRenderer)


def test_pdf(mock_pdf):
    pages = io.DocumentFile.from_pdf(mock_pdf)
    _check_doc_content(pages, 2)
