from io import BytesIO

import fitz
import numpy as np
import pytest
import requests

from doctr import io


def test_convert_page_to_numpy(mock_pdf):
    pdf = fitz.open(mock_pdf)
    # Check correct read
    rgb_page = io.pdf.convert_page_to_numpy(pdf[0], default_scales=(1, 1))
    assert isinstance(rgb_page, np.ndarray)
    assert rgb_page.shape == (792, 612, 3)

    # Check channel order
    bgr_page = io.pdf.convert_page_to_numpy(pdf[0], default_scales=(1, 1), bgr_output=True)
    assert np.all(bgr_page == rgb_page[..., ::-1])

    # Check resizing
    resized_page = io.pdf.convert_page_to_numpy(pdf[0], output_size=(396, 306))
    assert resized_page.shape == (396, 306, 3)

    # Check rescaling
    rgb_page = io.pdf.convert_page_to_numpy(pdf[0])
    assert isinstance(rgb_page, np.ndarray)
    assert rgb_page.shape == (1584, 1224, 3)


def _check_doc_content(doc_tensors, num_pages):
    # 1 doc of 8 pages
    assert(len(doc_tensors) == num_pages)
    assert all(isinstance(page, np.ndarray) for page in doc_tensors)
    assert all(page.dtype == np.uint8 for page in doc_tensors)


def test_read_pdf(mock_pdf, mock_pdf_stream):
    for file in [mock_pdf, mock_pdf_stream]:
        doc = io.read_pdf(file)
        assert isinstance(doc, fitz.Document)

    # Wrong input type
    with pytest.raises(TypeError):
        _ = io.read_pdf(123)

    # Wrong path
    with pytest.raises(FileNotFoundError):
        _ = io.read_pdf("my_imaginary_file.pdf")


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
    url = 'https://github.com/mindee/doctr/releases/download/v0.2.1/Grace_Hopper.jpg'
    file = BytesIO(requests.get(url).content)
    tmp_path = str(tmpdir_factory.mktemp("data").join("mock_img_file.jpg"))
    with open(tmp_path, 'wb') as f:
        f.write(file.getbuffer())

    # Path & stream
    with open(tmp_path, 'rb') as f:
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

    assert isinstance(io.DocumentFile.from_pdf(mock_pdf).doc, fitz.Document)
    assert isinstance(io.DocumentFile.from_url("https://www.google.com").doc, fitz.Document)


def test_pdf(mock_pdf):

    doc = io.DocumentFile.from_pdf(mock_pdf)

    # As images
    pages = doc.as_images()
    _check_doc_content(pages, 8)

    # Get words
    words = doc.get_words()
    assert isinstance(words, list) and len(words) == 8
    assert all(isinstance(bbox, tuple) and isinstance(value, str)
               for page_words in words for (bbox, value) in page_words)
    assert all(all(isinstance(coord, float) for coord in bbox) for page_words in words for (bbox, value) in page_words)

    # Get artefacts
    artefacts = doc.get_artefacts()
    assert isinstance(artefacts, list) and len(artefacts) == 8
    assert all(isinstance(bbox, tuple) for page_artefacts in artefacts for bbox in page_artefacts)
    assert all(all(isinstance(coord, float) for coord in bbox)
               for page_artefacts in artefacts for bbox in page_artefacts)
