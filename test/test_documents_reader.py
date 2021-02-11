import requests
import pytest
import fitz
import numpy as np
from io import BytesIO

from doctr.documents import reader


def test_convert_page_to_numpy(mock_pdf):
    pdf = fitz.open(mock_pdf)
    # Check correct read
    rgb_page = reader.convert_page_to_numpy(pdf[0])
    assert isinstance(rgb_page, np.ndarray)
    assert rgb_page.shape == (792, 612, 3)

    # Check channel order
    bgr_page = reader.convert_page_to_numpy(pdf[0], rgb_output=False)
    assert np.all(bgr_page == rgb_page[..., ::-1])

    # Check rescaling
    resized_page = reader.convert_page_to_numpy(pdf[0], output_size=(396, 306))
    assert resized_page.shape == (396, 306, 3)


def _check_pdf_content(doc_tensors):
    # 1 doc of 8 pages
    assert(len(doc_tensors) == 8)
    assert all(isinstance(page, np.ndarray) for page in doc_tensors)
    assert all(page.dtype == np.uint8 for page in doc_tensors)


def test_read_pdf(mock_pdf):
    doc_tensors = reader.read_pdf(mock_pdf)
    _check_pdf_content(doc_tensors)


def test_read_pdf_from_stream(mock_pdf_stream):
    doc_tensors = reader.read_pdf_from_stream(mock_pdf_stream)
    _check_pdf_content(doc_tensors)


def test_read_img(tmpdir_factory, mock_pdf):

    url = 'https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg'
    file = BytesIO(requests.get(url).content)
    tmp_path = str(tmpdir_factory.mktemp("data").join("mock_img_file.jpg"))
    with open(tmp_path, 'wb') as f:
        f.write(file.getbuffer())

    page = reader.read_img(tmp_path)

    # Data type
    assert isinstance(page, np.ndarray)
    assert page.dtype == np.uint8
    # Shape
    assert page.shape == (606, 517, 3)

    # RGB
    bgr_page = reader.read_img(tmp_path, rgb_output=False)
    assert np.all(page == bgr_page[..., ::-1])

    # Resize
    target_size = (200, 150)
    resized_page = reader.read_img(tmp_path, target_size)
    assert resized_page.shape[:2] == target_size

    # Non-existing file
    with pytest.raises(FileNotFoundError):
        reader.read_img("my_imaginary_file.jpg")
    # Invalid image
    with pytest.raises(ValueError):
        reader.read_img(str(mock_pdf))
