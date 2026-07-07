from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from doctr import io


def _check_doc_content(doc_tensors, num_pages):
    # 1 doc of 8 pages
    assert len(doc_tensors) == num_pages
    assert all(isinstance(page, np.ndarray) for page in doc_tensors)
    assert all(page.dtype == np.uint8 for page in doc_tensors)


def test_read_pdf(mock_pdf):
    doc = io.read_pdf(mock_pdf)
    _check_doc_content(doc, 2)

    # Test with Path
    doc = io.read_pdf(Path(mock_pdf))
    _check_doc_content(doc, 2)

    with open(mock_pdf, "rb") as f:
        doc = io.read_pdf(f.read())
    _check_doc_content(doc, 2)

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

    assert isinstance(io.DocumentFile.from_pdf(mock_pdf), list)
    assert isinstance(io.DocumentFile.from_url("https://www.google.com"), list)


def test_pdf(mock_pdf):
    pages = io.DocumentFile.from_pdf(mock_pdf)

    # As images
    num_pages = 2
    _check_doc_content(pages, num_pages)


def test_read_img_as_numpy_exif_orientation(tmpdir_factory):
    # A JPEG with EXIF orientation 6 (90° clockwise display rotation)
    folder = tmpdir_factory.mktemp("images")
    path = str(folder.join("exif_o6.jpg"))
    pil_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
    exif = pil_img.getexif()
    exif[0x0112] = 6  # orientation tag
    pil_img.save(path, format="JPEG", exif=exif)

    assert io.read_img_as_numpy(path).shape == (60, 40, 3)
    with open(path, "rb") as f:
        assert io.read_img_as_numpy(f.read()).shape == (60, 40, 3)
