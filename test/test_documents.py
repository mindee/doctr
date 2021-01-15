import requests
import pytest
import numpy as np
from io import BytesIO

from doctr import documents

DEFAULT_RES_MIN = int(0.8e6)
DEFAULT_RES_MAX = int(3e6)


@pytest.fixture(scope="session")
def mock_pdf(tmpdir_factory):
    url = 'https://arxiv.org/pdf/1911.08947.pdf'
    file = BytesIO(requests.get(url).content)
    fn = tmpdir_factory.mktemp("data").join("mock_pdf_file.pdf")
    with open(fn, 'wb') as f:
        f.write(file.getbuffer())
    return fn


def test_pdf_reader_with_pix(mock_pdf, num_pixels=2000000):
    documents_imgs, documents_names, documents_shapes = documents.reader.read_documents(
        filepaths=[mock_pdf],
        num_pixels=num_pixels)
    for doc_shapes, doc_images, doc_names in zip(documents_shapes, documents_img, documents_names):
        for shape, image, document_name in zip(doc_shapes, doc_images, doc_names):
            assert isinstance(shape, tuple)
            assert isinstance(document_name, str)
            assert isinstance(image, np.ndarray)
            assert shape[0] * shape[1] <= 1.003 * num_pixels
            assert shape[0] * shape[1] >= 0.997 * num_pixels


def test_pdf_reader(mock_pdf):
    documents_imgs, documents_names, documents_shapes = documents.reader.read_documents(
        filepaths=[mock_pdf],
        num_pixels=None)
    for doc_shapes, doc_images, doc_names in zip(documents_shapes, documents_img, documents_names):
        for shape, image, document_name in zip(doc_shapes, doc_images, doc_names):
            assert isinstance(shape, tuple)
            assert isinstance(document_name, str)
            assert isinstance(image, np.ndarray)
            assert shape[0] * shape[1] <= DEFAULT_RES_MAX
            assert shape[0] * shape[1] >= DEFAULT_RES_MIN


def test_pdf_reader(mock_pdf):
    shapes, raw_images, documents_names = documents.reader.read_documents(
        filepaths=[mock_pdf],
        num_pixels=None)
    for doc_shapes, doc_images, doc_names in zip(shapes, raw_images, documents_names):
        for shape, raw_image, document_name in zip(doc_shapes, doc_images, doc_names):
            assert isinstance(shape, tuple)
            assert isinstance(document_name, str)
            assert isinstance(raw_image, bytes)
            assert shape[0] * shape[1] <= DEFAULT_RES_MAX
            assert shape[0] * shape[1] >= DEFAULT_RES_MIN
