import requests
import pytest
from io import BytesIO

from doctr import documents

DEFAULT_RES_MIN = int(0.8e6)
DEFAULT_RES_MAX = int(3e6)


@pytest.fixture(scope="module")
def mock_pdf():
    url = 'https://arxiv.org/pdf/1911.08947.pdf'
    file = BytesIO(requests.get(url).content)
    with open('mock_pdf_file.pdf', 'wb') as f:
        f.write(file.getbuffer())
    return 'mock_pdf_file.pdf'


def test_pdf_reader_with_pix(mock_pdf, num_pixels=2000000):
    shapes, raw_images, documents_names = documents.reader.document_reader(
        filepaths=[mock_pdf],
        num_pixels=num_pixels)
    for doc_shapes, doc_images, doc_names in zip(shapes, raw_images, documents_names):
        for shape, raw_image, document_name in zip(doc_shapes, doc_images, doc_names):
            assert isinstance(shape, tuple)
            assert isinstance(document_name, str)
            assert isinstance(raw_image, bytes)
            assert shape[0] * shape[1] <= 1.003 * num_pixels
            assert shape[0] * shape[1] >= 0.997 * num_pixels


def test_pdf_reader(mock_pdf):
    shapes, raw_images, documents_names = documents.reader.document_reader(
        filepaths=[mock_pdf],
        num_pixels=None)
    for doc_shapes, doc_images, doc_names in zip(shapes, raw_images, documents_names):
        for shape, raw_image, document_name in zip(doc_shapes, doc_images, doc_names):
            assert isinstance(shape, tuple)
            assert isinstance(document_name, str)
            assert isinstance(raw_image, bytes)
            assert shape[0] * shape[1] <= DEFAULT_RES_MAX
            assert shape[0] * shape[1] >= DEFAULT_RES_MIN
