import pytest
import requests
from io import BytesIO

from doctr.models.preprocessor import Preprocessor
from doctr import documents
from test.test_documents import mock_pdf


def test_preprocess_documents(mock_pdf, num_docs=10, batch_size=3):
    docs = documents.reader.read_documents(
        filepaths=[mock_pdf for _ in range(num_docs)])
    preprocessor = Preprocessor(out_size=(600, 600), normalization=True, mode='symmetric', batch_size=batch_size)
    batched_docs, docs_indexes, pages_indexes = preprocessor(docs)
    assert len(docs_indexes) == len(pages_indexes)
    assert docs_indexes[-1] + 1 == num_docs
    if num_docs > batch_size:
        assert all(len(batch) == batch_size for batches in batched_docs[:-1] for batch in batches)
