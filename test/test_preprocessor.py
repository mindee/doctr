import pytest
import requests
from io import BytesIO

from doctr.models.preprocessor import Preprocessor
from doctr.documents.reader import read_documents


@pytest.fixture(scope="session")
def mock_pdf(tmpdir_factory):
    url = 'https://arxiv.org/pdf/1911.08947.pdf'
    file = BytesIO(requests.get(url).content)
    fn = tmpdir_factory.mktemp("data").join("mock_pdf_file.pdf")
    with open(fn, 'wb') as f:
        f.write(file.getbuffer())
    return fn


def test_preprocess_documents(num_docs=10, batch_size=3):
    documents = read_documents(
        filepaths=[mock_pdf for _ in range(num_docs)])
    preprocessor = Preprocessor(out_size=(600, 600), normalization=True, batch_size=batch_size)
    batched_docs, docs_indexes, pages_indexes = preprocessor(documents)
    assert len(docs_indexes) == len(pages_indexes)
    if num_docs > batch_size:
        for batch in batched_docs[:-1]:
            for i in range(len(batch)):
                assert len(batch[i]) == batch_size
