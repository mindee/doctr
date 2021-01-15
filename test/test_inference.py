import pytest

from doctr.models.detection import *
from doctr.documents.reader import *


@pytest.fixture(scope="session")
def mock_pdf(tmpdir_factory):
    url = 'https://arxiv.org/pdf/1911.08947.pdf'
    file = BytesIO(requests.get(url).content)
    fn = tmpdir_factory.mktemp("data").join("mock_pdf_file.pdf")
    with open(fn, 'wb') as f:
        f.write(file.getbuffer())
    return fn


def mock_infer_dict():
    mock_dict = 


def test_batch_documents(num_docs=10, batch_size=3):
    documents = documents.reader.read_documents(
        filepaths=[mock_pdf for _ in range(num_docs)])
    batched, docs_indexes, pages_indexes = batch_documents(documents, batch_size)
    assert len(docs_indexes) == len(pages_indexes)
    if num_docs > batch_size:
        assert len(batch[i]) == batch_size for batch in batched[:-1] for i in range(len(batch))


def test_inference():


    return infer_dict


def test_show_infer_pdf(num_docs=3):
    show_infer_pdf(
    filepaths=[mock_pdf for _ in range(num_docs)], infer_dict=mock_infer_dict)
 

infer_dict = inference(path_to_model, documents)
show_infer_pdf(filepaths, infer_dict)