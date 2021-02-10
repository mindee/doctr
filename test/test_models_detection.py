import pytest
import math
import numpy as np
import tensorflow as tf

from doctr.models import detection
from doctr.documents import read_pdf


def test_detpreprocessor(mock_pdf):  # noqa: F811
    num_docs = 3
    batch_size = 4
    docs = [read_pdf(mock_pdf) for _ in range(num_docs)]
    processor = detection.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])

    # Number of batches
    assert len(batched_docs) == math.ceil(8 * num_docs / batch_size)
    # Total number of samples
    assert sum(batch.shape[0] for batch in batched_docs) == 8 * num_docs
    # Batch size
    assert all(batch.shape[0] == batch_size for batch in batched_docs[:-1])
    assert batched_docs[-1].shape[0] == batch_size if (8 * num_docs) % batch_size == 0 else (8 * num_docs) % batch_size
    # Data type
    assert all(batch.dtype == tf.float32 for batch in batched_docs)
    # Image size
    assert all(batch.shape[1:] == (512, 512, 3) for batch in batched_docs)
    # Test with non-full last batch
    batch_size = 16
    processor = detection.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])
    assert batched_docs[-1].shape[0] == (8 * num_docs) % batch_size


def test_dbpostprocessor():
    postprocessor = detection.DBPostProcessor()
    mock_batch = tf.random.uniform(shape=[2, 512, 512, 1], minval=0, maxval=1)
    out = postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, np.ndarray) for sample in out)
    assert all(sample.shape[1] == 5 for sample in out)
    # Relative coords
    assert all(np.all(np.logical_and(sample[:4] >= 0, sample[:4] <= 1)) for sample in out)


def test_db_resnet50():
    model = detection.db_resnet50(pretrained=True)
    assert isinstance(model, tf.keras.Model)
    dbinput = tf.random.uniform(shape=[2, 1024, 1024, 3], minval=0, maxval=1)
    # test prediction model
    dboutput_notrain = model(dbinput)
    assert isinstance(dboutput_notrain, tf.Tensor)
    assert dboutput_notrain.numpy().shape == (2, 1024, 1024, 1)
    assert np.all(dboutput_notrain.numpy() > 0) and np.all(dboutput_notrain.numpy() < 1)
    # test training model
    dboutput_train = model(dbinput, training=True)
    assert isinstance(dboutput_train, tuple)
    assert len(dboutput_train) == 3
    assert all(np.all(np.logical_and(out_map.numpy() >= 0, out_map.numpy() <= 1)) for out_map in dboutput_train)
    # batch size
    assert all(out.numpy().shape == (2, 1024, 1024, 1) for out in dboutput_train)


@pytest.fixture(scope="session")
def test_detectionpredictor(mock_pdf):  # noqa: F811

    batch_size = 4
    predictor = detection.DetectionPredictor(
        detection.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size),
        detection.db_resnet50(input_shape=(512, 512, 3)),
        detection.DBPostProcessor()
    )

    pages = read_pdf(mock_pdf)
    out = predictor(pages)

    # The input PDF has 8 pages
    assert len(out) == 8

    return predictor
