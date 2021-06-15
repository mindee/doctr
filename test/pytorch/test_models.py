import pytest
import math
import numpy as np
import torch

from doctr import models
from doctr.documents import DocumentFile


def test_preprocessor(mock_pdf):
    preprocessor = models.PreProcessor(output_size=(1024, 1024), batch_size=2)
    input_tensor = torch.rand((2, 3, 512, 512))
    preprocessed = preprocessor(input_tensor)
    assert isinstance(preprocessed, list)
    for batch in preprocessed:
        assert batch.shape[0] == 2
        assert batch.shape[-2:] == (1024, 1024)

    with pytest.raises(AssertionError):
        _ = preprocessor(np.random.rand(3, 1024, 1024))

    num_docs = 3
    batch_size = 4
    docs = [DocumentFile.from_pdf(mock_pdf).as_images() for _ in range(num_docs)]
    processor = models.PreProcessor(output_size=(512, 512), batch_size=batch_size)
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
    assert all(batch.shape[1:] == (3, 512, 512) for batch in batched_docs)
    # Test with non-full last batch
    batch_size = 16
    processor = models.PreProcessor(output_size=(512, 512), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])
    assert batched_docs[-1].shape[0] == (8 * num_docs) % batch_size

    # Repr
    assert len(repr(processor).split('\n')) == 9

    # Assymetric
    processor = models.PreProcessor(output_size=(256, 128), batch_size=batch_size, preserve_aspect_ratio=True)
    batched_docs = processor([page for doc in docs for page in doc])
    # Image size
    assert all(batch.shape[1:] == (3, 256, 128) for batch in batched_docs)
