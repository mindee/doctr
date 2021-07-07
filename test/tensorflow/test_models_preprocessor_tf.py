import pytest

import numpy as np
import tensorflow as tf
from doctr.models.preprocessor import PreProcessor


@pytest.mark.parametrize(
    "batch_size, output_size, input_tensor, expected_batches, expected_value",
    [
        [2, (128, 128), [np.full((256, 128, 3), 255, dtype=np.uint8)] * 3, 2, .5],  # numpy uint8
        [2, (128, 128), [np.ones((256, 128, 3), dtype=np.float32)] * 3, 2, .5],  # numpy float32
        [2, (128, 128), tf.ones((3, 128, 128, 3), dtype=tf.float32), 1, .5],  # tf correct size
        [2, (128, 128), tf.ones((3, 256, 128, 3), dtype=tf.float32), 1, .5],  # tf incorrect size
        [2, (128, 128), tf.cast(tf.fill((3, 256, 128, 3), 255), dtype=tf.uint8), 1, .5],  # tf uint8

    ],
)
def test_preprocessor(batch_size, output_size, input_tensor, expected_batches, expected_value):

    processor = PreProcessor(output_size, batch_size)

    # Invalid input type
    with pytest.raises(AssertionError):
        processor(42)

    out = processor(input_tensor)
    assert isinstance(out, list) and len(out) == expected_batches
    assert all(isinstance(b, tf.Tensor) for b in out)
    assert all(b.shape[1:3] == output_size for b in out)
    assert all(tf.math.reduce_all(b == expected_value) for b in out)
