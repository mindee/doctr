import numpy as np
import pytest
import tensorflow as tf

from doctr.models.preprocessor import PreProcessor


@pytest.mark.parametrize(
    "batch_size, output_size, input_tensor, expected_batches, expected_value",
    [
        [2, (128, 128), np.full((3, 256, 128, 3), 255, dtype=np.uint8), 1, 0.5],  # numpy uint8
        [2, (128, 128), np.ones((3, 256, 128, 3), dtype=np.float32), 1, 0.5],  # numpy fp32
        [2, (128, 128), tf.cast(tf.fill((3, 256, 128, 3), 255), dtype=tf.uint8), 1, 0.5],  # tf uint8
        [2, (128, 128), tf.ones((3, 128, 128, 3), dtype=tf.float32), 1, 0.5],  # tf fp32
        [2, (128, 128), [np.full((256, 128, 3), 255, dtype=np.uint8)] * 3, 2, 0.5],  # list of numpy uint8
        [2, (128, 128), [np.ones((256, 128, 3), dtype=np.float32)] * 3, 2, 0.5],  # list of numpy fp32
        [2, (128, 128), [tf.cast(tf.fill((256, 128, 3), 255), dtype=tf.uint8)] * 3, 2, 0.5],  # list of tf uint8
        [2, (128, 128), [tf.ones((128, 128, 3), dtype=tf.float32)] * 3, 2, 0.5],  # list of tf fp32
    ],
)
def test_preprocessor(batch_size, output_size, input_tensor, expected_batches, expected_value):
    processor = PreProcessor(output_size, batch_size)

    # Invalid input type
    with pytest.raises(TypeError):
        processor(42)
    # 4D check
    with pytest.raises(AssertionError):
        processor(np.full((256, 128, 3), 255, dtype=np.uint8))
    with pytest.raises(TypeError):
        processor(np.full((1, 256, 128, 3), 255, dtype=np.int32))
    # 3D check
    with pytest.raises(AssertionError):
        processor([np.full((3, 256, 128, 3), 255, dtype=np.uint8)])
    with pytest.raises(TypeError):
        processor([np.full((256, 128, 3), 255, dtype=np.int32)])

    out = processor(input_tensor)
    assert isinstance(out, list) and len(out) == expected_batches
    assert all(isinstance(b, tf.Tensor) for b in out)
    assert all(b.dtype == tf.float32 for b in out)
    assert all(b.shape[1:3] == output_size for b in out)
    assert all(tf.math.reduce_all(b == expected_value) for b in out)
    assert len(repr(processor).split("\n")) == 4
