import pytest

import numpy as np
import torch
from doctr.models.preprocessor import PreProcessor


@pytest.mark.parametrize(
    "batch_size, output_size, input_tensor, expected_batches, expected_value",
    [
        [2, (128, 128), np.full((3, 256, 128, 3), 255, dtype=np.uint8), 1, .5],  # numpy uint8
        [2, (128, 128), np.ones((3, 256, 128, 3), dtype=np.float32), 1, .5],  # numpy fp32
        [2, (128, 128), torch.full((3, 3, 256, 128), 255, dtype=torch.uint8), 1, .5],  # torch uint8
        [2, (128, 128), torch.ones((3, 3, 256, 128), dtype=torch.float32), 1, .5],  # torch fp32
        [2, (128, 128), [np.full((256, 128, 3), 255, dtype=np.uint8)] * 3, 2, .5],  # list of numpy uint8
        [2, (128, 128), [np.ones((256, 128, 3), dtype=np.float32)] * 3, 2, .5],  # list of numpy fp32
        [2, (128, 128), [torch.full((3, 256, 128), 255, dtype=torch.uint8)] * 3, 2, .5],  # list of torch uint8
        [2, (128, 128), [torch.ones((3, 256, 128), dtype=torch.float32)] * 3, 2, .5],  # list of torch fp32
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
    # 3D check
    with pytest.raises(AssertionError):
        processor([np.full((3, 256, 128, 3), 255, dtype=np.uint8)])

    out = processor(input_tensor)
    assert isinstance(out, list) and len(out) == expected_batches
    assert all(isinstance(b, torch.Tensor) for b in out)
    assert all(b.dtype == torch.float32 for b in out)
    assert all(b.shape[-2:] == output_size for b in out)
    assert all(torch.all(b == expected_value) for b in out)
    assert len(repr(processor).split('\n')) == 4

    # Check FP16
    if isinstance(input_tensor, np.ndarray) and input_tensor.dtype == np.float32:
        with torch.no_grad():
            out = processor(input_tensor.astype(np.float16))
        assert all(elt.dtype == torch.float16 for elt in out)
    elif isinstance(input_tensor, torch.Tensor) and torch.cuda.is_available() and (input_tensor.dtype == torch.float32):
        input_tensor = input_tensor.to(dtype=torch.float16).cuda()
        processor = processor.cuda()
        with torch.no_grad():
            out = processor(input_tensor)
        assert all(elt.cpu().dtype == torch.float16 for elt in out)
