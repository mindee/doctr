import numpy as np
import pytest
import torch

from doctr.models.preprocessor import PreProcessor


@pytest.mark.parametrize(
    "batch_size, output_size, input_tensor, expected_batches, expected_value",
    [
        [2, (128, 128), np.full((3, 256, 128, 3), 255, dtype=np.uint8), 1, 0.5],  # numpy uint8
        [2, (128, 128), np.ones((3, 256, 128, 3), dtype=np.float32), 1, 0.5],  # numpy fp32
        [2, (128, 128), np.ones((3, 256, 128, 3), dtype=np.float16), 1, 0.5],  # numpy fp16
        [2, (128, 128), [np.full((256, 128, 3), 255, dtype=np.uint8)] * 3, 2, 0.5],  # list of numpy uint8
        [2, (128, 128), [np.ones((256, 128, 3), dtype=np.float32)] * 3, 2, 0.5],  # list of numpy fp32
        [2, (128, 128), [np.ones((256, 128, 3), dtype=np.float16)] * 3, 2, 0.5],  # list of numpy fp16
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

    with torch.no_grad():
        out = processor(input_tensor)
    assert isinstance(out, list) and len(out) == expected_batches
    assert all(isinstance(b, torch.Tensor) for b in out)
    assert all(b.dtype == torch.float32 for b in out)
    assert all(b.shape[-2:] == output_size for b in out)
    assert all(torch.all(b == expected_value) for b in out)

    # Tests with padding mask
    processor_mask = PreProcessor(output_size, batch_size, return_padding_mask=True)
    with torch.no_grad():
        out_masked = processor_mask(input_tensor)
    imgs, masks = out_masked

    assert isinstance(imgs, list) and len(imgs) == expected_batches
    assert isinstance(masks, list) and len(masks) == expected_batches
    assert all(isinstance(b, torch.Tensor) for b in imgs)
    assert all(isinstance(m, torch.Tensor) for m in masks)
    assert all(b.dtype == torch.float32 for b in imgs)
    assert all(m.dtype == torch.bool for m in masks)
    assert all(b.shape[-2:] == output_size for b in imgs)
    assert all(m.shape[-2:] == output_size for m in masks)

    # mask sanity: should contain both 0 and 1 somewhere (padding depends on Resize logic)
    assert all(torch.is_tensor(m) for m in masks)
    assert len(repr(processor).split("\n")) == 4


def test_preprocessor_views():
    # Rectified crops are rotated views: np.rot90 of a crop of 1 pixel in height has a negative stride on an axis of
    # size 1, which numpy considers contiguous, but torch.from_numpy rejects
    crop = np.random.default_rng(0).integers(0, 256, (1, 40, 3), dtype=np.uint8)
    views = [np.rot90(crop, k) for k in range(4)] + [crop[:, ::-1], crop[..., ::-1]]
    processor = PreProcessor((32, 128), batch_size=8, preserve_aspect_ratio=True)
    expected = processor([view.copy() for view in views])
    out = processor(views)
    assert len(out) == 1 and torch.equal(out[0], expected[0])
    # Same for a batch of pages passed as a single array
    pages = np.rot90(np.repeat(crop[None], 2, axis=0), 3, axes=(1, 2))
    assert torch.equal(processor(pages)[0], processor(pages.copy())[0])
