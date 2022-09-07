import numpy as np
import pytest

from doctr.models.recognition.predictor._utils import remap_preds, split_crops


@pytest.mark.parametrize(
    "crops, max_ratio, target_ratio, dilation, channels_last, num_crops",
    [
        # No split required
        [[np.zeros((32, 128, 3), dtype=np.uint8)], 8, 4, 1.4, True, 1],
        [[np.zeros((3, 32, 128), dtype=np.uint8)], 8, 4, 1.4, False, 1],
        # Split required
        [[np.zeros((32, 1024, 3), dtype=np.uint8)], 8, 6, 1.4, True, 5],
        [[np.zeros((3, 32, 1024), dtype=np.uint8)], 8, 6, 1.4, False, 5],
    ],
)
def test_split_crops(crops, max_ratio, target_ratio, dilation, channels_last, num_crops):
    new_crops, crop_map, should_remap = split_crops(crops, max_ratio, target_ratio, dilation, channels_last)
    assert len(new_crops) == num_crops
    assert len(crop_map) == len(crops)
    assert should_remap == (len(crops) != len(new_crops))


@pytest.mark.parametrize(
    "preds, crop_map, dilation, pred",
    [
        # Nothing to remap
        [[("hello", 0.5)], [0], 1.4, [("hello", 0.5)]],
        # Merge
        [[("hellowo", 0.5), ("loworld", 0.6)], [(0, 2)], 1.4, [("helloworld", 0.5)]],
    ],
)
def test_remap_preds(preds, crop_map, dilation, pred):
    preds = remap_preds(preds, crop_map, dilation)
    assert len(preds) == len(pred)
    assert preds == pred
    assert all(isinstance(pred, tuple) for pred in preds)
    assert all(isinstance(pred[0], str) and isinstance(pred[1], float) for pred in preds)
