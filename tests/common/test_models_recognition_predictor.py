import numpy as np
import pytest

from doctr.models.recognition.predictor._utils import remap_preds, split_crops


@pytest.mark.parametrize(
    "crops, max_ratio, target_ratio, target_overlap_ratio, channels_last, num_crops",
    [
        # No split required
        [[np.zeros((32, 128, 3), dtype=np.uint8)], 8, 4, 0.5, True, 1],
        [[np.zeros((3, 32, 128), dtype=np.uint8)], 8, 4, 0.5, False, 1],
        # Split required
        [[np.zeros((32, 1024, 3), dtype=np.uint8)], 8, 6, 0.5, True, 10],
        [[np.zeros((3, 32, 1024), dtype=np.uint8)], 8, 6, 0.5, False, 10],
    ],
)
def test_split_crops(crops, max_ratio, target_ratio, target_overlap_ratio, channels_last, num_crops):
    new_crops, crop_map, should_remap = split_crops(crops, max_ratio, target_ratio, target_overlap_ratio, channels_last)
    assert len(new_crops) == num_crops
    assert len(crop_map) == len(crops)
    assert should_remap == (len(crops) != len(new_crops))


@pytest.mark.parametrize(
    "preds, crop_map, split_overlap_ratio, pred",
    [
        # Nothing to remap
        ([("hello", 0.5)], [0], 0.5, [("hello", 0.5)]),
        # Merge
        ([("hellowo", 0.5), ("loworld", 0.6)], [(0, 2, 0.5)], 0.5, [("helloworld", 0.55)]),
    ],
)
def test_remap_preds(preds, crop_map, split_overlap_ratio, pred):
    preds = remap_preds(preds, crop_map, split_overlap_ratio)
    assert len(preds) == len(pred)
    assert preds == pred
    assert all(isinstance(pred, tuple) for pred in preds)
    assert all(isinstance(pred[0], str) and isinstance(pred[1], float) for pred in preds)


@pytest.mark.parametrize(
    "inputs, max_ratio, target_ratio, target_overlap_ratio, expected_remap_required, expected_len, expected_shape, "
    "expected_crop_map, channels_last",
    [
        # Don't split
        ([np.zeros((32, 32 * 4, 3))], 4, 4, 0.5, False, 1, (32, 128, 3), 0, True),
        # Split needed
        ([np.zeros((32, 32 * 4 + 1, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.9921875), True),
        # Larger max ratio prevents split
        ([np.zeros((32, 32 * 8, 3))], 8, 4, 0.5, False, 1, (32, 256, 3), 0, True),
        # Half-overlap, two crops
        ([np.zeros((32, 128 + 64, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.5), True),
        # Half-overlap, two crops, channels first
        ([np.zeros((3, 32, 128 + 64))], 4, 4, 0.5, True, 2, (3, 32, 128), (0, 2, 0.5), False),
        # Half-overlap with small max_ratio forces split
        ([np.zeros((32, 128 + 64, 3))], 2, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.5), True),
        # > half last overlap ratio
        ([np.zeros((32, 128 + 32, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.75), True),
        # 3 crops, half last overlap
        ([np.zeros((32, 128 + 128, 3))], 4, 4, 0.5, True, 3, (32, 128, 3), (0, 3, 0.5), True),
        # 3 crops, > half last overlap
        ([np.zeros((32, 128 + 64 + 32, 3))], 4, 4, 0.5, True, 3, (32, 128, 3), (0, 3, 0.75), True),
        # Split into larger crops
        ([np.zeros((32, 192 * 2, 3))], 4, 6, 0.5, True, 3, (32, 192, 3), (0, 3, 0.5), True),
        # Test fallback for empty splits
        ([np.empty((1, 0, 3))], -1, 4, 0.5, False, 1, (1, 0, 3), (0), True),
    ],
)
def test_split_crops_cases(
    inputs,
    max_ratio,
    target_ratio,
    target_overlap_ratio,
    expected_remap_required,
    expected_len,
    expected_shape,
    expected_crop_map,
    channels_last,
):
    new_crops, crop_map, _remap_required = split_crops(
        inputs,
        max_ratio=max_ratio,
        target_ratio=target_ratio,
        split_overlap_ratio=target_overlap_ratio,
        channels_last=channels_last,
    )

    assert _remap_required == expected_remap_required
    assert len(new_crops) == expected_len
    assert len(crop_map) == 1

    if expected_remap_required:
        assert isinstance(crop_map[0], tuple)

    assert crop_map[0] == expected_crop_map

    for crop in new_crops:
        assert crop.shape == expected_shape


@pytest.mark.parametrize(
    "split_overlap_ratio",
    [
        # lower bound
        0.0,
        # upper bound
        1.0,
    ],
)
def test_invalid_split_overlap_ratio(split_overlap_ratio):
    with pytest.raises(ValueError):
        split_crops(
            [np.zeros((32, 32 * 4, 3))],
            max_ratio=4,
            target_ratio=4,
            split_overlap_ratio=split_overlap_ratio,
        )
