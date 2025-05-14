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
        ([("hellowo", 0.5), ("loworld", 0.6)], [(0, 2, 0.5)], 0.5, [("helloworld", 0.5)]),
    ],
)
def test_remap_preds(preds, crop_map, split_overlap_ratio, pred):
    preds = remap_preds(preds, crop_map, split_overlap_ratio)
    assert len(preds) == len(pred)
    assert preds == pred
    assert all(isinstance(pred, tuple) for pred in preds)
    assert all(isinstance(pred[0], str) and isinstance(pred[1], float) for pred in preds)


def test_dont_split_if_not_necessary():
    max_ratio = 4
    inputs = [np.zeros((32, 32 * max_ratio, 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio, target_ratio=4, target_overlap_ratio=0.5)

    assert not _remap_required
    assert len(inputs) == len(new_crops)
    assert len(new_crops) == 1
    assert np.array_equal(inputs[0], new_crops[0])
    assert crop_map[0] == 0


def test_split_if_necessary():
    max_ratio = 4
    inputs = [np.zeros((32, 32 * max_ratio + 1, 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio, target_ratio=4, target_overlap_ratio=0.5)

    assert _remap_required
    assert len(inputs) < len(new_crops)
    assert len(new_crops) == 2
    assert not np.array_equal(inputs[0], new_crops[0])
    assert crop_map[0] == (0, 2, 0.9921875)


def test_split_only_if_necessary_with_larger_max_ratio():
    max_ratio = 8
    inputs = [np.zeros((32, 32 * max_ratio, 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio, target_ratio=4, target_overlap_ratio=0.5)

    assert not _remap_required
    assert len(inputs) == len(new_crops)
    assert len(new_crops) == 1
    assert np.array_equal(inputs[0], new_crops[0])
    assert crop_map[0] == 0


def test_split_in_two_equally_shaped_crops_with_half_overlap():
    inputs = [np.zeros((32, 128 + int(128 / 2), 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio=4, target_ratio=4, target_overlap_ratio=0.5)

    assert _remap_required
    assert len(new_crops) == 2
    for crop in new_crops:
        assert crop.shape == (32, 128, 3)
    assert crop_map[0] == (0, 2, 0.5)


def test_split_in_two_equally_shaped_crops_with_half_overlap_channels_first():
    inputs = [np.zeros((3, 32, 128 + int(128 / 2)))]

    new_crops, crop_map, _remap_required = split_crops(
        inputs, max_ratio=4, target_ratio=4, target_overlap_ratio=0.5, channels_last=False
    )

    assert _remap_required
    assert len(new_crops) == 2
    for crop in new_crops:
        assert crop.shape == (3, 32, 128)
    assert crop_map[0] == (0, 2, 0.5)


def test_split_in_two_equally_shaped_crops_with_half_overlap_with_max_ratio_below_target_split_ratio():
    inputs = [np.zeros((32, 128 + int(128 / 2), 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio=2, target_ratio=4, target_overlap_ratio=0.5)

    assert _remap_required
    assert len(new_crops) == 2
    for crop in new_crops:
        assert crop.shape == (32, 128, 3)
    assert crop_map[0] == (0, 2, 0.5)


def test_split_in_two_equally_shaped_crops_with_greater_than_half_last_overlap_ratio():
    inputs = [np.zeros((32, 128 + int(128 / 4), 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio=4, target_ratio=4, target_overlap_ratio=0.5)

    assert _remap_required
    assert len(new_crops) == 2
    for crop in new_crops:
        assert crop.shape == (32, 128, 3)
    assert crop_map[0] == (0, 2, 0.75)


def test_split_in_three_equally_shaped_crops_with_half_last_overlap_ratio():
    inputs = [np.zeros((32, 128 + 128, 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio=4, target_ratio=4, target_overlap_ratio=0.5)

    assert _remap_required
    assert len(new_crops) == 3
    for crop in new_crops:
        assert crop.shape == (32, 128, 3)
    assert crop_map[0] == (0, 3, 0.5)


def test_split_in_three_equally_shaped_crops_with_greater_than_half_last_overlap_ratio():
    inputs = [np.zeros((32, 128 + int(128 / 2) + int(128 / 4), 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio=4, target_ratio=4, target_overlap_ratio=0.5)

    assert _remap_required
    assert len(new_crops) == 3
    for crop in new_crops:
        assert crop.shape == (32, 128, 3)
    assert crop_map[0] == (0, 3, 0.75)


def test_split_into_larger_crops():
    inputs = [np.zeros((32, 192 * 2, 3))]

    new_crops, crop_map, _remap_required = split_crops(inputs, max_ratio=4, target_ratio=6, target_overlap_ratio=0.5)

    assert _remap_required
    assert len(new_crops) == 3
    for crop in new_crops:
        assert crop.shape == (32, 192, 3)
    assert crop_map[0] == (0, 3, 0.5)
