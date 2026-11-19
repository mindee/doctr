import numpy as np
import pytest

from doctr.models.recognition.predictor._utils import ambiguous_overlaps, remap_preds, resolve_overlap, split_crops


@pytest.mark.parametrize(
    "crops, max_ratio, target_ratio, target_overlap_ratio, num_crops",
    [
        # No split required
        [[np.zeros((32, 128, 3), dtype=np.uint8)], 8, 4, 0.5, 1],
        # Split required
        [[np.zeros((32, 1024, 3), dtype=np.uint8)], 8, 6, 0.5, 10],
    ],
)
def test_split_crops(crops, max_ratio, target_ratio, target_overlap_ratio, num_crops):
    new_crops, crop_map, should_remap = split_crops(crops, max_ratio, target_ratio, target_overlap_ratio)
    assert len(new_crops) == num_crops
    assert len(crop_map) == len(crops)
    assert should_remap == (len(crops) != len(new_crops))


@pytest.mark.parametrize(
    "preds, crop_map, split_overlap_ratio, pred",
    [
        # Nothing to remap
        ([("hello", 0.5)], [0], 0.5, [("hello", 0.5)]),
        # Merge: the confidence of the merged word is the one of its weakest part
        ([("hellowo", 0.5), ("loworld", 0.6)], [(0, 2, 0.5)], 0.5, [("helloworld", 0.5)]),
        # A weak part must not be averaged away
        ([("hello wor", 0.99), ("orld", 0.1)], [(0, 2, 0.5)], 0.5, [("hello world", 0.1)]),
        # Parts without text (e.g. blank padding) are ignored for the confidence
        ([("hello wor", 0.95), ("", 0.0)], [(0, 2, 0.99)], 0.5, [("hello wor", 0.95)]),
        # Only empty parts
        ([("", 0.2), ("", 0.4)], [(0, 2, 0.5)], 0.5, [("", 0.2)]),
        # Mixed: unsplit and split crops
        (
            [("single", 0.9), ("hellowo", 0.8), ("loworld", 0.7)],
            [0, (1, 3, 0.5)],
            0.5,
            [("single", 0.9), ("helloworld", 0.7)],
        ),
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
    "expected_crop_map",
    [
        # Don't split
        ([np.zeros((32, 32 * 4, 3))], 4, 4, 0.5, False, 1, (32, 128, 3), 0),
        # Split needed
        ([np.zeros((32, 32 * 4 + 1, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.9921875)),
        # Larger max ratio prevents split
        ([np.zeros((32, 32 * 8, 3))], 8, 4, 0.5, False, 1, (32, 256, 3), 0),
        # Half-overlap, two crops
        ([np.zeros((32, 128 + 64, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.5)),
        # Half-overlap with small max_ratio forces split
        ([np.zeros((32, 128 + 64, 3))], 2, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.5)),
        # > half last overlap ratio
        ([np.zeros((32, 128 + 32, 3))], 4, 4, 0.5, True, 2, (32, 128, 3), (0, 2, 0.75)),
        # 3 crops, half last overlap
        ([np.zeros((32, 128 + 128, 3))], 4, 4, 0.5, True, 3, (32, 128, 3), (0, 3, 0.5)),
        # 3 crops, > half last overlap
        ([np.zeros((32, 128 + 64 + 32, 3))], 4, 4, 0.5, True, 3, (32, 128, 3), (0, 3, 0.75)),
        # Split into larger crops
        ([np.zeros((32, 192 * 2, 3))], 4, 6, 0.5, True, 3, (32, 192, 3), (0, 3, 0.5)),
        # Test fallback for empty splits
        ([np.empty((1, 0, 3))], -1, 4, 0.5, False, 1, (1, 0, 3), (0)),
        # Zero-height crops are passed through instead of raising ZeroDivisionError
        ([np.empty((0, 64, 3))], 4, 4, 0.5, False, 1, (0, 64, 3), (0)),
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
):
    new_crops, crop_map, _remap_required = split_crops(
        inputs,
        max_ratio=max_ratio,
        target_ratio=target_ratio,
        split_overlap_ratio=target_overlap_ratio,
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


@pytest.mark.parametrize(
    "confidence_aggregation, conf",
    [("min", 0.1), ("mean", 0.5), ("geometric_mean", 0.3), (lambda confs: 0.42, 0.42)],
)
def test_remap_preds_confidence_aggregation(confidence_aggregation, conf):
    # The empty part is ignored for the confidence
    preds = [("hellowo", 0.9), ("", 0.0), ("loworld", 0.1)]
    assert remap_preds(preds, [(0, 3, 0.5)], 0.5, confidence_aggregation) == [("helloworld", pytest.approx(conf))]


def test_remap_preds_overlaps():
    preds = [("WARMUP", 0.9), ("MUP-0000", 0.8), ("-0000001", 0.7), ("single", 0.6), ("WARMUP", 0.9), ("MUP-0000", 0.8)]
    preds += [("-0000001", 0.7)]
    crop_map = [(0, 3, 0.519), 3, (4, 7, 0.519)]
    unresolved = [("WARMUP-00000001", 0.7), ("single", 0.6), ("WARMUP-00000001", 0.7)]
    assert remap_preds(preds, crop_map, 0.5) == unresolved
    assert remap_preds(preds, crop_map, 0.5, overlaps={}) == unresolved
    # The overlap of parts 1 and 2 of the first crop only
    assert remap_preds(preds, crop_map, 0.5, overlaps={(0, 1): 3}) == [
        ("WARMUP-0000001", 0.7),
        ("single", 0.6),
        ("WARMUP-00000001", 0.7),
    ]


def test_ambiguous_overlaps():
    crops = [np.zeros((32, 128, 3), dtype=np.uint8), np.arange(10 * 200 * 3, dtype=np.int64).reshape(10, 200, 3)]
    parts, crop_map, _ = split_crops(crops, 8, 6, 0.5)
    # The second crop is cut into 6 parts of 60 columns starting at 0, 30, 60, 90, 120 and 140
    assert crop_map == [0, (1, 7, pytest.approx(40 / 60))]
    texts = ["single", "XY-0000", "-000000", "00001AB", "1ABCDEF", "GH00000", "0000000"]

    pairs = ambiguous_overlaps([(text, 0.9) for text in texts], crop_map, crops, 6, 0.5)

    # Runs span the overlaps of parts 0 and 1, 1 and 2, and 4 and 5
    assert [key for key, *_ in pairs] == [(1, 0), (1, 1), (1, 4)]
    for (_, region, a, b, ratio, offset), start, end in zip(pairs, [0, 30, 120], [90, 120, 200]):
        np.testing.assert_array_equal(region, crops[1][:, start:end])
    assert [(a, b) for _, _, a, b, _, _ in pairs] == [
        ("XY-0000", "-000000"),
        ("-000000", "00001AB"),
        ("GH00000", "0000000"),
    ]
    assert [ratio for *_, ratio, _ in pairs] == [0.5, 0.5, pytest.approx(40 / 60)]
    assert [offset for *_, offset in pairs] == [0.5, 0.5, pytest.approx(20 / 60)]

    # Nothing to resolve where merge_strings picks no exact overlap, nor for a crop whose parts do not match its split
    assert ambiguous_overlaps([("aaab", 0.9), ("baabbbbb", 0.9)], [(0, 2, 0.2)], [crops[1][:, :90]], 6, 0.5) == []
    assert ambiguous_overlaps([(text, 0.9) for text in texts[:3]], [0, (1, 3, 0.5)], crops, 6, 0.5) == []


@pytest.mark.parametrize(
    "a, b, overlap_ratio, offset, region_text, length",
    [
        # A 154 x 14 crop of "WARMUP-0000001" read in 3 parts: merge_strings overlaps the last two by 3 chars, and adds
        # a zero; the reading of the region they cover and the positions of their chars both point at 4
        ("MUP-00000", "-0000001", 2 / 3, 1 / 3, "MUP-0000001", 4),
        # A reading of the region that matches merge_strings' overlap, or that is more than 2 edits from any merge
        ("MUP-00000", "-0000001", 2 / 3, 1 / 3, "MUP-00000001", None),
        ("MUP-00000", "-0000001", 2 / 3, 1 / 3, "MUP-0O0O0O1", None),
        ("MUP-00000", "-0000001", 2 / 3, 1 / 3, "XYZ", None),
        # A 210 x 24 crop of "U-0000000008911" read in 2 parts, each cutting a zero in two: the exact reading of the
        # region points at 5 chars, the positions of the chars spread evenly over the parts at 4, and the two tie
        ("U-000000000", "00000008911", 78 / 144, 66 / 144, "U-0000000008911", None),
        # Only one exact overlap length, or none that merge_strings picks
        ("hellowo", "loworld", 0.5, 0.5, "helloworld", None),
        ("aaab", "baabbbbb", 0.2, 0.5, "aaabbbbb", None),
    ],
)
def test_resolve_overlap(a, b, overlap_ratio, offset, region_text, length):
    assert resolve_overlap(a, b, overlap_ratio, offset, region_text) == length


def test_resolve_overlap_margin():
    # The region reading and the char positions score an overlap of 4 chars 1.28 better than merge_strings' 3
    assert resolve_overlap("MUP-00000", "-0000001", 2 / 3, 1 / 3, "MUP-0000001", margin=1.2) == 4
    assert resolve_overlap("MUP-00000", "-0000001", 2 / 3, 1 / 3, "MUP-0000001", margin=1.3) is None
    # Without a margin, a tie still keeps merge_strings' overlap
    assert resolve_overlap("U-000000000", "00000008911", 78 / 144, 66 / 144, "U-0000000008911", margin=0) is None
