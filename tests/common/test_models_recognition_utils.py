import pytest

from doctr.models.recognition.utils import merge_multi_strings, merge_strings


# Last character of first string and first of last string will be cropped when merging - indicated by X
@pytest.mark.parametrize(
    "a, b, overlap_ratio, merged",
    [
        ("abcX", "Xdef", 0.25, "abcdef"),
        ("abcdX", "Xdef", 0.75, "abcdef"),
        ("abcdeX", "Xdef", 0.9, "abcdef"),
        ("abcdefX", "Xdef", 0.9, "abcdef"),
        ("abccccX", "Xcccccc", 0.4, "abcccccccc"),
        ("abc", "", 0.5, "abc"),
        ("", "abc", 0.5, "abc"),
    ],
)
def test_merge_strings(a, b, overlap_ratio, merged):
    assert merged == merge_strings(a, b, overlap_ratio)


@pytest.mark.parametrize(
    "seq_list, overlap_ratio, last_overlap_ratio, merged",
    [
        (["abcX", "Xdef"], 0.25, 0.25, "abcdef"),
        (["abcdX", "XdefX", "XefghX", "Xijk"], 0.5, 0.5, "abcdefghijk"),
        (["abcdX", "XdefX", "XefghiX", "Xaijk"], 0.5, 0.8, "abcdefghijk"),
    ],
)
def test_merge_multi_strings(seq_list, overlap_ratio, last_overlap_ratio, merged):
    assert merged == merge_multi_strings(seq_list, overlap_ratio, last_overlap_ratio)


def test_example_from_docs_1():
    a = "abcd"
    b__ = "cdefgh"
    # weirdly named b__ to easily see where a and b overlap
    expected_result = "abcdefgh"
    overlap_ratio = 0.33

    result = merge_strings(a, b__, overlap_ratio)

    assert result == expected_result


def test_example_from_docs_2():
    a = "abcdi"
    b__ = "cdefgh"
    # weirdly named b__ to easily see where a and b overlap
    expected_result = "abcdefgh"
    overlap_ratio = 0.5

    result = merge_strings(a, b__, overlap_ratio)

    assert result == expected_result


def test_no_overlap_after_crop():
    a = "abcdX"
    b___ = "Xefghi"
    expected_result = "abcdefghi"
    overlap_ratio = 0.33

    result = merge_strings(a, b___, overlap_ratio)

    assert result == expected_result


def test_no_overlap_after_crop_shorter():
    a = "bcdX"
    b__ = "Xefgh"
    expected_result = "bcdefgh"
    overlap_ratio = 0.4

    result = merge_strings(a, b__, overlap_ratio)

    assert result == expected_result


def test_no_overlap_after_crop_even_shorter():
    a = "cdX"
    b_ = "Xefg"
    expected_result = "cdefg"
    overlap_ratio = 0.5

    result = merge_strings(a, b_, overlap_ratio)

    assert result == expected_result


def test_full_overlap():
    a = "abcdX"
    b = "Xbcde"
    expected_result = "abcde"
    overlap_ratio = 1.0

    result = merge_strings(a, b, overlap_ratio)

    assert result == expected_result


def test_one_repetition():
    a = "ababX"
    b_ = "Xabde"
    expected_result = "ababde"
    overlap_ratio = 0.8

    result = merge_strings(a, b_, overlap_ratio)

    assert result == expected_result


def test_multiple_repetitions():
    a = "ababX"
    b_ = "Xabab"
    expected_result = "ababab"
    overlap_ratio = 0.8

    result = merge_strings(a, b_, overlap_ratio)

    assert result == expected_result


def test_multiple_repetitions_shorter():
    a = "abaX"
    b = "Xbab"
    expected_result = "abab"
    overlap_ratio = 1.0

    result = merge_strings(a, b, overlap_ratio)

    assert result == expected_result


def test_multiple_repetitions_full_overlap():
    a_ = "ababX"
    b = "Xabab"
    expected_result = "abab"
    overlap_ratio = 1.0

    result = merge_strings(a_, b, overlap_ratio)

    assert result == expected_result


def test_longer_multiple_repetitions_half_overlap():
    a = "cabababX"
    b____ = "Xabababc"
    expected_result = "cabababababc"
    overlap_ratio = 0.5

    result = merge_strings(a, b____, overlap_ratio)

    assert result == expected_result


def test_longer_multiple_repetitions_full_overlap():
    a_ = "abababX"
    b = "Xababab"
    expected_result = "ababab"
    overlap_ratio = 1.0

    result = merge_strings(a_, b, overlap_ratio)

    assert result == expected_result


def test_one_different_letter():
    a = "one_differon"
    b_______ = "ferent_letter"
    expected_result = "one_differont_letter"
    overlap_ratio = 0.5

    result = merge_strings(a, b_______, overlap_ratio)

    assert result == expected_result


def test_merge_multi_strings_example_from_docs():
    strings = ["abc", "bcdef", "difghi", "aijkl"]
    expected_result = "abcdefghijkl"
    overlap_ratio = 0.5
    last_overlap_ratio = 0.1

    result = merge_multi_strings(strings, overlap_ratio, last_overlap_ratio)

    assert result == expected_result
