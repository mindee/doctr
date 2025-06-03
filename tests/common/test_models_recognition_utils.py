import pytest

from doctr.models.recognition.utils import merge_multi_strings, merge_strings


@pytest.mark.parametrize(
    "a, b, overlap_ratio, merged",
    [
        # Last character of first string and first of last string will be cropped when merging - indicated by X
        ("abcX", "Xdef", 0.5, "abcdef"),
        ("abcdX", "Xdef", 0.75, "abcdef"),
        ("abcdeX", "Xdef", 0.9, "abcdef"),
        ("abcdefX", "Xdef", 0.9, "abcdef"),
        # Long repetition - four of seven characters in the second string are in the estimated overlap
        # X-chars will be cropped during merge, because they might be cut off during splitting of corresponding image
        ("abccccX", "Xcccccc", 4 / 7, "abcccccccc"),
        ("abc", "", 0.5, "abc"),
        ("", "abc", 0.5, "abc"),
        ("a", "b", 0.5, "ab"),
        # No overlap of input strings after crop
        ("abcdX", "Xefghi", 0.33, "abcdefghi"),
        # No overlap of input strings after crop with shorter inputs
        ("bcdX", "Xefgh", 0.4, "bcdefgh"),
        # No overlap of input strings after crop with even shorter inputs
        ("cdX", "Xefg", 0.5, "cdefg"),
        # Full overlap of input strings
        ("abcdX", "Xbcde", 1.0, "abcde"),
        # One repetition within inputs
        ("ababX", "Xabde", 0.8, "ababde"),
        # Multiple repetitions within inputs
        ("ababX", "Xabab", 0.8, "ababab"),
        # Multiple repetitions within inputs with shorter input strings
        ("abaX", "Xbab", 1.0, "abab"),
        # Longer multiple repetitions within inputs with half overlap
        ("cabababX", "Xabababc", 0.5, "cabababababc"),
        # Longer multiple repetitions within inputs with full overlap
        ("ababaX", "Xbabab", 1.0, "ababab"),
        # One different letter in overlap
        ("one_differon", "ferent_letter", 0.5, "one_differont_letter"),
        # First string empty after crop
        ("-", "test", 0.9, "-test"),
        # Second string empty after crop
        ("test", "-", 0.9, "test-"),
    ],
)
def test_merge_strings(a, b, overlap_ratio, merged):
    assert merged == merge_strings(a, b, overlap_ratio)


@pytest.mark.parametrize(
    "seq_list, overlap_ratio, last_overlap_ratio, merged",
    [
        # One character at each conjunction point will be cropped when merging - indicated by X
        (["abcX", "Xdef"], 0.5, 0.5, "abcdef"),
        (["abcdX", "XdefX", "XefghX", "Xijk"], 0.5, 0.5, "abcdefghijk"),
        (["abcdX", "XdefX", "XefghiX", "Xaijk"], 0.5, 0.8, "abcdefghijk"),
        (["aaaa", "aaab", "aabc"], 0.8, 0.3, "aaaabc"),
        # Handle empty input
        ([], 0.5, 0.4, ""),
    ],
)
def test_merge_multi_strings(seq_list, overlap_ratio, last_overlap_ratio, merged):
    assert merged == merge_multi_strings(seq_list, overlap_ratio, last_overlap_ratio)
