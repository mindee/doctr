import pytest

from doctr.models.recognition.utils import merge_multi_strings, merge_strings


@pytest.mark.parametrize(
    "a, b, merged",
    [
        ["abc", "def", "abcdef"],
        ["abcd", "def", "abcdef"],
        ["abcde", "def", "abcdef"],
        ["abcdef", "def", "abcdef"],
        ["abcccc", "cccccc", "abcccccccc"],
        ["abc", "", "abc"],
        ["", "abc", "abc"],
    ],
)
def test_merge_strings(a, b, merged):
    assert merged == merge_strings(a, b, 1.4)


@pytest.mark.parametrize(
    "seq_list, merged",
    [
        [["abc", "def"], "abcdef"],
        [["abcd", "def", "efgh", "ijk"], "abcdefghijk"],
        [["abcdi", "defk", "efghi", "aijk"], "abcdefghijk"],
    ],
)
def test_merge_multi_strings(seq_list, merged):
    assert merged == merge_multi_strings(seq_list, 1.4)
