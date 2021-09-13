import pytest
from doctr.model.recognition.utils import merge_sequences, merge_multi_sequences


@pytest.mark.parametrize(
    "a, b, merged",
    [
        ['abc', 'def', 'abcdef'],
        ['abcd', 'def', 'abcdef'],
        ['abcde', 'def', 'abcdef'],
        ['abcee', 'def', 'abcdef'],
        ['abcdef', 'def', 'abcdef'],
        ['abccc', 'ccccc', 'abcccc'],
    ],
)
def test_merge_sequences(a, b, merged):
    assert merged == merge_sequences(a, b, 1.4)


@pytest.mark.parametrize(
    "seq_list, merged",
    [
        [['abc', 'def'], 'abcdef'],
        [['abcd', 'def', 'efgh', 'ijk'], 'abcdefghijk'],
        [['abcdi', 'defk', 'efghi', 'aijk'], 'abcdefghijk'],
    ],
)
def test_merge_multi_sequences(seq_list, merged):
    assert merged == merge_multi_sequences(seq_list, 1.4)
