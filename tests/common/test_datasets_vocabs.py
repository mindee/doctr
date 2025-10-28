from collections import Counter

from doctr.datasets import VOCABS


def test_vocabs_duplicates():
    for key, vocab in VOCABS.items():
        assert isinstance(vocab, str)

        duplicates = [char for char, count in Counter(vocab).items() if count > 1]
        assert not duplicates, f"Duplicate characters in {key} vocab: {duplicates}"
