# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

from rapidfuzz.string_metric import levenshtein

__all__ = ["merge_strings", "merge_multi_strings"]


def merge_strings(a: str, b: str, dil_factor: float) -> str:
    """Merges 2 character sequences in the best way to maximize the alignment of their overlapping characters.

    Args:
        a: first char seq, suffix should be similar to b's prefix.
        b: second char seq, prefix should be similar to a's suffix.
        dil_factor: dilation factor of the boxes to overlap, should be > 1. This parameter is
            only used when the mother sequence is splitted on a character repetition

    Returns:
        A merged character sequence.

    Example::
        >>> from doctr.model.recognition.utils import merge_sequences
        >>> merge_sequences('abcd', 'cdefgh', 1.4)
        'abcdefgh'
        >>> merge_sequences('abcdi', 'cdefgh', 1.4)
        'abcdefgh'
    """
    seq_len = min(len(a), len(b))
    if seq_len == 0:  # One sequence is empty, return the other
        return b if len(a) == 0 else b

    # Initialize merging index and corresponding score (mean Levenstein)
    min_score, index = 1.0, 0  # No overlap, just concatenate

    scores = [levenshtein(a[-i:], b[:i], processor=None) / i for i in range(1, seq_len + 1)]

    # Edge case (split in the middle of char repetitions): if it starts with 2 or more 0
    if len(scores) > 1 and (scores[0], scores[1]) == (0, 0):
        # Compute n_overlap (number of overlapping chars, geometrically determined)
        n_overlap = round(len(b) * (dil_factor - 1) / dil_factor)
        # Find the number of consecutive zeros in the scores list
        # Impossible to have a zero after a non-zero score in that case
        n_zeros = sum(val == 0 for val in scores)
        # Index is bounded by the geometrical overlap to avoid collapsing repetitions
        min_score, index = 0, min(n_zeros, n_overlap)

    else:  # Common case: choose the min score index
        for i, score in enumerate(scores):
            if score < min_score:
                min_score, index = score, i + 1  # Add one because first index is an overlap of 1 char

    # Merge with correct overlap
    if index == 0:
        return a + b
    return a[:-1] + b[index - 1 :]


def merge_multi_strings(seq_list: List[str], dil_factor: float) -> str:
    """Recursively merges consecutive string sequences with overlapping characters.

    Args:
        seq_list: list of sequences to merge. Sequences need to be ordered from left to right.
        dil_factor: dilation factor of the boxes to overlap, should be > 1. This parameter is
            only used when the mother sequence is splitted on a character repetition

    Returns:
        A merged character sequence

    Example::
        >>> from doctr.model.recognition.utils import merge_multi_sequences
        >>> merge_multi_sequences(['abc', 'bcdef', 'difghi', 'aijkl'], 1.4)
        'abcdefghijkl'
    """

    def _recursive_merge(a: str, seq_list: List[str], dil_factor: float) -> str:
        # Recursive version of compute_overlap
        if len(seq_list) == 1:
            return merge_strings(a, seq_list[0], dil_factor)
        return _recursive_merge(merge_strings(a, seq_list[0], dil_factor), seq_list[1:], dil_factor)

    return _recursive_merge("", seq_list, dil_factor)
