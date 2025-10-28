# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from rapidfuzz.distance import Hamming

__all__ = ["merge_strings", "merge_multi_strings"]


def merge_strings(a: str, b: str, overlap_ratio: float) -> str:
    """Merges 2 character sequences in the best way to maximize the alignment of their overlapping characters.

    Args:
        a: first char seq, suffix should be similar to b's prefix.
        b: second char seq, prefix should be similar to a's suffix.
        overlap_ratio: estimated ratio of overlapping characters.

    Returns:
        A merged character sequence.

    Example::
        >>> from doctr.models.recognition.utils import merge_strings
        >>> merge_strings('abcd', 'cdefgh', 0.5)
        'abcdefgh'
        >>> merge_strings('abcdi', 'cdefgh', 0.5)
        'abcdefgh'
    """
    seq_len = min(len(a), len(b))
    if seq_len <= 1:  # One sequence is empty or will be after cropping in next step, return both to keep data
        return a + b

    a_crop, b_crop = a[:-1], b[1:]  # Remove last letter of "a" and first of "b", because they might be cut off
    max_overlap = min(len(a_crop), len(b_crop))

    # Compute Hamming distances for all possible overlaps
    scores = [Hamming.distance(a_crop[-i:], b_crop[:i], processor=None) for i in range(1, max_overlap + 1)]

    # Find zero-score matches
    zero_matches = [i for i, score in enumerate(scores) if score == 0]

    expected_overlap = round(len(b) * overlap_ratio) - 3  # adjust for cropping and index

    # Case 1: One perfect match - exactly one zero score - just merge there
    if len(zero_matches) == 1:
        i = zero_matches[0]
        return a_crop + b_crop[i + 1 :]

    # Case 2: Multiple perfect matches - likely due to repeated characters.
    # Use the estimated overlap length to choose the match closest to the expected alignment.
    elif len(zero_matches) > 1:
        best_i = min(zero_matches, key=lambda x: abs(x - expected_overlap))
        return a_crop + b_crop[best_i + 1 :]

    # Case 3: Absence of zero scores indicates that the same character in the image was recognized differently OR that
    # the overlap was too small and we just need to merge the crops fully
    if expected_overlap < -1:
        return a + b
    elif expected_overlap < 0:
        return a_crop + b_crop

    # Find best overlap by minimizing Hamming distance + distance from expected overlap size
    combined_scores = [score + abs(i - expected_overlap) for i, score in enumerate(scores)]
    best_i = combined_scores.index(min(combined_scores))
    return a_crop + b_crop[best_i + 1 :]


def merge_multi_strings(seq_list: list[str], overlap_ratio: float, last_overlap_ratio: float) -> str:
    """
    Merges consecutive string sequences with overlapping characters.

    Args:
        seq_list: list of sequences to merge. Sequences need to be ordered from left to right.
        overlap_ratio: Estimated ratio of overlapping letters between neighboring strings.
        last_overlap_ratio: Estimated ratio of overlapping letters for the last element in seq_list.

    Returns:
        A merged character sequence

    Example::
        >>> from doctr.models.recognition.utils import merge_multi_strings
        >>> merge_multi_strings(['abc', 'bcdef', 'difghi', 'aijkl'], 0.5, 0.1)
        'abcdefghijkl'
    """
    if not seq_list:
        return ""
    result = seq_list[0]
    for i in range(1, len(seq_list)):
        text_b = seq_list[i]
        ratio = last_overlap_ratio if i == len(seq_list) - 1 else overlap_ratio
        result = merge_strings(result, text_b, ratio)
    return result
