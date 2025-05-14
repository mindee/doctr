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
        >>> merge_strings('abcd', 'cdefgh', 0.33)
        'abcdefgh'
        >>> merge_strings('abcdi', 'cdefgh', 0.5)
        'abcdefgh'
    """
    lower_inputs_len = min(len(a), len(b))
    if lower_inputs_len <= 1:  # One sequence is empty or will be after cropping in next step, return the other
        return b if len(a) == lower_inputs_len else a

    # Remove last letter of "a" and first of "b", because they might be cut off. The removed characters are still
    # present in the other string due to overlap
    a_crop = a[:-1]
    b_crop = b[1:]

    # Increase overlap step by step and compute number of different characters per overlap
    max_overlap = min(len(a_crop), len(b_crop))
    scores = [Hamming.distance(a_crop[-i:], b_crop[:i], processor=None) for i in range(1, max_overlap + 1)]

    indexes_of_zero_scores = [i for i, score in enumerate(scores) if score == 0]

    # Case 1: One perfect match - exactly one zero score - just merge there
    has_exactly_one_perfect_match = len(indexes_of_zero_scores) == 1
    if has_exactly_one_perfect_match:
        index_of_zero_score = indexes_of_zero_scores[0]
        return a_crop + b_crop[index_of_zero_score + 1 :]

    # Case 2: Multiple perfect matches - Indicates repetitions of characters in overlap: Use estimated number of
    # characters in overlap to estimate which zero score fits bests

    # Estimate the number of characters in the overlap - use "b" because "a" might be a merged string already
    number_of_chars_in_overlap = round(len(b) * overlap_ratio)
    # Account for the cropped letters in a and b
    number_of_chars_in_cropped_overlap = number_of_chars_in_overlap - 2

    has_multiple_perfect_matches = len(indexes_of_zero_scores) > 1
    if has_multiple_perfect_matches:
        closest_index_of_zero_score_to_expected_overlap = _get_index_closest_to_target_index(
            indexes_of_zero_scores, number_of_chars_in_cropped_overlap
        )
        return a_crop + b_crop[closest_index_of_zero_score_to_expected_overlap + 1 :]

    # Absence of zero scores indicates that the same character in the image was recognized differently OR that the
    # overlap was too small and we just need to merge the crops fully

    # Case 3: Merge crops fully
    if number_of_chars_in_cropped_overlap < 1:
        return a_crop + b_crop

    # Case 4: Merge where low score fits best to estimated number of characters in overlap
    combined_scores = []
    for i, score in enumerate(scores):
        distance_to_estimated_overlap = abs(number_of_chars_in_cropped_overlap - i)
        combined_scores.append(score + distance_to_estimated_overlap)
    index_of_min_combined_score = combined_scores.index(min(combined_scores))
    return a_crop + b_crop[index_of_min_combined_score + 1 :]


def _get_index_closest_to_target_index(indexes: list[int], target_index: int) -> int:
    closest_index_to_target = -1
    min_distance = float("inf")
    for i in indexes:
        distance_to_target_index = abs(target_index - i)
        if distance_to_target_index < min_distance:
            min_distance = distance_to_target_index
            closest_index_to_target = i
    return closest_index_to_target


def merge_multi_strings(seq_list: list[str], overlap_ratio: float, last_overlap_ratio: float) -> str:
    """Merges consecutive string sequences with overlapping characters.

    Args:
    ----
        seq_list: list of sequences to merge. Sequences need to be ordered from left to right.
        overlap_ratio: Estimated ratio of overlapping letters between neighboring strings.
        last_overlap_ratio: Estimated ratio of overlapping letters for the last element in seq_list.

    Returns:
    -------
        A merged character sequence

    Example::
        >>> from doctr.models.recognition.utils import merge_multi_strings
        >>> merge_multi_strings(['abc', 'bcdef', 'difghi', 'aijkl'], 0.5, 0.1)
        'abcdefghijkl'
    """
    result = seq_list[0]
    for i in range(1, len(seq_list)):
        text_a = result
        text_b = seq_list[i]
        if len(seq_list) == i + 1:
            overlap_ratio = last_overlap_ratio

        result = merge_strings(text_a, text_b, overlap_ratio)

    return result
