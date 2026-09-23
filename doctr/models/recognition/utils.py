# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math

from rapidfuzz.distance import Levenshtein

__all__ = ["merge_strings", "merge_multi_strings"]


def merge_strings(a: str, b: str, overlap_ratio: float) -> str:
    """Merges 2 character sequences in the best way to maximize the alignment of their overlapping characters.

    Args:
        a: first char seq, suffix should be similar to b's prefix.
        b: second char seq, prefix should be similar to a's suffix.
        overlap_ratio: estimated ratio of overlapping characters, relative to ``len(b)``, in [0, 1].

    Returns:
        A merged character sequence.

    >>> from doctr.models.recognition.utils import merge_strings
    >>> merge_strings('abcdX', 'Xdefgh', 0.5)
    'abcdefgh'
    >>> merge_strings('abcdi', 'cdefgh', 0.5)
    'abcdefgh'
    """
    overlap_ratio = min(max(overlap_ratio, 0.0), 1.0)
    if min(len(a), len(b)) <= 1:  # One sequence is empty or will be after cropping in next step, return both
        return a + b

    a_crop, b_crop = a[:-1], b[1:]  # Remove last letter of "a" and first of "b", because they might be cut off
    max_overlap = min(len(a_crop), len(b_crop))
    # Expected overlap length between the crops (2 chars were removed by cropping)
    expected = len(b) * overlap_ratio - 2

    # Start with the "no overlap" hypothesis: cost grows with the expected overlap
    best_cost, best_i, best_j, best_dist = max(expected, 0.0) + 1.0, 0, 0, math.inf

    # Compare the last i chars of a_crop with the first j chars of b_crop (|i - j| <= 2 absorbs inserted or
    # dropped chars). Lengths closest to the expected one are visited first, so the search stops as soon as no
    # remaining candidate can beat the best one.
    for i in sorted(range(1, max_overlap + 1), key=lambda n: (abs(n - expected), -n)):
        length_cost = abs(i - expected)
        if length_cost > best_cost:
            break
        for j in range(max(1, i - 2), min(len(b_crop), i + 2) + 1):
            fixed_cost = length_cost + 0.25 * abs(i - j)
            # Largest acceptable edit distance: at most 0.8 edits per char, and still able to win (or tie)
            cutoff = min(math.floor(0.8 * max(i, j)), math.floor(best_cost - fixed_cost))
            if cutoff < abs(i - j):  # the edit distance is at least the length difference
                continue
            dist = Levenshtein.distance(a_crop[-i:], b_crop[:j], score_cutoff=cutoff)
            if dist > cutoff:
                continue
            cost = dist + fixed_cost
            # On a tie, prefer the closer match, then the longer overlap
            if cost < best_cost or (cost == best_cost and (dist, -i) < (best_dist, -best_i)):
                best_cost, best_i, best_j, best_dist = cost, i, j, dist

    if best_i == 0:
        # No overlap: with less than ~1 char of expected overlap, the border chars are most likely complete
        # characters (each read by one crop only), so they are kept
        return a + b if len(b) * overlap_ratio < 1 else a_crop + b_crop

    # Join both readings of the overlap in the middle of their alignment: chars close to a crop border are the
    # least reliable, so the left half is taken from a_crop and the right half from b_crop
    tail, head = a_crop[-best_i:], b_crop[:best_j]
    cuts = [(0, 0)]
    for op in Levenshtein.opcodes(tail, head):
        if op.tag == "equal":
            cuts.extend((op.src_start + t, op.dest_start + t) for t in range(1, op.src_end - op.src_start + 1))
        else:
            cuts.append((op.src_end, op.dest_end))
    cut_a, cut_b = min(cuts, key=lambda c: abs(c[0] + c[1] - (best_i + best_j) / 2))
    return a_crop[: len(a_crop) - best_i] + tail[:cut_a] + head[cut_b:] + b_crop[best_j:]


def merge_multi_strings(seq_list: list[str], overlap_ratio: float, last_overlap_ratio: float) -> str:
    """
    Merges consecutive string sequences with overlapping characters.

    Args:
        seq_list: list of sequences to merge. Sequences need to be ordered from left to right.
        overlap_ratio: Estimated ratio of overlapping letters between neighboring strings.
        last_overlap_ratio: Estimated ratio of overlapping letters for the last element in seq_list.

    Returns:
        A merged character sequence

    >>> from doctr.models.recognition.utils import merge_multi_strings
    >>> merge_multi_strings(['abcdX', 'XdefX', 'XefghX', 'Xijk'], 0.5, 0.5)
    'abcdefghijk'
    """
    if not seq_list:
        return ""
    result = seq_list[0]
    for i in range(1, len(seq_list)):
        text_b = seq_list[i]
        ratio = last_overlap_ratio if i == len(seq_list) - 1 else overlap_ratio
        result = merge_strings(result, text_b, ratio)
    return result
