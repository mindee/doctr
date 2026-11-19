import random

import pytest

from doctr.models.recognition.utils import exact_overlaps, merge_multi_strings, merge_strings


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
        # One different letter in overlap: left half of the overlap is taken from "a", right half from "b"
        ("one_differon", "ferent_letter", 0.5, "one_different_letter"),
        # First string empty after crop
        ("-", "test", 0.9, "-test"),
        # Second string empty after crop
        ("test", "-", 0.9, "test-"),
        # Both strings empty / tiny
        ("", "", 0.5, ""),
        ("a", "bcd", 0.5, "abcd"),
        ("abc", "d", 0.5, "abcd"),
        # Longer overlaps
        ("abcdX", "Xdefgh", 0.5, "abcdefgh"),
        ("abcdi", "cdefgh", 0.5, "abcdefgh"),
        ("hello worX", "Xworld again", 0.5, "hello world again"),
        # Repeated digits / punctuation must neither be swallowed nor duplicated
        ("the 0000X", "X000 date", 0.5, "the 0000 date"),
        ("e 0000", "000 d", 0.5, "e 0000 d"),
        ("t ----X", "X--- w", 0.85, "t ---- w"),  # ratio says 3 dashes are shared
        ("t ----X", "X--- w", 0.5, "t ------ w"),  # ratio says 1 dash is shared
        ("......X", "X.....", 0.85, "........"),
        # OCR error in the left half of the overlap -> taken from "a", corrected
        ("the quick browX", "Xbeown fox", 0.6, "the quick brown fox"),  # substitution
        ("internationaX", "Xntional company", 0.47, "international company"),  # deletion
        ("invoice totalX", "Xtoltal amount", 0.5, "invoice total amount"),  # insertion
        # OCR error in the right half of the overlap -> taken from "b", nothing else is damaged
        ("the quick browX", "Xbrovn fox", 0.6, "the quick brovn fox"),
        # 1-char exact match beats a 2-char overlap with one error at equal cost
        ("iban DE8937040044", "04405320", 0.5, "iban DE893704004405320"),
        ("...... committee", "tee naïv", 0.5, "...... committee naïv"),
        # A short coincidental exact match must not beat the real (noisy) overlap
        ("lorem ipsum dolorX", "Xipsvm dolor sit", 0.7, "lorem ipsum dolor sit"),
        # Nothing matches: characters must not be silently dropped
        ("abcdef", "uvwxyz", 0.5, "abcdevwxyz"),
        # ... and with < 1 char of expected overlap the (complete) border chars are kept
        ("abcdef", "uvwxyz", 0.0, "abcdefuvwxyz"),
        ("abcdef", "uvwxyz", 0.1, "abcdefuvwxyz"),
        # Unicode
        ("Straße nX", "Xe naïve", 0.5, "Straße naïve"),
        ("ÄÖÜ äöX", "Xöü ß", 0.5, "ÄÖÜ äöü ß"),
        ("你好世界X", "X世界和平", 0.6, "你好世界和平"),
        ("ok 😀👍X", "X😀👍 yes", 0.5, "ok 😀👍 yes"),
        # Out-of-range ratios are clamped to [0, 1]
        ("abcdX", "Xcdef", 3.0, "abcdef"),
        ("abcdX", "Xcdef", -1.0, "abcdXXcdef"),
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
        # Ambiguous: the last ratio predicts (almost) no overlap, so the single matching "a" is not trusted
        (["aaaa", "aaab", "aabc"], 0.8, 0.3, "aaaaabc"),
        # Realistic doctr-like chunks (8 chars, 50 % overlap)
        (["the 0000", "0000 dat", " date ß ", " ß DE893", "DE893704"], 0.5, 0.5, "the 0000 date ß DE893704"),
        # Single element
        (["abc"], 0.5, 0.4, "abc"),
        # Empty chunks (e.g. blank image splits) are skipped transparently
        (["abcX", "", "Xdef"], 0.5, 0.5, "abcdef"),
        # Handle empty input
        ([], 0.5, 0.4, ""),
    ],
)
def test_merge_multi_strings(seq_list, overlap_ratio, last_overlap_ratio, merged):
    assert merged == merge_multi_strings(seq_list, overlap_ratio, last_overlap_ratio)


@pytest.mark.parametrize(
    "a, b, lengths",
    [
        # A run of zeros: the end of "WARMUP-000" matches the start of "0000001" at 1, 2 and 3 chars
        ("WARMUP-0000", "-0000001", [1, 2, 3]),
        ("abccccX", "Xcccccc", [1, 2, 3, 4]),
        # A repeating pattern: "EME" also matches at "E", "ICI" at "I"
        ("ENDORSEMEN", "SEMENT", [1, 3]),
        ("PARTICIP", "TICIPANTS", [1, 3]),
        # A single exact length
        ("abcdX", "Xdefgh", [1]),
        ("hellowo", "loworld", [2]),
        # Nothing left once the border chars are dropped
        ("abc", "", []),
        ("", "", []),
        ("a", "b", []),
    ],
)
def test_exact_overlaps(a, b, lengths):
    assert exact_overlaps(a, b) == lengths


@pytest.mark.parametrize(
    "overlaps, merged",
    [
        # merge_strings keeps the overlap expected from the ratio, one zero too short
        (None, "WARMUP-00000001"),
        ({}, "WARMUP-00000001"),
        # A known overlap length is imposed
        ({1: 3}, "WARMUP-0000001"),
        ({1: 2}, "WARMUP-00000001"),
        ({1: 1}, "WARMUP-000000001"),
        # ... unless it is not an exact overlap of the merge so far, which is then aligned by merge_strings
        ({1: 5}, "WARMUP-00000001"),
        ({0: 3}, "WARMUP-00000001"),
    ],
)
def test_merge_multi_strings_overlaps(overlaps, merged):
    # The parts of a 416 x 35 crop of "WARMUP-0000001", each read right
    assert merge_multi_strings(["WARMUP", "MUP-0000", "-0000001"], 0.5, 0.519, overlaps) == merged


def test_merge_multi_strings_imposing_its_own_overlaps():
    """Imposing the exact overlap lengths merge_strings picks, or none, merges as merge_strings does."""
    rng = random.Random(0)
    for _ in range(2000):
        alphabet = rng.choice(["0", "01", "0A-", "EMN", "abcdefghij"])
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 24)))
        width = rng.randint(3, 10)
        step = max(1, width // 2 + rng.randint(-1, 1))
        parts = [text[start : start + width] for start in range(0, max(1, len(text) - width) + step, step)]
        ratio, last_ratio = rng.choice([0.3, 0.5, 0.8]), rng.random()
        merged = merge_multi_strings(parts, ratio, last_ratio)

        overlaps, result = {}, parts[0]
        for i, part in enumerate(parts[1:]):
            following = merge_strings(result, part, last_ratio if i == len(parts) - 2 else ratio)
            length = len(result) + len(part) - 2 - len(following)
            if length in exact_overlaps(result, part) and following == result[: len(result) - 1 - length] + part[1:]:
                overlaps[i] = length
            result = following

        assert merge_multi_strings(parts, ratio, last_ratio, overlaps) == merged
        assert merge_multi_strings(parts, ratio, last_ratio, {}) == merged
