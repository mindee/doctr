# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import math

import numpy as np
from rapidfuzz.distance import Levenshtein

from ..._utils import ConfidenceAggregation, aggregate_confidence
from ..utils import exact_overlaps, merge_multi_strings, merge_strings

__all__ = ["split_crops", "remap_preds", "ambiguous_overlaps", "resolve_overlap"]


def split_crops(
    crops: list[np.ndarray],
    max_ratio: float,
    target_ratio: int,
    split_overlap_ratio: float,
) -> tuple[list[np.ndarray], list[int | tuple[int, int, float]], bool]:
    """
    Split crops horizontally if they exceed a given aspect ratio.

    Args:
        crops: List of image crops (H, W, C).
        max_ratio: Aspect ratio threshold above which crops are split.
        target_ratio: Target aspect ratio after splitting (e.g., 4 for 128x32).
        split_overlap_ratio: Desired overlap between splits (as a fraction of split width).

    Returns:
        A tuple containing:
            - The new list of crops (possibly with splits),
            - A mapping indicating how to reassemble predictions,
            - A boolean indicating whether remapping is required.
    """
    if split_overlap_ratio <= 0.0 or split_overlap_ratio >= 1.0:
        raise ValueError(f"Valid range for split_overlap_ratio is (0.0, 1.0), but is: {split_overlap_ratio}")

    remap_required = False
    new_crops: list[np.ndarray] = []
    crop_map: list[int | tuple[int, int, float]] = []

    for crop in crops:
        h, w = crop.shape[:2]

        if h > 0 and w / h > max_ratio:
            split_width = max(1, math.ceil(h * target_ratio))
            overlap_width = max(0, math.floor(split_width * split_overlap_ratio))

            splits, last_overlap = _split_horizontally(crop, split_width, overlap_width)

            # Remove any empty splits
            splits = [s for s in splits if all(dim > 0 for dim in s.shape)]
            if splits:
                crop_map.append((len(new_crops), len(new_crops) + len(splits), last_overlap))
                new_crops.extend(splits)
                remap_required = True
            else:
                # Fallback: treat it as a single crop
                crop_map.append(len(new_crops))
                new_crops.append(crop)
        else:
            crop_map.append(len(new_crops))
            new_crops.append(crop)

    return new_crops, crop_map, remap_required


def _split_horizontally(image: np.ndarray, split_width: int, overlap_width: int) -> tuple[list[np.ndarray], float]:
    """
    Horizontally split a single image with overlapping regions.

    Args:
        image: The image to split (H, W, C).
        split_width: Width of each split.
        overlap_width: Width of the overlapping region.

    Returns:
        - A list of horizontal image slices.
        - The actual overlap ratio of the last split.
    """
    image_width = image.shape[1]
    if image_width <= split_width:
        return [image], 0.0

    starts = _split_starts(image_width, split_width, overlap_width)

    splits = []
    for start_col in starts:
        end_col = start_col + split_width
        splits.append(image[:, start_col:end_col, :])

    # Calculate the last overlap ratio, if only one split no overlap
    last_overlap = 0
    if len(starts) > 1:
        last_overlap = (starts[-2] + split_width) - starts[-1]
    last_overlap_ratio = last_overlap / split_width if split_width else 0.0

    return splits, last_overlap_ratio


def _split_starts(image_width: int, split_width: int, overlap_width: int) -> list[int]:
    """Start columns of the splits of an image wider than ``split_width``, the last one reaching its end"""
    starts = list(range(0, image_width - split_width + 1, split_width - overlap_width))
    if starts[-1] + split_width < image_width:
        starts.append(image_width - split_width)
    return starts


def remap_preds(
    preds: list[tuple[str, float]],
    crop_map: list[int | tuple[int, int, float]],
    overlap_ratio: float,
    confidence_aggregation: ConfidenceAggregation = "min",
    overlaps: dict[tuple[int, int], int] | None = None,
) -> list[tuple[str, float]]:
    """
    Reconstruct predictions from possibly split crops.

    Args:
        preds: List of (text, confidence) tuples from each crop.
        crop_map: Map returned by `split_crops`.
        overlap_ratio: Overlap ratio used during splitting.
        confidence_aggregation: Aggregation method of the confidences of the non-empty parts.
        overlaps: the overlap length of parts ``i`` and ``i + 1`` of crop ``idx``, keyed by ``(idx, i)``, where it was
            resolved (cf. `resolve_overlap`)

    Returns:
        List of merged (text, confidence) tuples corresponding to original crops.
    """
    remapped = []
    for idx, item in enumerate(crop_map):
        if isinstance(item, int):
            remapped.append(preds[item])
        else:
            start_idx, end_idx, last_overlap = item
            text_parts, confidences = zip(*preds[start_idx:end_idx])
            crop_overlaps = {i: n for (crop_idx, i), n in (overlaps or {}).items() if crop_idx == idx}
            merged_text = merge_multi_strings(list(text_parts), overlap_ratio, last_overlap, crop_overlaps)
            text_confidences = [conf for text, conf in zip(text_parts, confidences) if text]
            merged_conf = aggregate_confidence(text_confidences or confidences, confidence_aggregation)
            remapped.append((merged_text, merged_conf))
    return remapped


def _overlap_lengths(a: str, b: str, overlap_ratio: float) -> tuple[dict[int, str], int] | None:
    """The merges of ``a`` and ``b`` at each exact overlap length, and the length `merge_strings` picks, when several
    lengths match exactly and `merge_strings` picks one of them"""
    lengths = exact_overlaps(a, b)
    if len(lengths) < 2:
        return None
    a_crop, b_crop = a[:-1], b[1:]
    merges = {n: a_crop[: len(a_crop) - n] + b_crop for n in lengths}
    merged = merge_strings(a, b, overlap_ratio)
    default = next((n for n, text in merges.items() if text == merged), None)
    return None if default is None else (merges, default)


def ambiguous_overlaps(
    preds: list[tuple[str, float]],
    crop_map: list[int | tuple[int, int, float]],
    crops: list[np.ndarray],
    target_ratio: int,
    split_overlap_ratio: float,
) -> list[tuple[tuple[int, int], np.ndarray, str, str, float, float]]:
    """
    Find the neighboring parts of split crops whose readings overlap exactly at several lengths.

    Across a repeated stretch (a run such as "0000", or a pattern such as "EMEM"), the characters of an overlap do not
    tell its length: `merge_strings` then relies on the length expected from the overlap ratio, which can be a
    character off.

    Args:
        preds: List of (text, confidence) tuples from each part, as read.
        crop_map: Map returned by `split_crops`.
        crops: The crops passed to `split_crops`.
        target_ratio: Target aspect ratio used during splitting.
        split_overlap_ratio: Overlap ratio used during splitting.

    Returns:
        For each such pair of parts ``i`` and ``i + 1`` of crop ``idx``: ``(idx, i)``, the region of the crop that the
        two parts cover, their readings, their overlap ratio, and the offset of part ``i + 1`` from part ``i`` as a
        ratio of the part width.
    """
    pairs = []
    for idx, (item, crop) in enumerate(zip(crop_map, crops)):
        if isinstance(item, int):
            continue
        start_idx, end_idx, last_overlap = item
        texts = [text for text, _ in preds[start_idx:end_idx]]
        height, width = crop.shape[:2]
        split_width = max(1, math.ceil(height * target_ratio))
        starts = _split_starts(width, split_width, max(0, math.floor(split_width * split_overlap_ratio)))
        if len(starts) != len(texts):  # an empty part was dropped
            continue
        for i in range(len(texts) - 1):
            ratio = last_overlap if i == len(texts) - 2 else split_overlap_ratio
            if _overlap_lengths(texts[i], texts[i + 1], ratio) is not None:
                region = crop[:, starts[i] : starts[i + 1] + split_width]
                offset = (starts[i + 1] - starts[i]) / split_width
                pairs.append(((idx, i), region, texts[i], texts[i + 1], ratio, offset))
    return pairs


def resolve_overlap(
    a: str,
    b: str,
    overlap_ratio: float,
    offset: float,
    region_text: str,
    max_distance: int = 2,
    margin: float = 0.25,
) -> int | None:
    """
    Choose the overlap length of two readings that overlap exactly at several lengths.

    Each length is scored by two independent signals, so that a misreading of one rarely decides alone:
    - the edit distance of its merge to ``region_text``, a reading of the region both parts cover (a less squeezed
      crop than the whole one, so it can count a repeated stretch),
    - how far apart, in chars, the overlapping chars sit in the two parts when the chars of each part are spread
      evenly over it.

    Args:
        a: reading of the left part.
        b: reading of the right part.
        overlap_ratio: estimated ratio of overlapping characters, relative to ``len(b)``, used by `merge_strings`.
        offset: start of the right part minus start of the left part, as a ratio of the part width.
        region_text: reading of the region both parts cover.
        max_distance: largest edit distance from ``region_text`` at which a length can replace `merge_strings`' one.
        margin: score by which it must beat `merge_strings`' one.

    Returns:
        The overlap length, or None to keep `merge_strings`' alignment.
    """
    candidates = _overlap_lengths(a, b, overlap_ratio)
    if candidates is None:
        return None
    merges, default = candidates

    def score(n: int) -> tuple[float, int]:
        # Char j of `a` is centred at (j + 0.5) / len(a) of the part width, char j of `b` at offset + (j + 0.5) / len(b)
        gaps = [abs((len(a) - 1 - n + t + 0.5) / len(a) - offset - (t + 1.5) / len(b)) for t in range(n)]
        distance = Levenshtein.distance(merges[n], region_text)
        return distance + len(b) * sum(gaps) / n, distance

    scores = {n: score(n) for n in merges}
    best = min(merges, key=lambda n: (scores[n][0], n != default))
    if best == default or scores[best][1] > max_distance or scores[best][0] + margin > scores[default][0]:
        return None
    return best
