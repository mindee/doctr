# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import math

import numpy as np

from ..utils import merge_multi_strings

__all__ = ["split_crops", "remap_preds"]


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
        aspect_ratio = w / h

        if aspect_ratio > max_ratio:
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

    # Compute start columns for each split
    step = split_width - overlap_width
    starts = list(range(0, image_width - split_width + 1, step))

    # Ensure the last patch reaches the end of the image
    if starts[-1] + split_width < image_width:
        starts.append(image_width - split_width)

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


def remap_preds(
    preds: list[tuple[str, float]],
    crop_map: list[int | tuple[int, int, float]],
    overlap_ratio: float,
) -> list[tuple[str, float]]:
    """
    Reconstruct predictions from possibly split crops.

    Args:
        preds: List of (text, confidence) tuples from each crop.
        crop_map: Map returned by `split_crops`.
        overlap_ratio: Overlap ratio used during splitting.

    Returns:
        List of merged (text, confidence) tuples corresponding to original crops.
    """
    remapped = []
    for item in crop_map:
        if isinstance(item, int):
            remapped.append(preds[item])
        else:
            start_idx, end_idx, last_overlap = item
            text_parts, confidences = zip(*preds[start_idx:end_idx])
            merged_text = merge_multi_strings(list(text_parts), overlap_ratio, last_overlap)
            merged_conf = sum(confidences) / len(confidences)  # average confidence
            remapped.append((merged_text, merged_conf))
    return remapped
