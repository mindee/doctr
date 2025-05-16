# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from math import ceil, floor

import numpy as np

from ..utils import merge_multi_strings

__all__ = ["split_crops", "remap_preds"]


def split_crops(
    crops: list[np.ndarray],
    max_ratio: float,
    target_ratio: int,
    target_overlap_ratio: float,
    channels_last: bool = True,
) -> tuple[list[np.ndarray], list[int | tuple[int, int, float]], bool]:
    """Split crops horizontally to match a given aspect ratio

    Args:
    ----
        crops: list of numpy array of shape (H, W, 3) if channels_last or (3, H, W) otherwise
        max_ratio: the maximum aspect ratio that won't trigger the splitting
        target_ratio: when a crop are split, it will be split to match this aspect ratio.
          The default value of 4 corresponds to the recognition models' input size of 32 times 128 pixels
        target_overlap_ratio: the target ratio how much a split overlaps with a neighboring split
        channels_last: whether the numpy array has dimensions in channels last order

    Returns:
    -------
        a tuple with the new crops, their mapping, and a boolean specifying whether any remap is required
    """
    _remap_required = False
    splits_map: list[int | tuple[int, int, float]] = []
    new_crops: list[np.ndarray] = []
    for crop in crops:
        h, w = crop.shape[:2] if channels_last else crop.shape[-2:]
        actual_crop_ratio = w / h
        if actual_crop_ratio > max_ratio:
            target_split_width = ceil(h * target_ratio)
            target_split_overlap_width = floor(target_split_width * target_overlap_ratio)

            splits, last_overlap_ratio = _split_horizontally(
                crop,
                target_split_width,
                target_split_overlap_width,
                channels_last,
            )

            # Avoid sending zero-sized crops
            splits = [split for split in splits if all(s > 0 for s in split.shape)]
            # Record the slice of crops
            splits_map.append((len(new_crops), len(new_crops) + len(splits), last_overlap_ratio))
            new_crops.extend(splits)
            # At least one crop will require merging
            _remap_required = True
        else:
            splits_map.append(len(new_crops))
            new_crops.append(crop)

    return new_crops, splits_map, _remap_required


def _split_horizontally(
    image: np.ndarray, split_width: int, split_overlap_width: int, channels_last: bool
) -> tuple[list[np.ndarray], float]:
    image_width = image.shape[1] if channels_last else image.shape[-1]
    if image_width <= split_width:
        return [image], 0

    splits = []
    current_split_start_column = 0
    previous_split_end_column = 0
    last_overlap_ratio = 0.0
    has_reached_end_of_image = False
    while not has_reached_end_of_image:
        current_split_end_column = current_split_start_column + split_width
        current_split_end_column = min(current_split_end_column, image_width)

        # Increase overlap of last split to prevent narrow last split
        has_reached_end_of_image = current_split_end_column == image_width
        if has_reached_end_of_image:
            current_split_start_column = max(0, image_width - split_width)

        if channels_last:
            image_split = image[:, current_split_start_column:current_split_end_column, :]
        else:
            image_split = image[:, :, current_split_start_column:current_split_end_column]

        # Save overlap ratio of the last split, because this one might be larger than other overlap ratios
        if has_reached_end_of_image:
            current_overlap_width = previous_split_end_column - current_split_start_column
            last_overlap_ratio = current_overlap_width / split_width

        splits.append(image_split)
        previous_split_end_column = current_split_end_column
        current_split_start_column = current_split_end_column - split_overlap_width
    return splits, last_overlap_ratio


def remap_preds(
    preds: list[tuple[str, float]], crop_map: list[int | tuple[int, int, float]], split_overlap_ratio: float
) -> list[tuple[str, float]]:
    remapped_out = []
    for _idx in crop_map:
        # Crop hasn't been split
        if isinstance(_idx, int):
            remapped_out.append(preds[_idx])
        else:
            vals, probs = zip(*preds[_idx[0] : _idx[1]])
            last_split_overlap_ratio = _idx[2]
            # Merge the string values
            merged_string = (merge_multi_strings(vals, split_overlap_ratio, last_split_overlap_ratio), min(probs))
            remapped_out.append(merged_string)
    return remapped_out
