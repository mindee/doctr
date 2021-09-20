# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from typing import List, Tuple, Union
from ..utils import merge_multi_strings

__all__ = ['split_crops', 'remap_preds']


def split_crops(
    crops: List[np.ndarray],
    max_ratio: float,
    target_ratio: int,
    dilation: float,
    channels_last: bool = True,
) -> Tuple[List[np.ndarray], List[Union[int, Tuple[int, int]]], bool]:

    _remap_required = False
    crop_map: List[Union[int, Tuple[int, int]]] = []
    new_crops: List[np.ndarray] = []
    for crop in crops:
        h, w = crop.shape[:2] if channels_last else crop.shape[-2:]
        aspect_ratio = w / h
        if aspect_ratio > max_ratio:
            # Determine the number of crops, reference aspect ratio = 4 = 128 / 32
            num_subcrops = int(aspect_ratio // target_ratio)
            # Find the new widths, additional dilation factor to overlap crops
            width = dilation * w / num_subcrops
            centers = [(w / num_subcrops) * (1 / 2 + i) for i in range(num_subcrops)]
            # Record the slice of crops
            crop_map.append((len(new_crops), len(new_crops) + len(centers)))
            if channels_last:
                new_crops.extend(
                    crop[:, max(0, int(round(center - width / 2))): min(w - 1, int(round(center + width / 2))), :]
                    for center in centers
                )
            else:
                new_crops.extend(
                    crop[:, :, max(0, int(round(center - width / 2))): min(w - 1, int(round(center + width / 2)))]
                    for center in centers
                )
            # At least one crop will require merging
            _remap_required = True
        else:
            crop_map.append(len(new_crops))
            new_crops.append(crop)

    return new_crops, crop_map, _remap_required


def remap_preds(
    preds: List[Tuple[str, float]],
    crop_map: List[Union[int, Tuple[int, int]]],
    dilation: float
) -> List[Tuple[str, float]]:
    remapped_out = []
    for _idx in crop_map:
        # Crop hasn't been split
        if isinstance(_idx, int):
            remapped_out.append(preds[_idx])
        else:
            # unzip
            vals, probs = zip(*preds[_idx[0]: _idx[1]])
            # Merge the string values
            remapped_out.append(
                (merge_multi_strings(vals, dilation), min(probs))  # type: ignore[arg-type]
            )
    return remapped_out
