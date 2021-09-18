# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import torch
from torch import nn
from typing import List, Any, Union, Tuple

from doctr.models.preprocessor import PreProcessor
from ..utils import merge_multi_strings


__all__ = ['RecognitionPredictor']


class RecognitionPredictor(nn.Module):
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        split_wide_crops: wether to use crop splitting for high aspect ratio crops
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
        split_wide_crops: bool = True,
    ) -> None:

        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()
        self.split_wide_crops = split_wide_crops
        self.critical_ar = 8  # Critical aspect ratio
        self.dil_factor = 1.4  # Dilation factor to overlap the crops
        self.target_ar = 4  # Target aspect ratio

    @torch.no_grad()
    def forward(
        self,
        crops: List[Union[np.ndarray, torch.Tensor]],
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:

        if len(crops) == 0:
            return []
        # Dimension check
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")

        # Split crops that are too wide
        _remap_required = False
        if self.split_wide_crops:
            crop_map: List[Union[int, Tuple[int, int]]] = []
            new_crops: List[np.ndarray] = []
            channels_last = isinstance(crops[0], np.ndarray)
            for crop in crops:
                h, w = crop.shape[:2] if channels_last else crop.shape[-2:]
                aspect_ratio = w / h
                if aspect_ratio > self.critical_ar:
                    # Determine the number of crops, reference aspect ratio = 4 = 128 / 32
                    num_subcrops = int(aspect_ratio // self.target_ar)
                    # Find the new widths, additional dilation factor to overlap crops
                    width = self.dil_factor * w / num_subcrops
                    centers = [(w / num_subcrops) * (1 / 2 + i) for i in range(num_subcrops)]
                    # Record the slice of crops
                    crop_map.append((len(new_crops), len(new_crops) + len(centers)))
                    new_crops.extend(
                        crop[:, max(0, int(round(center - width / 2))): min(w - 1, int(round(center + width / 2))), :]
                        if channels_last else
                        crop[:, :, max(0, int(round(center - width / 2))): min(w - 1, int(round(center + width / 2)))]
                        for center in centers
                    )
                    # At least one crop will require merging
                    _remap_required = True
                else:
                    crop_map.append(len(new_crops))
                    new_crops.append(crop)
            crops = new_crops

        # Resize & batch them
        processed_batches = self.pre_processor(crops)

        # Forward it
        raw = [
            self.model(batch, return_preds=True, **kwargs)['preds']  # type: ignore[operator]
            for batch in processed_batches
        ]

        # Process outputs
        out = [charseq for batch in raw for charseq in batch]

        # Remap crops
        if self.split_wide_crops and _remap_required:
            remapped_out = []
            for _idx in crop_map:
                # Crop hasn't been split
                if isinstance(_idx, int):
                    remapped_out.append(out[_idx])
                else:
                    # unzip
                    vals, probs = zip(out[_idx[0]: _idx[1]])
                    # Merge the string values
                    remapped_out.append(
                        (merge_multi_strings(vals, self.dil_factor), min(probs))  # type: ignore[arg-type]
                    )
            out = remapped_out

        return out
