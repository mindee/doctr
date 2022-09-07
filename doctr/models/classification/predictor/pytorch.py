# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List, Union

import numpy as np
import torch
from torch import nn

from doctr.models.preprocessor import PreProcessor

__all__ = ["CropOrientationPredictor"]


class CropOrientationPredictor(nn.Module):
    """Implements an object able to detect the reading direction of a text box.
    4 possible orientations: 0, 90, 180, 270 degrees counter clockwise.

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core classification architecture (backbone + classification head)
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
    ) -> None:

        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()

    @torch.no_grad()
    def forward(
        self,
        crops: List[Union[np.ndarray, torch.Tensor]],
    ) -> List[int]:

        # Dimension check
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(crops)
        _device = next(self.model.parameters()).device
        predicted_batches = [self.model(batch).to(device=_device) for batch in processed_batches]

        # Postprocess predictions
        predicted_batches = [out_batch.argmax(dim=1).cpu().detach().numpy() for out_batch in predicted_batches]

        return [int(pred) for batch in predicted_batches for pred in batch]
