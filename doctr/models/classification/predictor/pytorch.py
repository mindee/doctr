# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List, Union

import numpy as np
import torch
from torch import nn

from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype

__all__ = ["OrientationPredictor"]


class OrientationPredictor(nn.Module):
    """Implements an object able to detect the reading direction of a text box or a page.
    4 possible orientations: 0, 90, 180, 270 (-90) degrees counter clockwise.

    Args:
    ----
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

    @torch.inference_mode()
    def forward(
        self,
        inputs: List[Union[np.ndarray, torch.Tensor]],
    ) -> List[Union[List[int], List[float]]]:
        # Dimension check
        if any(input.ndim != 3 for input in inputs):
            raise ValueError("incorrect input shape: all inputs are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(inputs)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(
            self.model, processed_batches, _params.device, _params.dtype
        )
        predicted_batches = [self.model(batch) for batch in processed_batches]
        # confidence
        probs = [
            torch.max(torch.softmax(batch, dim=1), dim=1).values.cpu().detach().numpy() for batch in predicted_batches
        ]
        # Postprocess predictions
        predicted_batches = [out_batch.argmax(dim=1).cpu().detach().numpy() for out_batch in predicted_batches]

        class_idxs = [int(pred) for batch in predicted_batches for pred in batch]
        classes = [int(self.model.cfg["classes"][idx]) for idx in class_idxs]
        confs = [round(float(p), 2) for prob in probs for p in prob]

        return [class_idxs, classes, confs]
