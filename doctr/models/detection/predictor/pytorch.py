# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List, Union

import numpy as np
import torch
from torch import nn
import os
from doctr.models.preprocessor import PreProcessor

__all__ = ["DetectionPredictor"]

from doctr.utils.gpu import select_gpu_device


class DetectionPredictor(nn.Module):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.pre_processor = pre_processor
        self.postprocessor = self.model.postprocessor

        detected_device, selected_device = select_gpu_device()
        if "onnx" in str((type(self.model))):
            selected_device = 'cpu'
            # self.model = nn.DataParallel(self.model)
            # self.model = self.model.half()
        self.device = torch.device(selected_device)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def forward(
        self,
        pages: List[Union[np.ndarray, torch.Tensor]],
        return_model_output=False,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        predicted_batches = []

        for batch in processed_batches:
            if "onnx" not in str((type(self.model))):
                batch = batch.to(self.device)
                # if self.device == torch.device("cuda"):
                #     batch = batch.half()
            pred_map = self.model(batch)
            if type(pred_map) == torch.Tensor:
                pred_map = pred_map.detach().cpu().numpy()
            if return_model_output:
                return pred_map.astype(np.float32)
            pred_map = np.transpose(pred_map, (0, 2, 3, 1))
            predicted_batches += [pred[0] for pred in self.postprocessor(pred_map)]
        return predicted_batches
