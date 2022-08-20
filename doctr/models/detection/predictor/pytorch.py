# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from concurrent.futures import process
from typing import Any, List, Union

import numpy as np
import torch
from torch import nn

from doctr.models.preprocessor import PreProcessor
from openvino.runtime import Core

__all__ = ['DetectionPredictor']


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
        self.pre_processor = pre_processor
        self.model = model.eval()

        
        self.ie = Core()
        model_onnx = self.ie.read_model(model="det.onnx")
        self.compiled_model_onnx = self.ie.compile_model(model=model_onnx, device_name="CPU")
        self.output_layer_onnx = self.compiled_model_onnx.output(0)

    @torch.no_grad()
    def forward(
        self,
        pages: List[Union[np.ndarray, torch.Tensor]],
        **kwargs: Any,
    ) -> List[np.ndarray]:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        _device = next(self.model.parameters()).device
        predicted_batches = []
        for batch in processed_batches:
            print(batch.shape)
            pred_map = self.compiled_model_onnx([batch.detach().cpu().numpy()])[self.output_layer_onnx]
            pred_map = np.transpose(pred_map, (0, 2, 3, 1))
            predicted_batches += [pred[0] for pred in self.model.postprocessor(pred_map)]
        return predicted_batches
