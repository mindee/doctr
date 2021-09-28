# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import torch
from torch import nn
from typing import List, Any, Union

from doctr.io.elements import Document
from doctr.models.builder import DocumentBuilder
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from .base import _OCRPredictor


__all__ = ['OCRPredictor']


class OCRPredictor(nn.Module, _OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
        rotated_bbox: bool = False
    ) -> None:

        super().__init__()
        self.det_predictor = det_predictor.eval()  # type: ignore[attr-defined]
        self.reco_predictor = reco_predictor.eval()  # type: ignore[attr-defined]
        self.doc_builder = DocumentBuilder(rotated_bbox=rotated_bbox)

    @torch.no_grad()
    def forward(
        self,
        pages: List[Union[np.ndarray, torch.Tensor]],
        **kwargs: Any,
    ) -> Document:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        # Localize text elements
        loc_preds = self.det_predictor(pages, **kwargs)
        # Check whether crop mode should be switched to channels first
        channels_last = len(pages) == 0 or isinstance(pages[0], np.ndarray)
        # Crop images, rotate page if necessary
        crops, loc_preds = self._prepare_crops(pages, loc_preds, channels_last=channels_last)
        # Identify character sequences
        word_preds = self.reco_predictor([crop for page_crops in crops for crop in page_crops], **kwargs)

        boxes, text_preds = self._process_predictions(loc_preds, word_preds, self.doc_builder.rotated_bbox)
        out = self.doc_builder(
            boxes,
            text_preds,
            [
                page.shape[:2] if channels_last else page.shape[-2:]  # type: ignore[misc]
                for page in pages
            ]
        )
        return out
