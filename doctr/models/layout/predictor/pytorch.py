# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from doctr.models._utils import predict_by_chunks
from doctr.models.detection._utils import _remove_padding
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype

__all__ = ["LayoutPredictor"]


class LayoutPredictor(nn.Module):
    """Implements an object able to localize layout elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core layout architecture
        chunk_size: the maximum number of pages decoded and processed at once (32 by default). Longer
            inputs, lazy page sources (cf. `DocumentFile.from_images(..., lazy=True)`) and iterators are processed
            chunk by chunk, with a memory footprint that does not grow with the number of pages. It is rounded
            up to a multiple of the batch size, so that the predictions are the same as in a single pass.
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()
        if chunk_size < 1:
            raise ValueError("`chunk_size` must be a positive integer")
        self.chunk_size = chunk_size

    @torch.inference_mode()
    def forward(
        self,
        pages: Sequence[np.ndarray] | Iterable[np.ndarray],
        **kwargs: Any,
    ) -> list[dict[str, np.ndarray]]:
        """Detect the layout regions of pages

        Args:
            pages: a list of pages (numpy arrays of shape H x W x C), a lazy page source
                (cf. `DocumentFile.from_pdf(..., lazy=True)`) or any iterable of pages
            **kwargs: keyword arguments forwarded to the model

        Returns:
            the regions of each page: `{"class_names", "boxes", "scores"}`
        """
        return predict_by_chunks(
            lambda chunk: self._predict(chunk, **kwargs), pages, self.chunk_size, self.pre_processor.batch_size
        )

    @torch.inference_mode()
    def _predict(
        self,
        pages: list[np.ndarray],
        **kwargs: Any,
    ) -> list[dict[str, np.ndarray]]:
        if len(pages) == 0:
            return []
        # Extract parameters from the preprocessor
        preserve_aspect_ratio = self.pre_processor.resize.preserve_aspect_ratio
        symmetric_pad = self.pre_processor.resize.symmetric_pad
        assume_straight_pages = self.model.assume_straight_pages
        # This flag is needed to return the padding mask from the preprocessor
        self.pre_processor.resize.return_padding_mask = True

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(  # type: ignore[assignment]
            self.model, processed_batches, _params.device, _params.dtype
        )
        predicted_batches = [
            self.model(batch[0], batch[1], return_preds=True, return_model_output=True, **kwargs)
            for batch in processed_batches
        ]
        # remap idx to class names
        class_names = [
            [self.model.class_names[int(i)] for i in pred[0]]  # type: ignore[index]
            for batch in predicted_batches
            for pred in batch["preds"]
        ]
        boxes = [pred[1] for batch in predicted_batches for pred in batch["preds"]]
        scores = [pred[2] for batch in predicted_batches for pred in batch["preds"]]

        # Remove padding from loc predictions
        preds = _remove_padding(
            pages,
            [{"pred": box} for box in boxes],
            preserve_aspect_ratio=preserve_aspect_ratio,
            symmetric_pad=symmetric_pad,
            assume_straight_pages=assume_straight_pages,  # type: ignore[arg-type]
        )
        return [
            {"class_names": class_name, "boxes": pred["pred"], "scores": score}  # type: ignore[dict-item]
            for class_name, pred, score in zip(class_names, preds, scores)
        ]
