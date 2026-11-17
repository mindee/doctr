# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from doctr.models._utils import predict_by_chunks
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype
from doctr.utils.geometry import extract_crops, extract_rcrops

__all__ = ["FormulaPredictor"]


def _crop_region(page: np.ndarray, region: np.ndarray) -> np.ndarray | None:
    """Crop a page-relative region, a straight box (4,) or a polygon (4, 2), None for a degenerate region"""
    region = np.asarray(region, dtype=np.float32)
    if region.ndim == 1:
        crop = extract_crops(page, np.clip(region, 0, 1)[None])[0]
    else:
        crop = extract_rcrops(page, np.clip(region, 0, 1)[None], assume_horizontal=False)[0]
    return crop if crop.shape[0] >= 2 and crop.shape[1] >= 2 else None


class FormulaPredictor(nn.Module):
    """Implements an object able to recognize mathematical formulas as LaTeX strings

    Each formula image is resized (keeping its aspect ratio), padded and normalized. The margins around a formula
    (e.g. of a layout region) are kept: the model is trained to read formulas with margins.

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core formula recognition architecture
        chunk_size: the maximum number of formula images decoded and processed at once (128 by default). Longer
            inputs, lazy page sources (cf. `DocumentFile.from_images(..., lazy=True)`) and iterators are processed
            chunk by chunk, with a memory footprint that does not grow with the number of images. It is rounded
            up to a multiple of the batch size, so that the predictions are the same as in a single pass.
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
        chunk_size: int = 128,
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
        crops: Sequence[np.ndarray] | Iterable[np.ndarray],
        **kwargs: Any,
    ) -> list[tuple[str, float]]:
        """Recognize the formula of each image

        Args:
            crops: a list of formula images (numpy arrays of shape H x W x C), a lazy page source
                (cf. `DocumentFile.from_images(..., lazy=True)`) or any iterable of images
            **kwargs: keyword arguments forwarded to the model

        Returns:
            the LaTeX string (tokens separated by spaces) and the confidence of each image
        """
        return predict_by_chunks(
            lambda chunk: self._predict(chunk, **kwargs), crops, self.chunk_size, self.pre_processor.batch_size
        )

    @torch.inference_mode()
    def _predict(self, crops: Sequence[np.ndarray], **kwargs: Any) -> list[tuple[str, float]]:
        if len(crops) == 0:
            return []
        # Dimension check
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")
        empty = next((crop for crop in crops if not all(s > 0 for s in crop.shape)), None)
        if empty is not None:
            raise ValueError(f"incorrect input shape: crops are expected to be non-empty, got shape {empty.shape}.")

        processed_batches = self.pre_processor(list(crops))
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(  # type: ignore[assignment]
            self.model, processed_batches, _params.device, _params.dtype
        )
        raw = [self.model(batch, return_preds=True, **kwargs)["preds"] for batch in processed_batches]
        return [pred for batch in raw for pred in batch]

    @torch.inference_mode()
    def predict_regions(
        self,
        pages: list[np.ndarray],
        regions: list[dict[str, Any] | None],
        class_name: str = "Formula",
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Recognize the formulas a layout model located on pages

        Args:
            pages: the page images
            regions: the per-page layout predictions `{"class_names", "boxes", "scores"}` (None for none)
            class_name: the layout class of the formula regions
            **kwargs: keyword arguments forwarded to the model

        Returns:
            a per-page list of formulas `{"geometry", "value", "confidence", "score"}`, with the geometry of the
            layout region (page-relative) and its layout score
        """
        crops: list[np.ndarray] = []
        meta: list[tuple[int, np.ndarray, float | None]] = []
        for p_idx, (page, region) in enumerate(zip(pages, regions)):
            region = region or {}
            scores = region.get("scores")
            for r_idx, (cls_name, box) in enumerate(zip(region.get("class_names", []), region.get("boxes", []))):
                crop = _crop_region(page, box) if cls_name == class_name else None
                if crop is not None:
                    crops.append(crop)
                    meta.append((p_idx, np.asarray(box), None if scores is None else float(scores[r_idx])))

        formulas: list[list[dict[str, Any]]] = [[] for _ in pages]
        for (p_idx, box, score), (latex, confidence) in zip(meta, self(crops, **kwargs) if crops else []):
            formulas[p_idx].append({"geometry": box, "value": latex, "confidence": confidence, "score": score})
        return formulas
