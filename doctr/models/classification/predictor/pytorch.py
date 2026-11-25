# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from doctr.models._utils import merge_columns, merge_items, predict_by_chunks
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype

__all__ = ["OrientationPredictor", "FontStylePredictor"]


class OrientationPredictor(nn.Module):
    """Implements an object able to detect the reading direction of a text box or a page.
    4 possible orientations: 0, 90, 180, 270 (-90) degrees counter clockwise.

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core classification architecture (backbone + classification head)
        chunk_size: the maximum number of images decoded and processed at once (32 by default). Longer
            inputs, lazy page sources (cf. `DocumentFile.from_images(..., lazy=True)`) and iterators are processed
            chunk by chunk, with a memory footprint that does not grow with the number of images. It is rounded
            up to a multiple of the batch size, so that the predictions are the same as in a single pass.
    """

    def __init__(
        self,
        pre_processor: PreProcessor | None,
        model: nn.Module | None,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("`chunk_size` must be a positive integer")
        self.chunk_size = chunk_size
        self.pre_processor = pre_processor if isinstance(pre_processor, PreProcessor) else None
        self.model = model.eval() if isinstance(model, nn.Module) else None

    @torch.inference_mode()
    def forward(
        self,
        inputs: Sequence[np.ndarray] | Iterable[np.ndarray],
    ) -> list[list[int] | list[float]]:
        """Classify the orientation of pages or crops

        Args:
            inputs: a list of images (numpy arrays of shape H x W x C), a lazy page source
                (cf. `DocumentFile.from_pdf(..., lazy=True)`) or any iterable of images

        Returns:
            the lists of class indices, orientation angles and confidences, aligned with the inputs
        """
        batch_size = self.pre_processor.batch_size if self.pre_processor is not None else 1
        return predict_by_chunks(self._predict, inputs, self.chunk_size, batch_size, merge=merge_columns)

    @torch.inference_mode()
    def _predict(
        self,
        inputs: list[np.ndarray],
    ) -> list[list[int] | list[float]]:
        if len(inputs) == 0:
            return [[], [], []]

        # Dimension check
        if any(input.ndim != 3 for input in inputs):
            raise ValueError("incorrect input shape: all inputs are expected to be multi-channel 2D images.")

        if self.model is None or self.pre_processor is None:
            # predictor is disabled
            return [[0] * len(inputs), [0] * len(inputs), [1.0] * len(inputs)]

        processed_batches = self.pre_processor(inputs)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(  # type: ignore[assignment]
            self.model, processed_batches, _params.device, _params.dtype
        )
        predicted_batches = [self.model(batch) for batch in processed_batches]
        # confidence
        probs = [
            torch.max(torch.softmax(batch.float(), dim=1), dim=1).values.cpu().detach().numpy()
            for batch in predicted_batches
        ]
        # Postprocess predictions
        predicted_batches = [out_batch.argmax(dim=1).cpu().detach().numpy() for out_batch in predicted_batches]

        class_idxs = [int(pred) for batch in predicted_batches for pred in batch]
        classes = [int(self.model.cfg["classes"][idx]) for idx in class_idxs]  # type: ignore
        confs = [round(float(p), 2) for prob in probs for p in prob]

        return [class_idxs, classes, confs]


class FontStylePredictor(nn.Module):
    """Implements an object able to classify the font style of word crops: emphasis (bold, italic, underline,
    strikethrough), text and highlight colors, handwriting, font category and closest font family.

    The model outputs the logits of every style attribute concatenated, in the order of `model.cfg["classes"]`
    (a mapping from each attribute to its classes).

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core classification architecture (backbone + classification head)
        chunk_size: the maximum number of crops decoded and processed at once (1024 by default). Longer
            inputs and iterators are processed chunk by chunk, with a memory footprint that does not grow with the
            number of crops. It is rounded up to a multiple of the batch size, so that the predictions are the same
            as in a single pass.
        margin: the margin the OCR predictor adds on each side of a word box to crop it for this model, relative to
            the short side of the box: the training crops are framed with some background around the text, and a
            tight text box may cut out the underline of a word
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
        chunk_size: int = 1024,
        margin: float = 0.0,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("`chunk_size` must be a positive integer")
        cfg = getattr(model, "cfg", None)
        attributes = cfg.get("classes") if isinstance(cfg, dict) else None
        if not isinstance(attributes, dict) or len(attributes) == 0:
            raise ValueError("the model must define the classes of each style attribute in `model.cfg['classes']`")
        if margin < 0:
            raise ValueError("`margin` must be non-negative")
        self.chunk_size = chunk_size
        self.margin = margin
        self.pre_processor = pre_processor
        self.model = model.eval()
        self.attributes: dict[str, list[Any]] = dict(attributes)

    @torch.inference_mode()
    def forward(
        self,
        inputs: Sequence[np.ndarray] | Iterable[np.ndarray],
    ) -> list[dict[str, dict[str, Any]]]:
        """Classify the font style of word crops

        Args:
            inputs: a list of word crops (numpy arrays of shape H x W x C) or any iterable of crops

        Returns:
            for each crop, the predicted class of each style attribute and its confidence, e.g.
            `{"bold": {"value": True, "confidence": 0.98}, "family": {"value": "Lato", "confidence": 0.61}, ...}`
        """
        return predict_by_chunks(self._predict, inputs, self.chunk_size, self.pre_processor.batch_size, merge_items)

    @torch.inference_mode()
    def _predict(self, inputs: list[np.ndarray]) -> list[dict[str, dict[str, Any]]]:
        if len(inputs) == 0:
            return []

        # Dimension check
        if any(input.ndim != 3 for input in inputs):
            raise ValueError("incorrect input shape: all inputs are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(inputs)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(  # type: ignore[assignment]
            self.model, processed_batches, _params.device, _params.dtype
        )
        logits = torch.cat([self.model(batch).float() for batch in processed_batches], dim=0)
        num_outputs = sum(len(classes) for classes in self.attributes.values())
        if logits.shape[1] != num_outputs:
            raise ValueError(f"the model outputs {logits.shape[1]} logits, expected {num_outputs} (cf. its classes)")

        values: dict[str, list[Any]] = {}
        confidences: dict[str, list[float]] = {}
        for attribute, group in zip(
            self.attributes, torch.split(logits, [len(c) for c in self.attributes.values()], 1)
        ):
            probs, idxs = torch.softmax(group, dim=1).max(dim=1)
            classes = self.attributes[attribute]
            values[attribute] = [classes[idx] for idx in idxs.cpu().tolist()]
            confidences[attribute] = [round(float(p), 2) for p in probs.cpu().tolist()]

        return [
            {
                attribute: {"value": values[attribute][idx], "confidence": confidences[attribute][idx]}
                for attribute in self.attributes
            }
            for idx in range(len(inputs))
        ]
