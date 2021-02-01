# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict
from .preprocessor import PreProcessor
from .detection import DetectionPredictor
from .recognition import RecognitionPredictor
from ._utils import extract_crops
from ..documents.elements import Word, Line, Block, Page, Document

__all__ = ['OCRPredictor']


class OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
    ) -> None:

        self.det_predictor = det_predictor
        self.reco_predictor = reco_predictor

    @staticmethod
    def _build_blocks(boxes: np.ndarray, char_sequences: List[str]) -> List[Block]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 4)
            char_sequences: list of all detected words of the page, of shape N

        Returns:
            list of block elements
        """

        if boxes.shape[0] != len(char_sequences):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(char_sequences)}")

        # Sort bounding boxes from top to bottom, left to right
        idxs = (boxes[:, 0] + boxes[:, 1] * (boxes[:, 2] - boxes[:, 0])).argsort()
        # Try to arrange them in lines
        lines = []
        words: List[int] = []
        for idx in idxs:

            if len(words) == 0:
                words = [idx]
                continue
            # Check horizontal gaps
            horz_gap = boxes[idx, 0] - boxes[words[-1], 0] < 0.5 * (boxes[words[-1], 2] - boxes[words[-1], 0])
            # Check vertical gaps
            vert_gap = abs(boxes[idx, 3] - boxes[words[-1], 3]) > 0.5 * (boxes[words[-1], 3] - boxes[words[-1], 1])

            if horz_gap or vert_gap:
                lines.append(words)
                words.clear()

            words.append(idx)

        if len(words) > 0:
            lines.append(words)

        blocks = [
            Block([Line(
                [Word(
                    char_sequences[idx],
                    boxes[idx, 4],
                    ((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3]))
                ) for idx in line]
            )]) for line in lines
        ]

        return blocks

    def __call__(
        self,
        documents: List[List[np.ndarray]],
    ) -> List[Document]:

        pages = [page for doc in documents for page in doc]

        # Localize text elements
        boxes = self.det_predictor(pages)
        # Crop images
        crops = [crop for page, _boxes in zip(pages, boxes) for crop in extract_crops(page, _boxes[:, :4])]
        # Identify character sequences
        char_sequences = self.reco_predictor(crops)

        # Reorganize
        num_pages = [len(doc) for doc in documents]
        num_crops = [_boxes.shape[0] for _boxes in boxes]
        page_idx, crop_idx = 0, 0
        results = []
        for doc_idx, nb_pages in enumerate(num_pages):
            _pages = []
            for page_boxes in boxes[page_idx: page_idx + nb_pages]:
                # Assemble all detected words into structured blocks
                _pages.append(
                    Page(
                        self._build_blocks(
                            page_boxes[:num_crops[page_idx]],
                            char_sequences[crop_idx: crop_idx + num_crops[page_idx]]
                        ),
                        page_idx,
                        pages[page_idx].shape[:2],
                    )
                )
                crop_idx += num_crops[page_idx]
                page_idx += 1
            results.append(Document(_pages))

        return results
