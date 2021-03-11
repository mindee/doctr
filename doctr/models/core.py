# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import numpy as np
from typing import List, Any, Tuple
from .detection import DetectionPredictor
from .recognition import RecognitionPredictor
from ._utils import extract_crops
from doctr.documents.elements import Word, Line, Block, Page, Document
from doctr.utils.repr import NestedObject

__all__ = ['OCRPredictor', 'DocumentBuilder']


class OCRPredictor(NestedObject):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    _children_names: List[str] = ['det_predictor', 'reco_predictor', 'doc_builder']

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
    ) -> None:

        self.det_predictor = det_predictor
        self.reco_predictor = reco_predictor
        self.doc_builder = DocumentBuilder()

    def __call__(
        self,
        documents: List[List[np.ndarray]],
        **kwargs: Any,
    ) -> List[Document]:

        # Dimension check
        if any(page.ndim != 3 for doc in documents for page in doc):
            raise ValueError("incorrect input shape: all documents are expected to be list of multi-channel 2D images.")

        pages = [page for doc in documents for page in doc]

        # Localize text elements
        boxes = self.det_predictor(pages, **kwargs)
        # Crop images
        crops = [crop for page, _boxes in zip(pages, boxes) for crop in extract_crops(page, _boxes[:, :4])]
        # Identify character sequences
        char_sequences = self.reco_predictor(crops, **kwargs)

        # Reorganize
        num_pages = [len(doc) for doc in documents]
        results = self.doc_builder(boxes, char_sequences, num_pages, [page.shape[:2] for page in pages])

        return results


class DocumentBuilder(NestedObject):
    """Implements a document builder

    Args:
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs
    """

    def __init__(
        self,
        resolve_lines: bool = False,
        resolve_blocks: bool = False,
        paragraph_break: float = 0.15
    ) -> None:

        self.resolve_lines = resolve_lines

        if resolve_blocks:
            raise NotImplementedError

        self.paragraph_break = paragraph_break

    @staticmethod
    def _sort_boxes(boxes: np.ndarray) -> np.ndarray:
        """Sort bounding boxes from top to bottom, left to right

        Args:
            boxes: bounding boxes of shape (N, 4)

        Returns:
            indices of ordered boxes of shape (N,)
        """

        return (boxes[:, 0] + boxes[:, 3] / np.median(boxes[:, 3] - boxes[:, 1])).argsort()

    def _resolve_lines(self, boxes: np.ndarray, idxs: np.ndarray) -> List[List[int]]:
        """Uses ordered boxes to group them in lines

        Args:
            boxes: bounding boxes of shape (N, 4)
            idxs: indices of ordered boxes of shape (N,)

        Returns:
            nested list of box indices
        """

        # Try to arrange them in lines
        lines = []
        # Add the first word anyway
        words: List[int] = [idxs[0]]
        for idx in idxs[1:]:
            line_break = True

            prev_box = boxes[words[-1]]
            # Reduced vertical diff
            if boxes[idx, 1] < prev_box[[1, 3]].mean():
                # Words horizontally ordered and close
                if (boxes[idx, 0] - prev_box[2]) < self.paragraph_break:
                    line_break = False

            if line_break:
                lines.append(words)
                words = []

            words.append(idx)

        # Use the remaining words to form the last line
        if len(words) > 0:
            lines.append(words)

        return lines

    def _build_blocks(self, boxes: np.ndarray, char_sequences: List[str]) -> List[Block]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 4)
            char_sequences: list of all detected words of the page, of shape N

        Returns:
            list of block elements
        """

        if boxes.shape[0] != len(char_sequences):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(char_sequences)}")

        if boxes.shape[0] == 0:
            return []

        # Sort bounding boxes from top to bottom, left to right
        idxs = self._sort_boxes(boxes[:, :4])

        # Decide whether we try to form lines
        if self.resolve_lines:
            lines = self._resolve_lines(boxes[:, :4], idxs)
        else:
            # One line for all words
            lines = [idxs]

        # No automatic line grouping yet --> 1 block for all lines
        blocks = [
            Block(
                [Line(
                    [Word(
                        char_sequences[idx],
                        boxes[idx, 4],
                        ((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3]))
                    ) for idx in line]
                ) for line in lines]
            )
        ]

        return blocks

    def extra_repr(self) -> str:
        return f"resolve_lines={self.resolve_lines}, paragraph_break={self.paragraph_break}"

    def __call__(
        self,
        boxes: List[np.ndarray],
        char_sequences: List[str],
        num_pages: List[int],
        page_shapes: List[Tuple[int, int]]
    ) -> List[Document]:
        """Re-arrange detected words into structured blocks

        Args:
            boxes: list of localization predictions for all words, of shape (N, 5)
            char_sequences: list of all word values, of size N
            num_pages: number of pages for each document
            page_shape: shape of each page

        Returns:
            list of documents
        """

        # Check the number of crops for each page
        num_crops = [_boxes.shape[0] for _boxes in boxes]
        page_idx, crop_idx = 0, 0
        results = []
        for nb_pages in num_pages:
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
                        page_shapes[page_idx],
                    )
                )
                crop_idx += num_crops[page_idx]
                page_idx += 1
            results.append(Document(_pages))

        return results
