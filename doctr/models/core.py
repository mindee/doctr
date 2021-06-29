# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from typing import List, Any, Tuple, Dict
from .detection import DetectionPredictor
from .recognition import RecognitionPredictor
from ._utils import extract_crops, extract_rcrops, rotate_page, rotate_boxes
from doctr.documents.elements import Word, Line, Block, Page, Document
from doctr.utils.repr import NestedObject
from doctr.utils.geometry import resolve_enclosing_bbox, resolve_enclosing_rbbox

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
        rotated_bbox: bool = False
    ) -> None:

        self.det_predictor = det_predictor
        self.reco_predictor = reco_predictor
        self.doc_builder = DocumentBuilder(rotated_bbox=rotated_bbox)
        self.extract_crops_fn = extract_rcrops if rotated_bbox else extract_crops

    def __call__(
        self,
        pages: List[np.ndarray],
        **kwargs: Any,
    ) -> Document:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        # Localize text elements
        boxes = self.det_predictor(pages, **kwargs)
        # Crop images, rotate page if necessary
        crops = [crop for page, (_boxes, angle) in zip(pages, boxes) for crop in
                 self.extract_crops_fn(rotate_page(page, -angle), _boxes[:, :-1])]
        # Identify character sequences
        word_preds = self.reco_predictor(crops, **kwargs)

        # Rotate back boxes if necessary
        boxes, angles = zip(*boxes)
        boxes = [rotate_boxes(boxes_page, angle) for boxes_page, angle in zip(boxes, angles)]
        out = self.doc_builder(boxes, word_preds, [page.shape[:2] for page in pages])
        return out


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
        paragraph_break: float = 0.035,
        rotated_bbox: bool = False
    ) -> None:

        self.resolve_lines = resolve_lines
        self.resolve_blocks = resolve_blocks
        self.paragraph_break = paragraph_break
        self.rotated_bbox = rotated_bbox

    def _sort_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Sort bounding boxes from top to bottom, left to right

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 5) (in case of rotated bbox)

        Returns:
            indices of ordered boxes of shape (N,)
        """
        if self.rotated_bbox:
            return (boxes[:, 0] + 2 * boxes[:, 1] / np.median(boxes[:, 3])).argsort()
        return (boxes[:, 0] + 2 * boxes[:, 3] / np.median(boxes[:, 3] - boxes[:, 1])).argsort()

    def _resolve_sub_lines(self, boxes: np.ndarray, words: List[int]) -> List[List[int]]:
        """Split a line in sub_lines

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 5) in case of rotated bbox
            words: list of indexes for the words of the line

        Returns:
            A list of (sub-)lines computed from the original line (words)
        """
        lines = []
        # Sort words horizontally
        words = [words[j] for j in np.argsort([boxes[i, 0] for i in words]).tolist()]
        # Eventually split line horizontally
        if len(words) < 2:
            lines.append(words)
        else:
            sub_line = [words[0]]
            for i in words[1:]:
                horiz_break = True

                prev_box = boxes[sub_line[-1]]
                # Compute distance between boxes
                if self.rotated_bbox:
                    dist = boxes[i, 0] - prev_box[2] / 2 - (prev_box[0] + prev_box[2] / 2)
                else:
                    dist = boxes[i, 0] - prev_box[2]
                # If distance between boxes is lower than paragraph break, same sub-line
                if dist < self.paragraph_break:
                    horiz_break = False

                if horiz_break:
                    lines.append(sub_line)
                    sub_line = []

                sub_line.append(i)
            lines.append(sub_line)

        return lines

    def _resolve_lines(self, boxes: np.ndarray) -> List[List[int]]:
        """Order boxes to group them in lines

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 5) in case of rotated bbox

        Returns:
            nested list of box indices
        """
        # Compute median for boxes heights
        y_med = np.median(boxes[:, 3] if self.rotated_bbox else boxes[:, 3] - boxes[:, 1])

        # Sort boxes
        idxs = (boxes[:, 0] + 2 * boxes[:, 1 if self.rotated_bbox else 3] / y_med).argsort()

        lines = []
        words = [idxs[0]]  # Assign the top-left word to the first line
        # Define a mean y-center for the line
        if self.rotated_bbox:
            y_center_sum = boxes[idxs[0]][1]
        else:
            y_center_sum = boxes[idxs[0]][[1, 3]].mean()

        for idx in idxs[1:]:
            vert_break = True

            # Compute y_dist
            if self.rotated_bbox:
                y_dist = abs(boxes[idx][1] - y_center_sum / len(words))
            else:
                y_dist = abs(boxes[idx][[1, 3]].mean() - y_center_sum / len(words))
            # If y-center of the box is close enough to mean y-center of the line, same line
            if y_dist < y_med / 2:
                vert_break = False

            if vert_break:
                # Compute sub-lines (horizontal split)
                lines.extend(self._resolve_sub_lines(boxes, words))
                words = []
                y_center_sum = 0

            words.append(idx)
            y_center_sum = boxes[idxs[0]][1 if self.rotated_bbox else [1, 3]].mean()

        # Use the remaining words to form the last(s) line(s)
        if len(words) > 0:
            # Compute sub-lines (horizontal split)
            lines.extend(self._resolve_sub_lines(boxes, words))

        return lines

    def _resolve_blocks(self, boxes: np.ndarray, lines: List[List[int]]) -> List[List[List[int]]]:
        """Order lines to group them in blocks

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 5)
            lines: list of lines, each line is a list of idx

        Returns:
            nested list of box indices
        """
        # Resolve enclosing boxes of lines
        if self.rotated_bbox:
            box_lines = np.asarray([
                resolve_enclosing_rbbox([tuple(boxes[idx, :5]) for idx in line]) for line in lines  # type: ignore[misc]
            ])
        else:
            _box_lines = [
                resolve_enclosing_bbox([
                    (tuple(boxes[idx, :2]), tuple(boxes[idx, 2:])) for idx in line  # type: ignore[misc]
                ])
                for line in lines
            ]
            box_lines = np.asarray([(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in _box_lines])

        # Compute geometrical features of lines to clusterize
        # Clusterizing only with box centers yield to poor results for complex documents
        box_features = np.stack(
            (
                (box_lines[:, 0] + box_lines[:, 3]) / 2,
                (box_lines[:, 1] + box_lines[:, 2]) / 2,
                (box_lines[:, 0] + box_lines[:, 2]) / 2,
                (box_lines[:, 1] + box_lines[:, 3]) / 2,
                box_lines[:, 0],
                box_lines[:, 1],
            ), axis=-1
        )
        # Compute clusters
        clusters = fclusterdata(box_features, t=0.1, depth=4, criterion='distance', metric='euclidean')

        _blocks: Dict[int, List[int]] = {}
        # Form clusters
        for line_idx, cluster_idx in enumerate(clusters):
            if cluster_idx in _blocks.keys():
                _blocks[cluster_idx].append(line_idx)
            else:
                _blocks[cluster_idx] = [line_idx]

        # Retrieve word-box level to return a fully nested structure
        blocks = [[lines[idx] for idx in block] for block in _blocks.values()]

        return blocks

    def _build_blocks(self, boxes: np.ndarray, word_preds: List[Tuple[str, float]]) -> List[Block]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 5) or (N, 6)
            word_preds: list of all detected words of the page, of shape N

        Returns:
            list of block elements
        """

        if boxes.shape[0] != len(word_preds):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(word_preds)}")

        if boxes.shape[0] == 0:
            return []

        # Decide whether we try to form lines
        if self.resolve_lines:
            lines = self._resolve_lines(boxes[:, :-1])
            # Decide whether we try to form blocks
            if self.resolve_blocks:
                _blocks = self._resolve_blocks(boxes[:, :-1], lines)
            else:
                _blocks = [lines]
        else:
            # Sort bounding boxes, one line for all boxes, one block for the line
            lines = [self._sort_boxes(boxes[:, :-1])]
            _blocks = [lines]

        blocks = [
            Block(
                [Line(
                    [
                        Word(
                            *word_preds[idx],
                            (boxes[idx, 0], boxes[idx, 1], boxes[idx, 2], boxes[idx, 3], boxes[idx, 4])
                        ) if self.rotated_bbox else
                        Word(
                            *word_preds[idx],
                            ((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3]))
                        ) for idx in line
                    ]
                ) for line in lines]
            ) for lines in _blocks
        ]

        return blocks

    def extra_repr(self) -> str:
        return (f"resolve_lines={self.resolve_lines}, resolve_blocks={self.resolve_blocks}, "
                f"paragraph_break={self.paragraph_break}")

    def __call__(
        self,
        boxes: List[np.ndarray],
        word_preds: List[Tuple[str, float]],
        page_shapes: List[Tuple[int, int]]
    ) -> Document:
        """Re-arrange detected words into structured blocks

        Args:
            boxes: list of localization predictions for all words, of shape (N, 5) or (N, 6)
            word_preds: list of all word values, of size N
            page_shape: shape of each page

        Returns:
            list of documents
        """

        # Check the number of crops for each page
        page_idx, crop_idx = 0, 0
        _pages = []
        for page_boxes in boxes:
            # Assemble all detected words into structured blocks
            _pages.append(
                Page(
                    self._build_blocks(
                        page_boxes,
                        word_preds[crop_idx: crop_idx + page_boxes.shape[0]]
                    ),
                    page_idx,
                    page_shapes[page_idx],
                )
            )
            crop_idx += page_boxes.shape[0]
            page_idx += 1

        return Document(_pages)
