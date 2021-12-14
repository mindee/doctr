# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from doctr.io.elements import Block, Document, Line, Page, Word
from doctr.utils.geometry import rbbox_to_polygon, resolve_enclosing_bbox, resolve_enclosing_rbbox, rotate_boxes
from doctr.utils.repr import NestedObject

__all__ = ['DocumentBuilder']


class DocumentBuilder(NestedObject):
    """Implements a document builder

    Args:
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs
        export_as_straight_boxes: if True, force straight boxes in the export (fit a rectangle
            box to all rotated boxes). Else, keep the boxes format unchanged, no matter what it is.
    """

    def __init__(
        self,
        resolve_lines: bool = True,
        resolve_blocks: bool = True,
        paragraph_break: float = 0.035,
        export_as_straight_boxes: bool = False,
    ) -> None:

        self.resolve_lines = resolve_lines
        self.resolve_blocks = resolve_blocks
        self.paragraph_break = paragraph_break
        self.export_as_straight_boxes = export_as_straight_boxes

    @staticmethod
    def _sort_boxes(boxes: np.ndarray) -> np.ndarray:
        """Sort bounding boxes from top to bottom, left to right

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 5) (in case of rotated bbox)

        Returns:
            indices of ordered boxes of shape (N,)
        """
        if boxes.shape[1] == 5:
            boxes = rotate_boxes(
                loc_preds=np.concatenate((boxes, np.zeros((boxes.shape[0], 1))), -1),
                angle=-np.median(boxes[:, 4]),
                orig_shape=(1024, 1024),
                min_angle=5.,
            )
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
                if boxes.shape[1] == 5:
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
        y_med = np.median(boxes[:, 3] if boxes.shape[1] == 5 else boxes[:, 3] - boxes[:, 1])

        # Sort boxes
        idxs = self._sort_boxes(boxes)

        lines = []
        words = [idxs[0]]  # Assign the top-left word to the first line
        # Define a mean y-center for the line
        if boxes.shape[1] == 5:
            y_center_sum = boxes[idxs[0]][1]
        else:
            y_center_sum = boxes[idxs[0]][[1, 3]].mean()

        for idx in idxs[1:]:
            vert_break = True

            # Compute y_dist
            if boxes.shape[1] == 5:
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
            y_center_sum += boxes[idx][1 if boxes.shape[1] == 5 else [1, 3]].mean()

        # Use the remaining words to form the last(s) line(s)
        if len(words) > 0:
            # Compute sub-lines (horizontal split)
            lines.extend(self._resolve_sub_lines(boxes, words))

        return lines

    @staticmethod
    def _resolve_blocks(boxes: np.ndarray, lines: List[List[int]]) -> List[List[List[int]]]:
        """Order lines to group them in blocks

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 5)
            lines: list of lines, each line is a list of idx

        Returns:
            nested list of box indices
        """
        # Resolve enclosing boxes of lines
        if boxes.shape[1] == 5:
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
        _boxes = boxes
        if self.resolve_lines:
            lines = self._resolve_lines(_boxes[:, :-1])
            # Decide whether we try to form blocks
            if self.resolve_blocks and len(lines) > 1:
                _blocks = self._resolve_blocks(_boxes[:, :-1], lines)
            else:
                _blocks = [lines]
        else:
            # Sort bounding boxes, one line for all boxes, one block for the line
            lines = [self._sort_boxes(_boxes[:, :-1])]
            _blocks = [lines]

        blocks = [
            Block(
                [Line(
                    [
                        Word(
                            *word_preds[idx],
                            (boxes[idx, 0], boxes[idx, 1], boxes[idx, 2], boxes[idx, 3], boxes[idx, 4])
                        ) if boxes.shape[1] == 6 else
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
                f"paragraph_break={self.paragraph_break}, "
                f"export_as_straight_boxes={self.export_as_straight_boxes}")

    def __call__(
        self,
        boxes: List[np.ndarray],
        text_preds: List[List[Tuple[str, float]]],
        page_shapes: List[Tuple[int, int]]
    ) -> Document:
        """Re-arrange detected words into structured blocks

        Args:
            boxes: list of N elements, where each element represents the localization predictions, of shape (*, 5)
                or (*, 6) for all words for a given page
            text_preds: list of N elements, where each element is the list of all word prediction (text + confidence)
            page_shape: shape of each page, of size N

        Returns:
            document object
        """

        if len(boxes) != len(text_preds) or len(boxes) != len(page_shapes):
            raise ValueError("All arguments are expected to be lists of the same size")

        if self.export_as_straight_boxes and len(boxes) > 0:
            # If boxes are already straight OK, else fit a bounding rect
            if boxes[0].shape[-1] == 6:
                straight_boxes = []
                # Iterate over pages
                for page_boxes in boxes:
                    straight_boxes_page = []
                    # Iterate over boxes of the pages
                    for box in page_boxes:
                        x, y, w, h, a, c = box
                        points = rbbox_to_polygon((x, y, w, h, a))
                        x_coords, y_coords = zip(*points)
                        xmin, xmax = min(x_coords), max(x_coords)
                        ymin, ymax = min(y_coords), max(y_coords)
                        straight_boxes_page.append([xmin, ymin, xmax, ymax, c])
                    straight_boxes.append(np.asarray(straight_boxes_page))
                boxes = straight_boxes

        _pages = [
            Page(
                self._build_blocks(
                    page_boxes,
                    word_preds,
                ),
                _idx,
                shape,
            )
            for _idx, shape, page_boxes, word_preds in zip(range(len(boxes)), page_shapes, boxes, text_preds)
        ]

        return Document(_pages)
