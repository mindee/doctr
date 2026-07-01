# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from doctr.io.elements import (
    Block,
    Document,
    KIEDocument,
    KIEPage,
    LayoutElement,
    Line,
    Page,
    Prediction,
    Table,
    TableCell,
    Word,
)
from doctr.utils.geometry import estimate_page_angle, resolve_enclosing_bbox, resolve_enclosing_rbbox, rotate_boxes
from doctr.utils.repr import NestedObject

__all__ = ["DocumentBuilder"]


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
        resolve_blocks: bool = False,
        paragraph_break: float = 0.035,
        export_as_straight_boxes: bool = False,
    ) -> None:
        self.resolve_lines = resolve_lines
        self.resolve_blocks = resolve_blocks
        self.paragraph_break = paragraph_break
        self.export_as_straight_boxes = export_as_straight_boxes

    @staticmethod
    def _sort_boxes(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sort bounding boxes from top to bottom, left to right

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 4, 2) (in case of rotated bbox)

        Returns:
            tuple: indices of ordered boxes of shape (N,), boxes
                If straight boxes are passed tpo the function, boxes are unchanged
                else: boxes returned are straight boxes fitted to the straightened rotated boxes
                so that we fit the lines afterwards to the straigthened page
        """
        if boxes.ndim == 3:
            boxes = rotate_boxes(
                loc_preds=boxes,
                angle=-estimate_page_angle(boxes),
                orig_shape=(1024, 1024),
                min_angle=5.0,
            )
            boxes = np.concatenate((boxes.min(1), boxes.max(1)), -1)
        return (boxes[:, 0] + 2 * boxes[:, 3] / np.median(boxes[:, 3] - boxes[:, 1])).argsort(), boxes

    def _resolve_sub_lines(self, boxes: np.ndarray, word_idcs: list[int]) -> list[list[int]]:
        """Split a line in sub_lines

        Args:
            boxes: bounding boxes of shape (N, 4)
            word_idcs: list of indexes for the words of the line

        Returns:
            A list of (sub-)lines computed from the original line (words)
        """
        lines = []
        # Sort words horizontally
        word_idcs = [word_idcs[idx] for idx in boxes[word_idcs, 0].argsort().tolist()]

        # Eventually split line horizontally
        if len(word_idcs) < 2:
            lines.append(word_idcs)
        else:
            sub_line = [word_idcs[0]]
            for i in word_idcs[1:]:
                horiz_break = True

                prev_box = boxes[sub_line[-1]]
                # Compute distance between boxes
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

    def _resolve_lines(self, boxes: np.ndarray) -> list[list[int]]:
        """Order boxes to group them in lines

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 4, 2) in case of rotated bbox

        Returns:
            nested list of box indices
        """
        # Sort boxes, and straighten the boxes if they are rotated
        idxs, boxes = self._sort_boxes(boxes)

        # Compute median for boxes heights
        y_med = np.median(boxes[:, 3] - boxes[:, 1])

        lines = []
        words = [idxs[0]]  # Assign the top-left word to the first line
        # Define a mean y-center for the line
        y_center_sum = boxes[idxs[0]][[1, 3]].mean()

        for idx in idxs[1:]:
            vert_break = True

            # Compute y_dist
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
            y_center_sum += boxes[idx][[1, 3]].mean()

        # Use the remaining words to form the last(s) line(s)
        if len(words) > 0:
            # Compute sub-lines (horizontal split)
            lines.extend(self._resolve_sub_lines(boxes, words))

        return lines

    @staticmethod
    def _resolve_blocks(boxes: np.ndarray, lines: list[list[int]]) -> list[list[list[int]]]:
        """Order lines to group them in blocks

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 4, 2)
            lines: list of lines, each line is a list of idx

        Returns:
            nested list of box indices
        """
        # Resolve enclosing boxes of lines
        if boxes.ndim == 3:
            box_lines: np.ndarray = np.asarray([
                resolve_enclosing_rbbox([tuple(boxes[idx, :, :]) for idx in line])  # type: ignore[misc]
                for line in lines
            ])
        else:
            _box_lines = [
                resolve_enclosing_bbox([(tuple(boxes[idx, :2]), tuple(boxes[idx, 2:])) for idx in line])
                for line in lines
            ]
            box_lines = np.asarray([(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in _box_lines])

        # Compute geometrical features of lines to clusterize
        # Clusterizing only with box centers yield to poor results for complex documents
        if boxes.ndim == 3:
            box_features: np.ndarray = np.stack(
                (
                    (box_lines[:, 0, 0] + box_lines[:, 0, 1]) / 2,
                    (box_lines[:, 0, 0] + box_lines[:, 2, 0]) / 2,
                    (box_lines[:, 0, 0] + box_lines[:, 2, 1]) / 2,
                    (box_lines[:, 0, 1] + box_lines[:, 2, 1]) / 2,
                    (box_lines[:, 0, 1] + box_lines[:, 2, 0]) / 2,
                    (box_lines[:, 2, 0] + box_lines[:, 2, 1]) / 2,
                ),
                axis=-1,
            )
        else:
            box_features = np.stack(
                (
                    (box_lines[:, 0] + box_lines[:, 3]) / 2,
                    (box_lines[:, 1] + box_lines[:, 2]) / 2,
                    (box_lines[:, 0] + box_lines[:, 2]) / 2,
                    (box_lines[:, 1] + box_lines[:, 3]) / 2,
                    box_lines[:, 0],
                    box_lines[:, 1],
                ),
                axis=-1,
            )
        # Compute clusters
        clusters = fclusterdata(box_features, t=0.1, depth=4, criterion="distance", metric="euclidean")

        _blocks: dict[int, list[int]] = {}
        # Form clusters
        for line_idx, cluster_idx in enumerate(clusters):
            if cluster_idx in _blocks.keys():
                _blocks[cluster_idx].append(line_idx)
            else:
                _blocks[cluster_idx] = [line_idx]

        # Retrieve word-box level to return a fully nested structure
        blocks = [[lines[idx] for idx in block] for block in _blocks.values()]

        return blocks

    @staticmethod
    def _build_layout_elements(regions: dict[str, Any] | None) -> list[LayoutElement]:
        """Convert a raw layout prediction into exportable `LayoutElement` objects.

        Args:
            regions: a layout prediction `{"boxes": (R, 4) | (R, 4, 2), "class_names": [...], "scores": [...]}`
                as returned by a `LayoutPredictor`, or None.

        Returns:
            list of `LayoutElement` (empty if no layout was provided).
        """
        if regions is None or len(regions.get("boxes", [])) == 0:
            return []
        boxes = np.asarray(regions["boxes"])
        class_names = regions.get("class_names") or ["" for _ in range(len(boxes))]
        scores = regions.get("scores")
        scores = scores if scores is not None else [1.0 for _ in range(len(boxes))]

        elements: list[LayoutElement] = []
        for box, cname, score in zip(boxes, class_names, scores):
            if box.ndim == 2:  # rotated polygon (4, 2)
                geometry: Any = tuple(tuple(float(c) for c in pt) for pt in box.tolist())
            else:  # straight (x1, y1, x2, y2)
                geometry = ((float(box[0]), float(box[1])), (float(box[2]), float(box[3])))
            elements.append(LayoutElement(layout_type=str(cname), confidence=float(score), geometry=geometry))
        return elements

    @staticmethod
    def _word_centers(boxes: np.ndarray) -> np.ndarray:
        """Return the (x, y) center of each word box.

        Args:
            boxes: word boxes of shape (N, 4) (straight: x1, y1, x2, y2) or (N, 4, 2) (rotated polygon)

        Returns:
            array of shape (N, 2) with the relative center coordinates of each box
        """
        if boxes.ndim == 3:  # rotated polygons (N, 4, 2)
            return boxes.mean(axis=1)
        return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)

    @staticmethod
    def _point_in_poly(point: np.ndarray, poly: np.ndarray) -> bool:
        """Test whether a 2D point lies inside a polygon using the ray casting algorithm.

        Args:
            point: array of shape (2,) with the (x, y) coordinates of the point
            poly: array of shape (M, 2) with the polygon vertices

        Returns:
            True if the point is inside the polygon
        """
        x, y = float(point[0]), float(point[1])
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = float(poly[i][0]), float(poly[i][1])
            xj, yj = float(poly[j][0]), float(poly[j][1])
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _localize_logic(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        """Re-index a table's logical coordinates to a local 0-based grid.

        The table model returns logical (row/column) coordinates that may carry a constant offset; shifting them
        so that the smallest row/column start is 0 makes the grid directly usable by :meth:`Table.to_grid`.

        Args:
            cells: the cells of a single table

        Returns:
            a tuple `(cells, num_rows, num_cols)` with the re-indexed cells and the table dimensions
        """
        min_row = min(int(c["row_start"]) for c in cells)
        min_col = min(int(c["col_start"]) for c in cells)
        norm: list[dict[str, Any]] = []
        max_row = max_col = 0
        for c in cells:
            nc = dict(c)
            nc["row_start"] = int(c["row_start"]) - min_row
            nc["row_end"] = int(c["row_end"]) - min_row
            nc["col_start"] = int(c["col_start"]) - min_col
            nc["col_end"] = int(c["col_end"]) - min_col
            max_row, max_col = max(max_row, nc["row_end"]), max(max_col, nc["col_end"])
            norm.append(nc)
        return norm, max_row + 1, max_col + 1

    def _build_tables(
        self,
        boxes: np.ndarray,
        word_preds: list[tuple[str, float]],
        page_table: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> tuple[list[Table], np.ndarray]:
        """Assign detected words to table cells and build the page tables.

        A page may contain several tables; each one is provided as its own grid (the OCR pipeline detects table
        regions with the layout model, then runs the table model on every cropped region). Both a single grid and
        a list of grids are accepted. Each word whose center falls inside a cell polygon is assigned to (at most)
        one cell, across all tables, and flagged so it can be removed from the regular `blocks` output. Words
        are joined per cell in reading order (top to bottom, then left to right).

        Args:
            boxes: word boxes of the page, of shape (N, 4) or (N, 4, 2), in relative coordinates
            word_preds: list of (text, confidence) for each of the N words
            page_table: the table structure prediction(s) for the page. Either a single grid
                `{"cells": [{"geometry", "score", "row_start", "row_end", "col_start", "col_end"}], "num_rows",
                "num_cols"}` (cell geometries in page-relative coordinates), a list of such grids, or None

        Returns:
            a tuple with the list of `Table` objects of the page (one per provided table) and a boolean mask of
            shape (N,) that is True for words assigned to a table (to be removed from `blocks`)
        """
        num_words = boxes.shape[0]
        consumed = np.zeros(num_words, dtype=bool)
        if page_table is None:
            return [], consumed

        # Normalize the prediction(s) to a list of per-table grids with local 0-based logical coordinates
        raw_tables = [page_table] if isinstance(page_table, dict) else list(page_table)
        table_dicts: list[dict[str, Any]] = []
        for raw in raw_tables:
            if not raw or len(raw.get("cells", [])) == 0:
                continue
            cells, n_rows, n_cols = self._localize_logic(raw["cells"])
            table_dicts.append({"cells": cells, "num_rows": n_rows, "num_cols": n_cols})
        if len(table_dicts) == 0:
            return [], consumed

        centers = self._word_centers(boxes) if num_words > 0 else np.empty((0, 2))

        tables_out: list[Table] = []
        for table_dict in table_dicts:
            cells = table_dict["cells"]
            cell_polys = [np.asarray(cell["geometry"], dtype=np.float32) for cell in cells]

            # Assign each (still unassigned) word to at most one cell of this table
            cell_word_idcs: list[list[int]] = [[] for _ in cells]
            for w_idx in range(num_words):
                if consumed[w_idx]:
                    continue
                for c_idx, poly in enumerate(cell_polys):
                    if self._point_in_poly(centers[w_idx], poly):
                        cell_word_idcs[c_idx].append(w_idx)
                        consumed[w_idx] = True
                        break

            # Build the cells
            table_cells: list[TableCell] = []
            for cell, poly, w_idcs in zip(cells, cell_polys, cell_word_idcs):
                if len(w_idcs) > 0:
                    # Reading order inside the cell: top to bottom, then left to right
                    ordered = sorted(w_idcs, key=lambda i: (round(float(centers[i][1]), 3), float(centers[i][0])))
                    value = " ".join(word_preds[i][0] for i in ordered)
                    confidence = float(np.mean([word_preds[i][1] for i in ordered]))
                else:
                    value, confidence = "", float(cell["score"])
                geometry = tuple(tuple(float(c) for c in pt) for pt in poly.tolist())
                table_cells.append(
                    TableCell(
                        value=value,
                        confidence=confidence,
                        geometry=geometry,  # type: ignore[arg-type]
                        row_start=int(cell["row_start"]),
                        row_end=int(cell["row_end"]),
                        col_start=int(cell["col_start"]),
                        col_end=int(cell["col_end"]),
                    )
                )

            # Enclosing geometry of the whole table (relative bbox)
            all_pts = np.concatenate(cell_polys, axis=0)
            table_geometry = (
                (float(all_pts[:, 0].min()), float(all_pts[:, 1].min())),
                (float(all_pts[:, 0].max()), float(all_pts[:, 1].max())),
            )
            table_confidence = float(np.mean([cell["score"] for cell in cells]))

            tables_out.append(
                Table(
                    cells=table_cells,
                    num_rows=int(table_dict["num_rows"]),
                    num_cols=int(table_dict["num_cols"]),
                    geometry=table_geometry,
                    confidence=table_confidence,
                )
            )

        return tables_out, consumed

    def _build_blocks(
        self,
        boxes: np.ndarray,
        objectness_scores: np.ndarray,
        word_preds: list[tuple[str, float]],
        crop_orientations: list[dict[str, Any]],
    ) -> list[Block]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 4) or (N, 4, 2)
            objectness_scores: objectness scores of all detected words of the page, of shape N
            word_preds: list of all detected words of the page, of shape N
            crop_orientations: list of dictoinaries containing
                the general orientation (orientations + confidences) of the crops

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
            lines = self._resolve_lines(_boxes if _boxes.ndim == 3 else _boxes[:, :4])
            # Decide whether we try to form blocks
            if self.resolve_blocks and len(lines) > 1:
                _blocks = self._resolve_blocks(_boxes if _boxes.ndim == 3 else _boxes[:, :4], lines)
            else:
                _blocks = [lines]
        else:
            # Sort bounding boxes, one line for all boxes, one block for the line
            lines = [self._sort_boxes(_boxes if _boxes.ndim == 3 else _boxes[:, :4])[0]]  # type: ignore[list-item]
            _blocks = [lines]

        blocks = [
            Block([
                Line([
                    Word(
                        *word_preds[idx],
                        tuple(tuple(pt) for pt in boxes[idx].tolist()),  # type: ignore[arg-type]
                        float(objectness_scores[idx]),
                        crop_orientations[idx],
                    )
                    if boxes.ndim == 3
                    else Word(
                        *word_preds[idx],
                        ((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3])),
                        float(objectness_scores[idx]),
                        crop_orientations[idx],
                    )
                    for idx in line
                ])
                for line in lines
            ])
            for lines in _blocks
        ]

        return blocks

    def extra_repr(self) -> str:
        return (
            f"resolve_lines={self.resolve_lines}, resolve_blocks={self.resolve_blocks}, "
            f"paragraph_break={self.paragraph_break}, "
            f"export_as_straight_boxes={self.export_as_straight_boxes}"
        )

    def __call__(
        self,
        pages: list[np.ndarray],
        boxes: list[np.ndarray],
        objectness_scores: list[np.ndarray],
        text_preds: list[list[tuple[str, float]]],
        page_shapes: list[tuple[int, int]],
        crop_orientations: list[dict[str, Any]],
        orientations: list[dict[str, Any]] | None = None,
        languages: list[dict[str, Any]] | None = None,
        regions: list[dict[str, Any] | None] | None = None,
        tables: list[dict[str, Any] | None] | None = None,
    ) -> Document:
        """Re-arrange detected words into structured blocks

        Args:
            pages: list of N elements, where each element represents the page image
            boxes: list of N elements, where each element represents the localization predictions, of shape (*, 4)
                or (*, 4, 2) for all words for a given page
            objectness_scores: list of N elements, where each element represents the objectness scores
            text_preds: list of N elements, where each element is the list of all word prediction (text + confidence)
            page_shapes: shape of each page, of size N
            crop_orientations: list of N elements, where each element is
                a dictionary containing the general orientation (orientations + confidences) of the crops
            orientations: optional, list of N elements,
                where each element is a dictionary containing the orientation (orientation + confidence)
            languages: optional, list of N elements,
                where each element is a dictionary containing the language (language + confidence)
            regions: optional, list of N elements, where each element is a layout prediction
                `{"boxes": (R, 4|4x2), "class_names": [...], "scores": [...]}` attached to each page
            tables: optional, list of N elements, where each element is the table structure prediction(s) of a
                page: a single grid `{"cells": [...], "num_rows": int, "num_cols": int}` or a list of such grids
                (one per table region detected by the layout model). Words assigned to any table are removed from
                the `blocks` output of that page.

        Returns:
            document object
        """
        if len(boxes) != len(text_preds) != len(crop_orientations) != len(objectness_scores) or len(boxes) != len(
            page_shapes
        ) != len(crop_orientations) != len(objectness_scores):
            raise ValueError("All arguments are expected to be lists of the same size")

        _orientations = orientations if isinstance(orientations, list) else [None] * len(boxes)
        _languages = languages if isinstance(languages, list) else [None] * len(boxes)
        _regions = regions if isinstance(regions, list) else [None] * len(boxes)
        _tables = tables if isinstance(tables, list) else [None] * len(boxes)
        if self.export_as_straight_boxes and len(boxes) > 0:
            # If boxes are already straight OK, else fit a bounding rect
            if boxes[0].ndim == 3:
                # Iterate over pages and boxes
                boxes = [np.concatenate((p_boxes.min(1), p_boxes.max(1)), 1) for p_boxes in boxes]

        _pages = []
        for (
            page,
            _idx,
            shape,
            page_boxes,
            loc_scores,
            word_preds,
            word_crop_orientations,
            orientation,
            language,
            page_regions,
            page_table,
        ) in zip(  # noqa: E501
            pages,
            range(len(boxes)),
            page_shapes,
            boxes,
            objectness_scores,
            text_preds,
            crop_orientations,
            _orientations,
            _languages,
            _regions,
            _tables,
        ):
            # Build the page tables and flag the words that belong to a table
            page_tables, consumed = self._build_tables(page_boxes, word_preds, page_table)
            if consumed.any():
                # Remove the words assigned to a table from the regular blocks output
                keep = ~consumed
                page_boxes = page_boxes[keep]
                loc_scores = loc_scores[keep]
                word_preds = [wp for wp, k in zip(word_preds, keep) if k]
                word_crop_orientations = [co for co, k in zip(word_crop_orientations, keep) if k]

            _pages.append(
                Page(
                    page,
                    self._build_blocks(
                        page_boxes,
                        loc_scores,
                        word_preds,
                        word_crop_orientations,
                    ),
                    _idx,
                    shape,
                    orientation,
                    language,
                    self._build_layout_elements(page_regions),
                    page_tables,
                )
            )

        return Document(_pages)


class KIEDocumentBuilder(DocumentBuilder):
    """Implements a KIE document builder

    Args:
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs
        export_as_straight_boxes: if True, force straight boxes in the export (fit a rectangle
            box to all rotated boxes). Else, keep the boxes format unchanged, no matter what it is.
    """

    def __call__(  # type: ignore[override]
        self,
        pages: list[np.ndarray],
        boxes: list[dict[str, np.ndarray]],
        objectness_scores: list[dict[str, np.ndarray]],
        text_preds: list[dict[str, list[tuple[str, float]]]],
        page_shapes: list[tuple[int, int]],
        crop_orientations: list[dict[str, list[dict[str, Any]]]],
        orientations: list[dict[str, Any]] | None = None,
        languages: list[dict[str, Any]] | None = None,
        regions: list[dict[str, Any] | None] | None = None,
        tables: list[list[dict[str, Any] | None] | None] | None = None,
    ) -> KIEDocument:
        """Re-arrange detected words into structured predictions

        Args:
            pages: list of N elements, where each element represents the page image
            boxes: list of N dictionaries, where each element represents the localization predictions for a class,
                of shape (*, 5) or (*, 6) for all predictions
            objectness_scores: list of N dictionaries, where each element represents the objectness scores for a class
            text_preds: list of N dictionaries, where each element is the list of all word prediction
            page_shapes: shape of each page, of size N
            crop_orientations: list of N dictonaries, where each element is
                a list containing the general crop orientations (orientations + confidences) of the crops
            orientations: optional, list of N elements,
                where each element is a dictionary containing the orientation (orientation + confidence)
            languages: optional, list of N elements,
                where each element is a dictionary containing the language (language + confidence)
            regions: optional, list of N elements, where each element is a layout prediction
                `{"boxes": (R, 4|4x2), "class_names": [...], "scores": [...]}` attached to each page
            tables: optional, list of N elements, where each element is the table structure prediction(s) of a
                page: a single grid `{"cells": [...], "num_rows": int, "num_cols": int}` or a list of such grids
                (one per table region detected by the layout model). Words assigned to any table are removed from
                the `blocks` output of that page. Unused for KIE documents, as tables are not supported in KIE.

        Returns:
            document object
        """
        if len(boxes) != len(text_preds) != len(crop_orientations) != len(objectness_scores) or len(boxes) != len(
            page_shapes
        ) != len(crop_orientations) != len(objectness_scores):
            raise ValueError("All arguments are expected to be lists of the same size")
        _orientations = orientations if isinstance(orientations, list) else [None] * len(boxes)
        _languages = languages if isinstance(languages, list) else [None] * len(boxes)
        _regions = regions if isinstance(regions, list) else [None] * len(boxes)
        if self.export_as_straight_boxes and len(boxes) > 0:
            # If boxes are already straight OK, else fit a bounding rect
            if next(iter(boxes[0].values())).ndim == 3:
                straight_boxes: list[dict[str, np.ndarray]] = []
                # Iterate over pages
                for p_boxes in boxes:
                    # Iterate over boxes of the pages
                    straight_boxes_dict = {}
                    for k, box in p_boxes.items():
                        straight_boxes_dict[k] = np.concatenate((box.min(1), box.max(1)), 1)
                    straight_boxes.append(straight_boxes_dict)
                boxes = straight_boxes

        _pages = [
            KIEPage(
                page,
                {
                    k: self._build_blocks(
                        page_boxes[k],
                        loc_scores[k],
                        word_preds[k],
                        word_crop_orientations[k],
                    )
                    for k in page_boxes.keys()
                },
                _idx,
                shape,
                orientation,
                language,
                self._build_layout_elements(page_regions),
            )
            for page, _idx, shape, page_boxes, loc_scores, word_preds, word_crop_orientations, orientation, language, page_regions in zip(  # noqa: E501
                pages,
                range(len(boxes)),
                page_shapes,
                boxes,
                objectness_scores,
                text_preds,
                crop_orientations,
                _orientations,
                _languages,
                _regions,
            )
        ]

        return KIEDocument(_pages)

    def _build_blocks(  # type: ignore[override]
        self,
        boxes: np.ndarray,
        objectness_scores: np.ndarray,
        word_preds: list[tuple[str, float]],
        crop_orientations: list[dict[str, Any]],
    ) -> list[Prediction]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 4) or (N, 4, 2)
            objectness_scores: objectness scores of all detected words of the page
            word_preds: list of all detected words of the page, of shape N
            crop_orientations: list of orientations for each word crop

        Returns:
            list of block elements
        """
        if boxes.shape[0] != len(word_preds):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(word_preds)}")

        if boxes.shape[0] == 0:
            return []

        # Decide whether we try to form lines
        _boxes = boxes
        idxs, _ = self._sort_boxes(_boxes if _boxes.ndim == 3 else _boxes[:, :4])
        predictions = [
            Prediction(
                value=word_preds[idx][0],
                confidence=word_preds[idx][1],
                geometry=tuple(tuple(pt) for pt in boxes[idx].tolist()),  # type: ignore[arg-type]
                objectness_score=float(objectness_scores[idx]),
                crop_orientation=crop_orientations[idx],
            )
            if boxes.ndim == 3
            else Prediction(
                value=word_preds[idx][0],
                confidence=word_preds[idx][1],
                geometry=((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3])),
                objectness_score=float(objectness_scores[idx]),
                crop_orientation=crop_orientations[idx],
            )
            for idx in idxs
        ]
        return predictions
