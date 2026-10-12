# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import bisect
import statistics
from collections.abc import Sequence
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
from doctr.utils.geometry import (
    estimate_page_angle,
    order_points,
    resolve_enclosing_bbox,
    resolve_enclosing_rbbox,
    rotate_boxes,
)
from doctr.utils.repr import NestedObject

__all__ = ["DocumentBuilder"]

# Median word height of body text on an A4 page, relative to the page width: `paragraph_break` is scaled up on pages
# whose type is larger than that (e.g. a crop of a page)
_BODY_WORD_HEIGHT = 0.02
# Cost of the horizontal distance between a box and a row, relative to their vertical offset, when several rows are
# level with the box: 50 units of horizontal distance weigh as much as 1 unit of vertical offset
_ROW_DISTANCE_WEIGHT = 0.02


class DocumentBuilder(NestedObject):
    """Implements a document builder

    Args:
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs. It is scaled up on pages with
            large type (e.g. a crop of a page), and lines can be split at a smaller distance adapted to the word
            spacing of the page.
        export_as_straight_boxes: if True, force straight boxes in the export (fit a rectangle
            box to all rotated boxes). Else, keep the boxes format unchanged, no matter what it is.
        keep_reading_order: if True, arrange the content of every page in reading order (cf.
            :mod:`doctr.models.reading_order`). The reading direction is inferred from the recognized text.
            When a layout is available it is always preferred: the text blocks and the page furniture
            (headers, footers, ...) are taken from the layout regions.
    """

    def __init__(
        self,
        resolve_lines: bool = True,
        resolve_blocks: bool = False,
        paragraph_break: float = 0.035,
        export_as_straight_boxes: bool = False,
        keep_reading_order: bool = False,
    ) -> None:
        self.resolve_lines = resolve_lines
        self.resolve_blocks = resolve_blocks
        self.paragraph_break = paragraph_break
        self.export_as_straight_boxes = export_as_straight_boxes
        self.keep_reading_order = keep_reading_order

    @staticmethod
    def _sort_boxes(boxes: np.ndarray, shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Sort bounding boxes from top to bottom, left to right

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 4, 2)
            shape: the page dimensions (height, width)

        Returns:
            tuple: indices of ordered boxes of shape (N,), boxes
                If straight boxes are passed tpo the function, boxes are unchanged
                else: boxes returned are straight boxes fitted to the straightened rotated boxes
                so that we fit the lines afterwards to the straigthened page
        """
        if boxes.ndim == 3:
            height, width = shape if shape is not None else (1024, 1024)
            scale = np.array([width, height], dtype=boxes.dtype)
            angle = estimate_page_angle(boxes * scale)
            rotated = rotate_boxes(
                loc_preds=boxes,
                angle=-angle,
                orig_shape=(height, width),
                min_angle=1.0,
            )
            # On rotated pages, detectors can output a mix of properly rotated polygons and axis-aligned
            # enclosing boxes. A box that is not itself rotated carries no rotation to remove
            if abs(angle) >= 1.0:
                edges = (boxes[:, 1] - boxes[:, 0]) * scale
                own_angle = np.rad2deg(np.arctan2(-edges[:, 1], edges[:, 0]))
                keep = np.abs(own_angle) < abs(angle) / 2
                if keep.any():
                    centers = boxes.mean(axis=1, keepdims=True)
                    new_centers = rotated.mean(axis=1, keepdims=True)
                    rotated[keep] = boxes[keep] - centers[keep] + new_centers[keep]
            boxes = np.concatenate((rotated.min(1), rotated.max(1)), -1)
        med_height = float(np.median(boxes[:, 3] - boxes[:, 1]))
        if not np.isfinite(med_height) or med_height <= 0:
            med_height = 1.0
        return (boxes[:, 0] + 2 * boxes[:, 3] / med_height).argsort(), boxes

    def _resolve_sub_lines(
        self,
        boxes: np.ndarray,
        word_idcs: list[int],
        break_dist: float | None = None,
        word_regions: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[list[int]]:
        """Split a line in sub_lines

        Args:
            boxes: bounding boxes of shape (N, 4)
            word_idcs: list of indexes for the words of the line
            break_dist: horizontal distance above which the line is split (defaults to `self.paragraph_break`)
            word_regions: optional layout region (group) of each box (-1 for none) and coverage of each box by each
                region group (cf. `_word_regions`): the line is also split between two boxes of distinct regions (e.g.
                two columns separated by a gutter as narrow as a word gap), unless one also lies in the region of the
                other (nested or overlapping regions)

        Returns:
            A list of (sub-)lines computed from the original line (words)
        """
        break_dist = self.paragraph_break if break_dist is None else break_dist
        lines = []
        # Sort words horizontally
        word_idcs = [word_idcs[idx] for idx in boxes[word_idcs, 0].argsort().tolist()]

        # Eventually split line horizontally
        if len(word_idcs) < 2:
            lines.append(word_idcs)
        else:
            sub_line = [word_idcs[0]]
            # Right edge of the current sub-line: a box can end before an earlier, wider one (e.g. a punctuation box
            # nested in a word box), so the running max is tracked instead of the right edge of the last box
            right = boxes[word_idcs[0], 2]
            groups, coverage = word_regions if word_regions is not None else (None, None)
            region = -1 if groups is None else int(groups[word_idcs[0]])  # region of the sub-line
            prev = word_idcs[0]
            for i in word_idcs[1:]:
                horiz_break = True

                # Compute distance between the sub-line and the box
                dist = boxes[i, 0] - right
                # If distance between boxes is lower than the break distance, same sub-line
                if dist < break_dist:
                    horiz_break = False
                # Unless the box lies in another layout region than the sub-line, and neither the box lies in the
                # region of the sub-line nor the previous box in the region of the box
                box_region = -1 if groups is None else int(groups[i])
                if (
                    coverage is not None
                    and box_region >= 0
                    and region >= 0
                    and box_region != region
                    and coverage[i, region] < 0.5
                    and coverage[prev, box_region] < 0.5
                ):
                    horiz_break = True

                if horiz_break:
                    lines.append(sub_line)
                    sub_line = []
                    right = boxes[i, 2]
                    region = -1

                sub_line.append(i)
                right = max(right, boxes[i, 2])
                if region < 0:
                    region = box_region
                prev = i
            lines.append(sub_line)

        return lines

    @staticmethod
    def _word_regions(
        boxes: np.ndarray,
        region_geoms: list[Any],
        region_labels: list[str],
        shape: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Assign each box to the layout region covering it, overlapping regions of the same label being grouped

        A box covered by no region by half (e.g. a word sticking out of a region slightly too short) is assigned to
        the smallest region holding it horizontally and covering it at least partly vertically.

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 4, 2)
            region_geoms: the geometries of the layout regions
            region_labels: the labels of the layout regions
            shape: the page dimensions (height, width), used to de-skew rotated pages exactly

        Returns:
            the region group of each box (-1 for none) and the coverage of each box by each region group (indexed by
            group), None without layout regions
        """
        from doctr.models.reading_order.base import (
            _covering_regions,
            _region_groups,
            _to_boxes,
            deskew_reading_geometries,
        )

        if len(region_geoms) == 0 or boxes.shape[0] == 0:
            return None
        word_geoms: Any = boxes
        if boxes.ndim == 3:  # de-skew the words and the regions together
            word_geoms, region_geoms = deskew_reading_geometries(
                list(boxes), region_geoms, page_shape=shape, angle_geoms=list(boxes)
            )
        regions, words = _to_boxes(region_geoms), _to_boxes(word_geoms)
        groups = np.asarray(_region_groups(regions, region_labels))
        covering = np.asarray(_covering_regions(words, regions, 0.5, region_labels))
        # Coverage of each box by each region, on both axes and in area
        inter_w = np.minimum(words[:, None, 2], regions[None, :, 2]) - np.maximum(
            words[:, None, 0], regions[None, :, 0]
        )
        inter_h = np.minimum(words[:, None, 3], regions[None, :, 3]) - np.maximum(
            words[:, None, 1], regions[None, :, 1]
        )
        x_cov = np.clip(inter_w, 0, None) / np.clip(words[:, 2:3] - words[:, 0:1], 1e-9, None)
        y_cov = np.clip(inter_h, 0, None) / np.clip(words[:, 3:4] - words[:, 1:2], 1e-9, None)
        uncovered = np.flatnonzero(covering < 0)
        if uncovered.size > 0:
            areas = (regions[:, 2] - regions[:, 0]) * (regions[:, 3] - regions[:, 1])
            holding = (x_cov[uncovered] >= 0.9) & (y_cov[uncovered] >= 0.3)
            smallest = np.where(holding, areas[None, :], np.inf).argmin(axis=1)
            covering[uncovered] = np.where(holding.any(axis=1), smallest, -1)
        coverage = np.zeros((words.shape[0], regions.shape[0]))
        np.maximum.at(coverage.T, groups, (x_cov * y_cov).T)  # coverage by a group: the max over its regions
        return np.where(covering >= 0, groups[covering], -1), coverage

    @staticmethod
    def _resolve_rows(boxes: np.ndarray, idxs: np.ndarray, y_med: float) -> list[list[int]]:
        """Group straight boxes into visual rows

        A box can join a row if its vertical center is close enough (less than half a median box height) to the median
        center of its horizontally closest boxes in that row. Comparing a box with its neighbors rather than with the
        mean of the whole row keeps a slanted line in a single row, and prevents a word from being drawn into the row
        of another column whose baseline is slightly offset. When several rows are level with the box, the best
        aligned one wins, the horizontal distance to the row being added as a small penalty.

        Args:
            boxes: straight bounding boxes of shape (N, 4)
            idxs: the order in which the boxes are visited (cf. `_sort_boxes`: top to bottom, left to right)
            y_med: the median box height

        Returns:
            the rows (lists of box indices), from top to bottom
        """
        if not y_med > 0:  # degenerate heights: no box is close enough to another one
            return [[idx] for idx in idxs.tolist()]
        lefts, rights = boxes[:, 0].tolist(), boxes[:, 2].tolist()
        centers = ((boxes[:, 1] + boxes[:, 3]) / 2).tolist()
        rows: list[list[int]] = []
        row_lefts: list[list[float]] = []  # left edges of the boxes of each row, sorted
        row_boxes: list[list[int]] = []  # boxes of each row, sorted by left edge
        highest: list[float] = []  # highest box center of each row
        lowest: list[float] = []  # lowest box center of each row
        active: list[int] = []  # rows which can still receive a box
        for idx in idxs.tolist():
            center, left, right = centers[idx], lefts[idx], rights[idx]
            # Boxes come roughly top to bottom: a row whose lowest box is far above can no longer be joined
            active = [row for row in active if lowest[row] > center - 2 * y_med]
            best, best_key = -1, (float("inf"), float("inf"))
            for row in active:
                # The neighbors of the box can only be level with it if the row spans its center
                if highest[row] >= center + y_med / 2 or lowest[row] <= center - y_med / 2:
                    continue
                # The horizontally closest boxes of the row lie around the position of the box in that row. A few of
                # them rather than a single one, so that a small mark (accent, punctuation) can't steer the box.
                pos = bisect.bisect_left(row_lefts[row], left)
                nearest = sorted(
                    (max(lefts[other] - right, left - rights[other], 0.0), other)
                    for other in row_boxes[row][max(pos - 3, 0) : pos + 3]
                )[:3]
                offset = abs(statistics.median(centers[other] for _, other in nearest) - center)
                # On a slanted line, the next line can start before the end of the current one: the offset alone
                # would let a distant row of similar height take the box
                key = (offset + _ROW_DISTANCE_WEIGHT * nearest[0][0], nearest[0][0])
                if offset < y_med / 2 and key < best_key:
                    best, best_key = row, key
            if best < 0:
                best = len(rows)
                rows.append([])
                row_lefts.append([])
                row_boxes.append([])
                highest.append(center)
                lowest.append(center)
                active.append(best)
            rows[best].append(idx)
            pos = bisect.bisect_right(row_lefts[best], left)
            row_lefts[best].insert(pos, left)
            row_boxes[best].insert(pos, idx)
            highest[best] = min(highest[best], center)
            lowest[best] = max(lowest[best], center)

        return rows

    def _resolve_lines(
        self,
        boxes: np.ndarray,
        shape: tuple[int, int] | None = None,
        word_regions: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[list[int]]:
        """Order boxes to group them in lines

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 4, 2) in case of rotated bbox
            shape: the page dimensions (height, width), used to de-skew rotated pages exactly
            word_regions: optional layout region (group) of each box and coverage by each group (cf. `_word_regions`):
                a line never spans two distinct regions

        Returns:
            nested list of box indices
        """
        # Sort boxes, and straighten the boxes if they are rotated
        idxs, boxes = self._sort_boxes(boxes, shape)

        # Compute median for boxes heights
        y_med = np.median(boxes[:, 3] - boxes[:, 1])

        # Group the words into visual rows
        rows = self._resolve_rows(boxes, idxs, float(y_med))

        # Split the rows horizontally, using a break distance adapted to the page spacing.
        # The gaps are collected with one vectorized pass per row (the loop is over rows, not word pairs).
        gap_chunks = []
        n_pairs = 0
        for row in rows:
            if len(row) < 2:
                continue
            row_idcs = np.asarray(row)
            row_idcs = row_idcs[np.argsort(boxes[row_idcs, 0], kind="stable")]
            n_pairs += row_idcs.shape[0] - 1
            # Same gap definition as `_resolve_sub_lines`: distance to the right edge of everything on the left
            gap_chunks.append(boxes[row_idcs[1:], 0] - np.maximum.accumulate(boxes[row_idcs, 2])[:-1])
        all_gaps = np.concatenate(gap_chunks) if gap_chunks else np.empty(0, dtype=boxes.dtype)
        pos_gaps = all_gaps[all_gaps > 0]
        # Median word height, converted to width units
        aspect = (shape[0] / shape[1]) if shape is not None else 1.0
        floor = float(y_med) * aspect
        # The width-relative paragraph break grows with the type size (large type on a narrow page or crop)
        cap = self.paragraph_break * max(1.0, floor / _BODY_WORD_HEIGHT)
        if pos_gaps.shape[0] >= 5 and pos_gaps.shape[0] >= 0.5 * n_pairs:
            break_dist = min(cap, max(3.0 * float(np.median(pos_gaps)), floor))
        elif n_pairs >= 5:
            break_dist = min(cap, floor)
        else:
            break_dist = self.paragraph_break

        lines = []
        for row in rows:
            lines.extend(self._resolve_sub_lines(boxes, row, break_dist, word_regions))

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
    def _as_cell_polygon(geometry: Any) -> np.ndarray:
        """Normalize a cell geometry to a (4, 2) polygon.

        Straight-page table predictions store cells as flat (xmin, ymin, xmax, ymax) boxes, while rotated-page
        predictions store (4, 2) corner polygons: both are mapped to a (4, 2) polygon (TL, TR, BR, BL).

        Args:
            geometry: the raw cell geometry

        Returns:
            array of shape (4, 2) with the polygon vertices
        """
        arr = np.asarray(geometry, dtype=np.float32)
        if arr.ndim == 1:  # straight box (xmin, ymin, xmax, ymax)
            x0, y0, x1, y1 = arr
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        return order_points(arr.reshape(-1, 2))

    @staticmethod
    def _points_in_polygons(points: np.ndarray, polys: np.ndarray) -> np.ndarray:
        """Vectorized ray casting: test every point against every polygon at once.

        Args:
            points: array of shape (N, 2) with the (x, y) coordinates of the points
            polys: array of shape (C, V, 2) with the polygon vertices

        Returns:
            boolean array of shape (N, C), True where point n lies inside polygon c
        """
        if points.shape[0] == 0 or polys.shape[0] == 0:
            return np.zeros((points.shape[0], polys.shape[0]), dtype=bool)
        px = points[:, 0].astype(np.float64)[:, None, None]
        py = points[:, 1].astype(np.float64)[:, None, None]
        vi = polys.astype(np.float64)  # (C, V, 2)
        vj = np.roll(vi, 1, axis=1)  # previous vertex, with wrap-around
        xi, yi = vi[None, ..., 0], vi[None, ..., 1]
        xj, yj = vj[None, ..., 0], vj[None, ..., 1]
        crossing = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        # A point is inside when a ray crosses the polygon boundary an odd number of times
        return (crossing.sum(axis=-1) % 2).astype(bool)

    @staticmethod
    def _order_cell_words(w_idcs: list[int], centers: np.ndarray, heights: np.ndarray) -> list[int]:
        """Order the words of a table cell in reading order: rows top to bottom, words left to right.

        Args:
            w_idcs: indices of the cell's words
            centers: word centers of shape (N, 2), de-skewed for rotated pages
            heights: per-word heights of shape (N,) (rotation-invariant)

        Returns:
            the cell's word indices in reading order
        """
        idcs = sorted(w_idcs, key=lambda i: float(centers[i][1]))
        med_height = float(np.median([heights[i] for i in idcs]))
        if not np.isfinite(med_height) or med_height <= 0:
            med_height = 1.0
        rows: list[list[int]] = [[idcs[0]]]
        y_sum = float(centers[idcs[0]][1])
        for idx in idcs[1:]:
            if float(centers[idx][1]) - y_sum / len(rows[-1]) < med_height / 2:
                rows[-1].append(idx)
                y_sum += float(centers[idx][1])
            else:
                rows.append([idx])
                y_sum = float(centers[idx][1])
        return [idx for row in rows for idx in sorted(row, key=lambda i: float(centers[i][0]))]

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
        # Geometry format follows the page's word geometry: straight 2-point boxes when the word boxes are
        # (N, 4), 4-point polygons when they are (N, 4, 2).
        straight = boxes.ndim != 3
        # Rotation-invariant word heights (left edge length for polygons), used to cluster cell words into rows
        if num_words == 0:
            word_heights = np.empty(0)
        elif straight:
            word_heights = boxes[:, 3] - boxes[:, 1]
        else:
            word_heights = np.linalg.norm(boxes[:, 3] - boxes[:, 0], axis=1)

        tables_out: list[Table] = []
        for table_dict in table_dicts:
            cells = table_dict["cells"]
            if len(cells) == 0:
                continue
            cell_polys = [self._as_cell_polygon(cell["geometry"]) for cell in cells]
            polys_arr = np.stack(cell_polys)  # (C, 4, 2), vertices ordered TL, TR, BR, BL

            # Order along the table's own axes instead by de-skewing the centers with the table angle
            order_centers = centers
            if not straight and centers.shape[0] > 0:
                top_edges = polys_arr[:, 1] - polys_arr[:, 0]  # TR - TL
                angle = float(np.median(np.arctan2(top_edges[:, 1], top_edges[:, 0])))
                cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                pivot = centers.mean(axis=0)
                shifted = centers - pivot
                order_centers = np.stack(
                    [
                        pivot[0] + shifted[:, 0] * cos_a - shifted[:, 1] * sin_a,
                        pivot[1] + shifted[:, 0] * sin_a + shifted[:, 1] * cos_a,
                    ],
                    axis=1,
                )

            # Assign each (still unassigned) word to at most one cell of this table
            cell_word_idcs: list[list[int]] = [[] for _ in cells]
            free_idcs = np.flatnonzero(~consumed)
            if free_idcs.size > 0:
                # The first cell (in cell order) whose polygon contains the word center
                inside = self._points_in_polygons(centers[free_idcs], polys_arr)  # (F, C)
                assigned = inside.any(axis=1)
                first_cell = np.where(assigned, inside.argmax(axis=1), -1)
                for w_idx, c_idx in zip(free_idcs, first_cell):
                    if c_idx >= 0:
                        cell_word_idcs[c_idx].append(int(w_idx))
                        consumed[w_idx] = True

                # Words that landed just outside every cell but still inside the table region are attached
                # to the nearest cell, so table text is not dropped into the body.
                # The capture radius is bounded by the cell size to avoid pulling in body words.
                leftover = free_idcs[~assigned]
                if leftover.size > 0:
                    tx0, ty0 = polys_arr[..., 0].min(), polys_arr[..., 1].min()
                    tx1, ty1 = polys_arr[..., 0].max(), polys_arr[..., 1].max()
                    cell_centers = polys_arr.mean(axis=1)  # (C, 2)
                    cell_sizes = np.linalg.norm(polys_arr[:, 2] - polys_arr[:, 0], axis=1)  # TL->BR diagonal
                    max_dist = 0.5 * float(np.median(cell_sizes))
                    in_region = (
                        (centers[leftover, 0] >= tx0)
                        & (centers[leftover, 0] <= tx1)
                        & (centers[leftover, 1] >= ty0)
                        & (centers[leftover, 1] <= ty1)
                    )
                    for w_idx in leftover[in_region]:
                        dists = np.linalg.norm(cell_centers - centers[w_idx], axis=1)
                        nearest = int(dists.argmin())
                        if dists[nearest] <= max_dist:
                            cell_word_idcs[nearest].append(int(w_idx))
                            consumed[w_idx] = True

            # Build the cells
            table_cells: list[TableCell] = []
            for cell, poly, w_idcs in zip(cells, cell_polys, cell_word_idcs):
                if len(w_idcs) > 0:
                    # Reading order inside the cell: rows top to bottom, words left to right (table axes)
                    ordered = self._order_cell_words(w_idcs, order_centers, word_heights)
                    value = " ".join(word_preds[i][0] for i in ordered)
                    confidence = float(np.mean([word_preds[i][1] for i in ordered]))
                else:
                    value, confidence = "", float(cell["score"])
                if straight:
                    xs, ys = poly[:, 0], poly[:, 1]
                    geometry: Any = (
                        (float(xs.min()), float(ys.min())),
                        (float(xs.max()), float(ys.max())),
                    )
                else:
                    geometry = tuple(tuple(float(c) for c in pt) for pt in poly.tolist())
                table_cells.append(
                    TableCell(
                        value=value,
                        confidence=confidence,
                        geometry=geometry,
                        row_start=int(cell["row_start"]),
                        row_end=int(cell["row_end"]),
                        col_start=int(cell["col_start"]),
                        col_end=int(cell["col_end"]),
                        objectness_score=float(cell["score"]),
                    )
                )

            # Enclosing geometry of the whole table
            all_pts = np.concatenate(cell_polys, axis=0)
            xmin, ymin = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
            xmax, ymax = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
            table_geometry: Any = (
                ((xmin, ymin), (xmax, ymax)) if straight else ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
            )
            table_confidence = float(np.mean([float(cell["score"]) for cell in cells]))

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
        shape: tuple[int, int] | None = None,
        layout: list[LayoutElement] | None = None,
    ) -> list[Block]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 4) or (N, 4, 2)
            objectness_scores: objectness scores of all detected words of the page, of shape N
            word_preds: list of all detected words of the page, of shape N
            crop_orientations: list of dictoinaries containing
                the general orientation (orientations + confidences) of the crops
            shape: the page dimensions (height, width), used to de-skew rotated pages exactly when
                resolving lines
            layout: optional layout regions of the page: a line never spans two of them

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
            word_regions = (
                self._word_regions(_boxes, [elt.geometry for elt in layout], [elt.type for elt in layout], shape)
                if layout
                else None
            )
            lines = self._resolve_lines(_boxes if _boxes.ndim == 3 else _boxes[:, :4], shape, word_regions)
            # Decide whether we try to form blocks
            if self.resolve_blocks and len(lines) > 1:
                _blocks = self._resolve_blocks(_boxes if _boxes.ndim == 3 else _boxes[:, :4], lines)
            else:
                _blocks = [lines]
        else:
            # Sort bounding boxes, one line for all boxes, one block for the line
            lines = [self._sort_boxes(_boxes if _boxes.ndim == 3 else _boxes[:, :4], shape)[0]]  # type: ignore[list-item]
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

    @staticmethod
    def _words_to_boxes(words: list[Word]) -> np.ndarray:
        """Rebuild a localization array from word geometries (rotated polygons or straight boxes)."""
        geoms = [np.asarray(word.geometry, dtype=np.float32) for word in words]
        if len(geoms) > 0 and geoms[0].shape == (4, 2):
            return np.stack(geoms)  # (N, 4, 2) rotated polygons
        return np.asarray(
            [[g[0][0], g[0][1], g[1][0], g[1][1]] for g in geoms], dtype=np.float32
        )  # (N, 4) straight boxes

    def _apply_reading_order(self, page: Page) -> None:
        """Arrange the blocks of a page in reading order, in place.

        Args:
            page: the page whose blocks should be reordered in place
        """
        from doctr.io.exporters import _reading_order_signature, _store_reading_order, page_reading_order

        if not page.blocks:
            return

        collapse = not self.resolve_lines
        if collapse:
            words = [word for block in page.blocks for line in block.lines for word in line.words]
            if len(words) < 2:
                return
            word_boxes = self._words_to_boxes(words)
            word_regions = self._word_regions(
                word_boxes, [elt.geometry for elt in page.layout], [elt.type for elt in page.layout], page.dimensions
            )
            groups = self._resolve_lines(word_boxes, page.dimensions, word_regions)
            page.blocks = [Block([Line([words[idx] for idx in group]) for group in groups])]

        # With the figures: the result is stored back in the cache read by the exports
        items, labels, direction = page_reading_order(page, include_figures=True)
        blocks = [item for item in items if isinstance(item, Block)]
        page.blocks = (
            [Block(lines=[Line([word for block in blocks for line in block.lines for word in line.words])])]
            if collapse
            else blocks
        )
        if not collapse:
            # The page now *is* the reading order, so an export would recompute the very same linearization.
            # Re-key the cached result to the new block list so the first export reuses it.
            # Skipped when collapsing: the single flattened line no longer matches the line-level items.
            _store_reading_order(page, _reading_order_signature(page, "auto"), (items, labels, direction))

    def extra_repr(self) -> str:
        return (
            f"resolve_lines={self.resolve_lines}, resolve_blocks={self.resolve_blocks}, "
            f"paragraph_break={self.paragraph_break}, "
            f"export_as_straight_boxes={self.export_as_straight_boxes}, "
            f"keep_reading_order={self.keep_reading_order}"
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
        tables: Sequence[dict[str, Any] | list[dict[str, Any]] | None] | None = None,
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
        expected_len = len(boxes)
        if any(
            len(arg) != expected_len for arg in (pages, text_preds, crop_orientations, objectness_scores, page_shapes)
        ):
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

            page_layout = self._build_layout_elements(page_regions)
            page_blocks = self._build_blocks(
                page_boxes,
                loc_scores,
                word_preds,
                word_crop_orientations,
                shape,
                page_layout,
            )
            _page = Page(
                page,
                page_blocks,
                _idx,
                shape,
                orientation,
                language,
                page_layout,
                page_tables,
            )
            if self.keep_reading_order:
                self._apply_reading_order(_page)

            _pages.append(_page)

        return Document(_pages)


class KIEDocumentBuilder(DocumentBuilder):
    """Implements a KIE document builder

    Args:
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs. It is scaled up on pages with
            large type (e.g. a crop of a page), and lines can be split at a smaller distance adapted to the word
            spacing of the page.
        export_as_straight_boxes: if True, force straight boxes in the export (fit a rectangle
            box to all rotated boxes). Else, keep the boxes format unchanged, no matter what it is.
        keep_reading_order: if True, arrange the content of every page in reading order (cf.
            :mod:`doctr.models.reading_order`). The reading direction is inferred from the recognized text.
            When a layout is available it is always preferred: the text blocks and the page furniture
            (headers, footers, ...) are taken from the layout regions.
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
        tables: Sequence[dict[str, Any] | list[dict[str, Any]] | None] | None = None,
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
        expected_len = len(boxes)
        if any(
            len(arg) != expected_len for arg in (pages, text_preds, crop_orientations, objectness_scores, page_shapes)
        ):
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
                        shape,
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
        shape: tuple[int, int] | None = None,
    ) -> list[Prediction]:
        """Gather independent words in structured blocks

        Args:
            boxes: bounding boxes of all detected words of the page, of shape (N, 4) or (N, 4, 2)
            objectness_scores: objectness scores of all detected words of the page
            word_preds: list of all detected words of the page, of shape N
            crop_orientations: list of orientations for each word crop
            shape: the page dimensions (height, width), used to de-skew rotated pages exactly

        Returns:
            list of block elements
        """
        if boxes.shape[0] != len(word_preds):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(word_preds)}")

        if boxes.shape[0] == 0:
            return []

        # Decide whether we try to form lines
        _boxes = boxes
        idxs, _ = self._sort_boxes(_boxes if _boxes.ndim == 3 else _boxes[:, :4], shape)
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
