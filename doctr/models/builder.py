# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import bisect
import statistics
from collections.abc import Sequence
from typing import Any

import cv2
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

# Median word height of body text relative to the page width, above which `paragraph_break` is scaled up
_BODY_WORD_HEIGHT = 0.02
# Weight of the horizontal distance to a row, relative to the vertical offset, when several rows are level with a box
_ROW_DISTANCE_WEIGHT = 0.02


class DocumentBuilder(NestedObject):
    """Implements a document builder

    Args:
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs (scaled up for large type)
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
            word_regions: optional region group of each box and coverage by each group (cf. `_word_regions`): the
                line is also split between two distinct regions, unless one box lies in the region of the other

        Returns:
            A list of (sub-)lines computed from the original line (words)
        """
        break_dist = self.paragraph_break if break_dist is None else break_dist
        groups, coverage = word_regions if word_regions is not None else (None, None)
        lines: list[list[int]] = []
        right, region, prev = 0.0, -1, -1  # right edge, region and last word of the current sub-line
        for idx in [word_idcs[i] for i in boxes[word_idcs, 0].argsort().tolist()]:
            box_region = -1 if groups is None else int(groups[idx])
            # A new sub-line after a gap of at least `break_dist` from the sub-line (a box can be nested in an earlier
            # one: its max right edge is used), or between two distinct regions (e.g. columns with a narrow gutter)
            if (
                not lines
                or not boxes[idx, 0] - right < break_dist
                or (
                    coverage is not None
                    and box_region >= 0
                    and region >= 0
                    and box_region != region
                    and coverage[idx, region] < 0.5
                    and coverage[prev, box_region] < 0.5
                )
            ):
                lines.append([])
                right, region = boxes[idx, 2], -1
            lines[-1].append(idx)
            right = max(right, boxes[idx, 2])
            region = box_region if region < 0 else region
            prev = idx
        return lines

    @staticmethod
    def _word_regions(
        boxes: np.ndarray,
        region_geoms: list[Any],
        region_labels: list[str],
        shape: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Assign each box to the group of the layout region covering it (or holding it horizontally)

        Args:
            boxes: bounding boxes of shape (N, 4) or (N, 4, 2)
            region_geoms: the geometries of the layout regions
            region_labels: the labels of the layout regions
            shape: the page dimensions (height, width), used to de-skew rotated pages exactly

        Returns:
            the region group of each box (-1 for none) and the coverage of each box by each group, None without regions
        """
        from doctr.models.reading_order.base import (
            _box_areas,
            _covering_regions,
            _pairwise_overlaps,
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
        inter_w, inter_h = _pairwise_overlaps(words, regions)
        x_cov = inter_w / np.clip(words[:, 2:3] - words[:, 0:1], 1e-9, None)
        y_cov = inter_h / np.clip(words[:, 3:4] - words[:, 1:2], 1e-9, None)
        uncovered = np.flatnonzero(covering < 0)
        if (
            uncovered.size > 0
        ):  # e.g. a word sticking out of a region slightly too short: the smallest region holding it
            holding = (x_cov[uncovered] >= 0.9) & (y_cov[uncovered] >= 0.3)
            smallest = np.where(holding, _box_areas(regions)[None, :], np.inf).argmin(axis=1)
            covering[uncovered] = np.where(holding.any(axis=1), smallest, -1)
        coverage = np.zeros((words.shape[0], regions.shape[0]))
        np.maximum.at(coverage.T, groups, (x_cov * y_cov).T)
        return np.where(covering >= 0, groups[covering], -1), coverage

    @staticmethod
    def _resolve_rows(boxes: np.ndarray, idxs: np.ndarray, y_med: float) -> list[list[int]]:
        """Group straight boxes into visual rows

        A box joins the row whose horizontally closest boxes are the best aligned with it (less than half a median
        height apart), which follows slanted lines and keeps the rows of offset columns apart. A box right under or over
        one of them, away from the others (e.g. stacked figure labels), does not join the row.

        Args:
            boxes: straight bounding boxes of shape (N, 4)
            idxs: the order in which the boxes are visited (cf. `_sort_boxes`)
            y_med: the median box height

        Returns:
            the rows (lists of box indices)
        """
        if not y_med > 0:
            return [[idx] for idx in idxs.tolist()]
        lefts, rights = boxes[:, 0].tolist(), boxes[:, 2].tolist()
        centers = ((boxes[:, 1] + boxes[:, 3]) / 2).tolist()
        heights = (boxes[:, 3] - boxes[:, 1]).tolist()
        row_lefts: list[list[float]] = []  # sorted left edges of the boxes of each row
        row_boxes: list[list[int]] = []  # boxes of each row, sorted by left edge
        spans: list[list[float]] = []  # highest and lowest box center of each row

        def _stacked(idx: int, other: int, median: float) -> bool:
            """Whether two full-height boxes lie one over the other, `other` being level with its row"""
            overlap = min(rights[idx], rights[other]) - max(lefts[idx], lefts[other])
            return (
                overlap >= 0.5 * min(rights[idx] - lefts[idx], rights[other] - lefts[other])
                and abs(centers[idx] - centers[other]) >= y_med / 2
                and abs(centers[other] - median) < y_med / 2
                and min(heights[idx], heights[other]) >= 0.75 * y_med
            )

        def _best_row(idx: int, rows: list[int]) -> int:
            center, left, right = centers[idx], lefts[idx], rights[idx]
            best, best_key = -1, (float("inf"), float("inf"))
            for row in rows:
                if spans[row][0] >= center + y_med / 2 or spans[row][1] <= center - y_med / 2 or not row_boxes[row]:
                    continue
                # A few neighbors rather than one, so that a small mark (accent, punctuation) can't steer the box
                pos = bisect.bisect_left(row_lefts[row], left)
                nearest = sorted(
                    (max(lefts[other] - right, left - rights[other], 0.0), other)
                    for other in row_boxes[row][max(pos - 3, 0) : pos + 3]
                )[:3]
                median = statistics.median(centers[other] for _, other in nearest)
                stacked = [_stacked(idx, other, median) for _, other in nearest]
                if any(stacked) and all(dist >= y_med for (dist, _), flag in zip(nearest, stacked) if not flag):
                    continue
                offset = abs(median - center)
                # The distance breaks ties on slanted lines, where the next line can start before this one ends
                key = (offset + _ROW_DISTANCE_WEIGHT * nearest[0][0], nearest[0][0])
                if offset < y_med / 2 and key < best_key:
                    best, best_key = row, key
            return best

        def _add(row: int, idx: int) -> None:
            if row == len(row_boxes):
                row_lefts.append([])
                row_boxes.append([])
                spans.append([centers[idx], centers[idx]])
            pos = bisect.bisect_right(row_lefts[row], lefts[idx])
            row_lefts[row].insert(pos, lefts[idx])
            row_boxes[row].insert(pos, idx)
            spans[row] = [min(spans[row][0], centers[idx]), max(spans[row][1], centers[idx])]

        row_of: dict[int, int] = {}
        active: list[int] = []  # rows which can still receive a box: the boxes come roughly top to bottom
        for idx in idxs.tolist():
            active = [row for row in active if spans[row][1] > centers[idx] - 2 * y_med]
            row = _best_row(idx, active)
            if row < 0:
                row = len(row_boxes)
                active.append(row)
            _add(row, idx)
            row_of[idx] = row
        # Assign each box again once all rows exist: a box visited before the rest of its line (e.g. a short word
        # ending higher) may have joined a level row of another column
        highest, lowest = np.asarray(spans).T
        for idx in idxs.tolist():
            own = row_of[idx]
            pos = row_boxes[own].index(idx)
            del row_boxes[own][pos], row_lefts[own][pos]
            candidates = np.flatnonzero((highest < centers[idx] + y_med / 2) & (lowest > centers[idx] - y_med / 2))
            row = _best_row(idx, candidates.tolist())
            row_of[idx] = own if row < 0 else row
            _add(row_of[idx], idx)

        return [row for row in row_boxes if row]

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
            word_regions: optional region group of each box and coverage by each group (cf. `_word_regions`)

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
            gap_chunks.append(boxes[row_idcs[1:], 0] - np.maximum.accumulate(boxes[row_idcs, 2])[:-1])
        all_gaps = np.concatenate(gap_chunks) if gap_chunks else np.empty(0, dtype=boxes.dtype)
        pos_gaps = all_gaps[all_gaps > 0]
        # Median word height, converted to width units
        aspect = (shape[0] / shape[1]) if shape is not None else 1.0
        floor = float(y_med) * aspect
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
        if arr.size == 4:  # straight box (xmin, ymin, xmax, ymax) or ((xmin, ymin), (xmax, ymax))
            x0, y0, x1, y1 = arr.ravel()
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

    @classmethod
    def _table_grids(
        cls, page_table: dict[str, Any] | list[dict[str, Any]] | None
    ) -> list[tuple[list[dict[str, Any]], int, int, np.ndarray | None]]:
        """Normalize the table structure prediction(s) of a page to non-empty grids with local logical coordinates

        Args:
            page_table: a single grid, a list of grids or None (cf. `_build_tables`)

        Returns:
            the cells, number of rows, number of columns and (4, 2) region polygon (None if unknown) of each
            non-empty grid
        """
        raw_tables = [] if page_table is None else [page_table] if isinstance(page_table, dict) else list(page_table)
        grids = []
        for raw in raw_tables:
            if not raw or len(raw.get("cells", [])) == 0:
                continue
            region = raw.get("geometry")
            grids.append((
                *cls._localize_logic(raw["cells"]),
                None if region is None else cls._as_cell_polygon(region),
            ))
        return grids

    @classmethod
    def _assign_cell_words(cls, centers: np.ndarray, polys: np.ndarray, consumed: np.ndarray) -> list[list[int]]:
        """Assign each word not yet consumed whose center lies in a cell to the first such cell, and flag it as
        consumed

        Args:
            centers: the (N, 2) word centers
            polys: the (C, 4, 2) cell polygons
            consumed: the (N,) mask of the words already assigned to a table, updated in place

        Returns:
            the word indices of each cell
        """
        cell_words: list[list[int]] = [[] for _ in polys]
        free = np.flatnonzero(~consumed)
        if free.size == 0:
            return cell_words
        inside = cls._points_in_polygons(centers[free], polys)  # (F, C)
        assigned = inside.any(axis=1)
        for w_idx, c_idx in zip(free[assigned], inside[assigned].argmax(axis=1)):
            cell_words[c_idx].append(int(w_idx))
            consumed[w_idx] = True
        return cell_words

    @staticmethod
    def _rotate_points(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
        """Rotate (..., 2) points by `angle` radians around `pivot`"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        shifted = points - pivot
        return np.stack(
            [
                pivot[0] + shifted[..., 0] * cos_a - shifted[..., 1] * sin_a,
                pivot[1] + shifted[..., 0] * sin_a + shifted[..., 1] * cos_a,
            ],
            axis=-1,
        )

    @staticmethod
    def _grid_lines(start: np.ndarray, end: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: int) -> np.ndarray:
        """Positions of the `size + 1` grid lines of an axis, from the cell edges lying on each of them

        Line `k` is where index `k` starts and index `k - 1` ends: it lies at the median of the corresponding cell
        edges, the lines on which no cell edge lies are interpolated between their neighbours.

        Args:
            start: the first index of each cell along the axis, shape (C,)
            end: the last index of each cell along the axis, shape (C,)
            lo: the lower edge of each cell along the axis (in the table frame), shape (C,)
            hi: the upper edge of each cell along the axis (in the table frame), shape (C,)
            size: the number of indices (rows or columns) of the axis

        Returns:
            the non-decreasing positions of the grid lines, shape (size + 1,)
        """
        edges: list[list[float]] = [[] for _ in range(size + 1)]
        for s, e, low, high in zip(start, end, lo, hi):
            edges[int(s)].append(float(low))
            edges[int(e) + 1].append(float(high))
        known = [k for k, values in enumerate(edges) if values]
        lines = np.interp(np.arange(size + 1), known, [float(np.median(edges[k])) for k in known])
        return np.maximum.accumulate(lines)

    @staticmethod
    def _axis_slots(
        center: np.ndarray, lo: np.ndarray, hi: np.ndarray, lines: np.ndarray, line_tol: float | None = None
    ) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
        """Index of each word along an axis of a table grid, adding new indices for the words beyond the grid

        A word whose center lies between two grid lines takes their index. The words before the first line (resp.
        after the last one) are grouped into new indices before (resp. after) the grid: by text line (centers closer
        than `line_tol`) for rows, by overlapping extents for columns (`line_tol` is None).

        Args:
            center: the word centers along the axis (in the table frame), shape (W,)
            lo: the lower word edges along the axis, shape (W,)
            hi: the upper word edges along the axis, shape (W,)
            lines: the positions of the grid lines of the axis, shape (size + 1,)
            line_tol: the maximum distance between the centers of the words of a same new row

        Returns:
            the (possibly negative or beyond the grid) index of each word, and the (lo, hi) extent of every index
        """
        size = lines.size - 1
        idx = np.clip(np.searchsorted(lines, center, side="right") - 1, 0, size - 1)
        bands = {k: (float(lines[k]), float(lines[k + 1])) for k in range(size)}
        for before, mask in ((True, center < lines[0]), (False, center > lines[-1])):
            members = np.flatnonzero(mask)
            if members.size == 0:
                continue
            groups: list[list[int]] = []
            if line_tol is not None:
                for j in members[np.argsort(center[members], kind="stable")]:
                    if groups and center[j] - np.mean(center[groups[-1]]) < line_tol:
                        groups[-1].append(int(j))
                    else:
                        groups.append([int(j)])
            else:
                reach = -np.inf
                for j in members[np.argsort(lo[members], kind="stable")]:
                    if groups and lo[j] <= reach:
                        groups[-1].append(int(j))
                    else:
                        groups.append([int(j)])
                        reach = -np.inf
                    reach = max(reach, float(hi[j]))
            new_ids = range(-len(groups), 0) if before else range(size, size + len(groups))
            for g_id, group in zip(new_ids, groups):
                idx[group] = g_id
                bands[g_id] = (float(lo[group].min()), float(hi[group].max()))
        return idx, bands

    @classmethod
    def _complete_grid(
        cls,
        cells: list[dict[str, Any]],
        polys: np.ndarray,
        cell_words: list[list[int]],
        leftover: np.ndarray,
        word_polys: np.ndarray,
        angle: float,
    ) -> tuple[list[dict[str, Any]], np.ndarray, list[list[int]]]:
        """Place the words of a table that lie in no cell into its grid, so that no table text is left out

        The row and column lines of the grid are located from the cell edges. A word between two lines of each axis
        joins the cell covering that grid slot, or a new cell when no cell covers it (e.g. a missed cell). The words
        beyond the grid (e.g. a missed header row or a row of totals) make new rows / columns: in a new row, the
        words closer than half a text line, like the words of a sentence, share a single (possibly spanning) cell.

        Args:
            cells: the cells of the table, with local logical coordinates
            polys: the (C, 4, 2) cell polygons (absolute coordinates)
            cell_words: the word indices of each cell
            leftover: the indices of the words of the table that lie in no cell
            word_polys: the (N, 4, 2) word polygons (absolute coordinates)
            angle: the angle of the table rows (radians)

        Returns:
            the cells, their polygons and the word indices of each cell, new cells included (the logical coordinates
            of the new rows / columns before the grid are negative)
        """
        pivot = polys.reshape(-1, 2).mean(axis=0)
        # Table frame: the rows and columns are horizontal and vertical
        cell_pts = cls._rotate_points(polys, -angle, pivot)
        word_pts = cls._rotate_points(word_polys[leftover], -angle, pivot)
        w_x0, w_x1 = word_pts[..., 0].min(axis=1), word_pts[..., 0].max(axis=1)
        w_y0, w_y1 = word_pts[..., 1].min(axis=1), word_pts[..., 1].max(axis=1)
        logic = np.array([[c["row_start"], c["row_end"], c["col_start"], c["col_end"]] for c in cells], dtype=int)
        num_rows, num_cols = int(logic[:, 1].max()) + 1, int(logic[:, 3].max()) + 1
        row_lines = cls._grid_lines(
            logic[:, 0], logic[:, 1], cell_pts[..., 1].min(axis=1), cell_pts[..., 1].max(axis=1), num_rows
        )
        col_lines = cls._grid_lines(
            logic[:, 2], logic[:, 3], cell_pts[..., 0].min(axis=1), cell_pts[..., 0].max(axis=1), num_cols
        )
        med_h = float(np.median(w_y1 - w_y0))
        med_h = med_h if np.isfinite(med_h) and med_h > 0 else 1.0
        col_idx, col_bands = cls._axis_slots((w_x0 + w_x1) / 2, w_x0, w_x1, col_lines)
        row_idx, row_bands = cls._axis_slots((w_y0 + w_y1) / 2, w_y0, w_y1, row_lines, line_tol=med_h / 2)

        # The cell covering each slot of the grid
        occupied: dict[tuple[int, int], int] = {}
        for c_idx, (rs, re, cs, ce) in enumerate(logic):
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    occupied.setdefault((r, c), c_idx)

        # New cells: (row_start, row_end, col_start, col_end) -> positions in `leftover`
        new_cells: dict[tuple[int, int, int, int], list[int]] = {}
        in_grid = (row_idx >= 0) & (row_idx < num_rows)
        for j in np.flatnonzero(in_grid):
            slot = (int(row_idx[j]), int(col_idx[j]))
            if slot in occupied:
                cell_words[occupied[slot]].append(int(leftover[j]))
            else:
                new_cells.setdefault((slot[0], slot[0], slot[1], slot[1]), []).append(int(j))
        # New rows: the words of a same column, or closer than half a text line, share a cell
        for r in np.unique(row_idx[~in_grid]):
            members = np.flatnonzero(row_idx == r)
            members = members[np.argsort(w_x0[members], kind="stable")]
            group = [int(members[0])]
            for prev, j in zip(members[:-1], members[1:]):
                if col_idx[j] == col_idx[group[-1]] or w_x0[j] - w_x1[prev] < med_h / 2:
                    group.append(int(j))
                    continue
                new_cells[(int(r), int(r), int(col_idx[group].min()), int(col_idx[group].max()))] = group
                group = [int(j)]
            new_cells[(int(r), int(r), int(col_idx[group].min()), int(col_idx[group].max()))] = group

        cells, new_polys = list(cells), []
        for (rs, re, cs, ce), positions in new_cells.items():
            x0 = min(col_bands[cs][0], float(w_x0[positions].min()))
            x1 = max(col_bands[ce][1], float(w_x1[positions].max()))
            y0 = min(row_bands[rs][0], float(w_y0[positions].min()))
            y1 = max(row_bands[re][1], float(w_y1[positions].max()))
            rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
            new_polys.append(cls._rotate_points(rect, angle, pivot))
            # A cell added from the words: the table structure model did not score it
            cells.append({"score": None, "row_start": rs, "row_end": re, "col_start": cs, "col_end": ce})
            cell_words.append([int(leftover[j]) for j in positions])
        if new_polys:
            polys = np.concatenate([polys, np.stack(new_polys).astype(polys.dtype)])
        return cells, polys, cell_words

    @classmethod
    def _simplify_grid(
        cls,
        cells: list[dict[str, Any]],
        polys: np.ndarray,
        cell_words: list[list[int]],
        word_polys: np.ndarray,
        angle: float,
    ) -> tuple[list[dict[str, Any]], np.ndarray, list[list[int]]]:
        """Merge the rows / columns of a table split by the structure model, and remove the ones without any text

        Two adjacent rows (resp. columns) are merged when their text is complementary (no column, resp. row, has text
        in both) and lies on the same text line (resp. column), i.e. their text extents mostly overlap: the table
        structure model sometimes splits a row whose cell edges do not line up. A row (resp. column) without text of its
        own (no cell holds text in this row only) is removed: the grid only keeps the rows and columns of the table's
        content, and the cells spanning the removed ones span one less.

        Args:
            cells: the cells of the table, with local logical coordinates
            polys: the (C, 4, 2) cell polygons (absolute coordinates)
            cell_words: the word indices of each cell
            word_polys: the (N, 4, 2) word polygons (absolute coordinates)
            angle: the angle of the table rows (radians)

        Returns:
            the remaining cells, their polygons and the word indices of each cell
        """
        filled = np.array([len(w) > 0 for w in cell_words])
        if not filled.any():
            return cells, polys, cell_words
        pivot = polys.reshape(-1, 2).mean(axis=0)
        frame = cls._rotate_points(polys, -angle, pivot)
        rects = np.concatenate([frame.min(axis=1), frame.max(axis=1)], axis=1)  # (C, 4) x0, y0, x1, y1
        resized = np.zeros(len(cells), dtype=bool)
        # Text extent of each cell in the table frame: x0, y0, x1, y1
        content = np.full((len(cells), 4), np.nan)
        for c_idx in np.flatnonzero(filled):
            pts = cls._rotate_points(word_polys[cell_words[c_idx]], -angle, pivot).reshape(-1, 2)
            content[c_idx] = [*pts.min(axis=0), *pts.max(axis=0)]
        # row_start, row_end, col_start, col_end
        logic = np.array([[c["row_start"], c["row_end"], c["col_start"], c["col_end"]] for c in cells], dtype=int)
        alive = np.ones(len(cells), dtype=bool)

        def mergeable(axis: int, k: int) -> bool:
            start, end = logic[:, 2 * axis], logic[:, 2 * axis + 1]
            o_start, o_end = logic[:, 2 - 2 * axis], logic[:, 3 - 2 * axis]
            in_k, in_next = alive & (start <= k) & (end >= k), alive & (start <= k + 1) & (end >= k + 1)
            only_k, only_next = np.flatnonzero(in_k & ~in_next & filled), np.flatnonzero(in_next & ~in_k & filled)
            if only_k.size == 0 or only_next.size == 0:
                return False
            # Complementary: no index of the other axis holds text in both
            if np.any(
                (o_start[only_k, None] <= o_end[None, only_next]) & (o_start[None, only_next] <= o_end[only_k, None])
            ):
                return False
            # On the same line: the text extents along the axis mostly overlap
            lo, hi = (1, 3) if axis == 0 else (0, 2)
            a_lo, a_hi = content[only_k, lo].min(), content[only_k, hi].max()
            b_lo, b_hi = content[only_next, lo].min(), content[only_next, hi].max()
            overlap = min(a_hi, b_hi) - max(a_lo, b_lo)
            return bool(overlap >= 0.5 * min(a_hi - a_lo, b_hi - b_lo))

        def shift(axis: int, k: int, merge: bool) -> None:
            # Merge the index k + 1 into the index k, or remove the index k (its cells are removed beforehand)
            start, end = logic[:, 2 * axis], logic[:, 2 * axis + 1]
            end[alive & ((end > k) if merge else (end >= k))] -= 1
            start[alive & (start > k)] -= 1

        def resolve_overlaps() -> None:
            # The cells merged onto a same slot: the empty ones are absorbed by the cells holding text
            idx = np.flatnonzero(alive)
            for pos, i in enumerate(idx):
                if not alive[i]:
                    continue
                others = idx[pos + 1 :]
                others = others[alive[others]]
                hits = others[
                    (logic[others, 0] <= logic[i, 1])
                    & (logic[i, 0] <= logic[others, 1])
                    & (logic[others, 2] <= logic[i, 3])
                    & (logic[i, 2] <= logic[others, 3])
                ]
                for j in hits:
                    keep, drop = (j, i) if filled[j] and not filled[i] else (i, j)
                    if filled[drop]:  # both hold text: not merged
                        continue
                    rects[keep, :2] = np.minimum(rects[keep, :2], rects[drop, :2])
                    rects[keep, 2:] = np.maximum(rects[keep, 2:], rects[drop, 2:])
                    logic[keep, ::2] = np.minimum(logic[keep, ::2], logic[drop, ::2])
                    logic[keep, 1::2] = np.maximum(logic[keep, 1::2], logic[drop, 1::2])
                    resized[keep] = True
                    alive[drop] = False
                    if drop == i:
                        break

        changed = True
        while changed:
            changed = False
            for axis in (0, 1):
                size = int(logic[alive, 2 * axis + 1].max()) + 1
                for k in range(size - 1):
                    if mergeable(axis, k):
                        shift(axis, k, merge=True)
                        resolve_overlaps()
                        changed = True
                        break
        # Remove the rows / columns without text of their own: no text, or only the text of cells spanning other rows
        # / columns too (which then span one less)
        for axis in (0, 1):
            start, end = logic[:, 2 * axis], logic[:, 2 * axis + 1]
            for k in range(int(end[alive].max()), -1, -1):
                if np.any(alive & filled & (start == k) & (end == k)):
                    continue
                alive[alive & (start == k) & (end == k)] = False
                shift(axis, k, merge=False)

        keep = np.flatnonzero(alive)
        new_polys = polys.copy()
        for c_idx in np.flatnonzero(resized & alive):
            x0, y0, x1, y1 = rects[c_idx]
            rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
            new_polys[c_idx] = cls._rotate_points(rect, angle, pivot)
        new_cells = []
        for c_idx in keep:
            rs, re, cs, ce = (int(v) for v in logic[c_idx])
            new_cells.append({**cells[c_idx], "row_start": rs, "row_end": re, "col_start": cs, "col_end": ce})
        return new_cells, new_polys[keep], [cell_words[c_idx] for c_idx in keep]

    @staticmethod
    def _text_lines(centers: np.ndarray, tol: float) -> np.ndarray:
        """Group 1D word centers (along the rows' normal) into text lines: the line index of each word, from the top"""
        order = np.argsort(centers, kind="stable")
        line_ids = np.zeros(centers.size, dtype=np.int64)
        line, members = 0, [int(order[0])] if centers.size else []
        for j in order[1:]:
            if centers[j] - np.mean(centers[members]) >= tol:
                line, members = line + 1, []
            members.append(int(j))
            line_ids[j] = line
        return line_ids

    @classmethod
    def _split_rows(
        cls,
        cells: list[dict[str, Any]],
        polys: np.ndarray,
        cell_words: list[list[int]],
        word_polys: np.ndarray,
        angle: float,
    ) -> tuple[list[dict[str, Any]], np.ndarray, list[list[int]]]:
        """Split the rows merged by the table structure model, from the text lines of their cells

        The model may merge several rows of a dense table (e.g. without rulings) into a single one: a row is split
        into its text lines when most of its cells holding text (at least 2) hold several of them, so that a single
        cell wrapping its text (e.g. a long header) does not split its row. Likewise, a cell spanning several rows
        is split when each of its text lines lies in a different row, aligned with the text of that row.

        Args:
            cells: the cells of the table, with local logical coordinates
            polys: the (C, 4, 2) cell polygons (absolute coordinates)
            cell_words: the word indices of each cell
            word_polys: the (N, 4, 2) word polygons (absolute coordinates)
            angle: the angle of the table rows (radians)

        Returns:
            the cells, their polygons and the word indices of each cell
        """
        if not any(cell_words):
            return cells, polys, cell_words
        pivot = polys.reshape(-1, 2).mean(axis=0)
        frame = cls._rotate_points(polys, -angle, pivot)
        rects = [np.concatenate([f.min(axis=0), f.max(axis=0)]) for f in frame]  # x0, y0, x1, y1
        all_words = np.unique(np.concatenate([w for w in cell_words if w])).astype(int)
        w_pts = cls._rotate_points(word_polys[all_words], -angle, pivot)
        w_y0 = dict(zip(all_words.tolist(), w_pts[..., 1].min(axis=1)))
        w_y1 = dict(zip(all_words.tolist(), w_pts[..., 1].max(axis=1)))
        heights = w_pts[..., 1].max(axis=1) - w_pts[..., 1].min(axis=1)
        tol = 0.5 * float(np.median(heights)) if heights.size else 0.0
        if not tol > 0:
            return cells, polys, cell_words
        cells = [dict(c) for c in cells]
        cell_words = [list(w) for w in cell_words]
        resized: set[int] = set()

        def center(w: int) -> float:
            return (w_y0[w] + w_y1[w]) / 2

        def split_cell(c_idx: int, line_ids: np.ndarray, rows: list[tuple[int, int]]) -> None:
            """Split a cell into one cell per text line, spanning the given rows"""
            words = np.asarray(cell_words[c_idx])
            x0, y0, x1, y1 = rects[c_idx]
            n_lines = int(line_ids.max()) + 1
            # The sub-cells share the cell's extent, split midway between its text lines
            tops = [min(w_y0[w] for w in words[line_ids == line]) for line in range(n_lines)]
            bottoms = [max(w_y1[w] for w in words[line_ids == line]) for line in range(n_lines)]
            bounds = [y0] + [(bottoms[line] + tops[line + 1]) / 2 for line in range(n_lines - 1)] + [y1]
            for line, (rs, re) in enumerate(rows):
                new = {**cells[c_idx], "row_start": rs, "row_end": re}
                rect = np.array([x0, bounds[line], x1, bounds[line + 1]])
                if line == 0:
                    cells[c_idx], rects[c_idx], cell_words[c_idx] = new, rect, words[line_ids == 0].tolist()
                    resized.add(c_idx)
                else:
                    cells.append(new)
                    rects.append(rect)
                    cell_words.append(words[line_ids == line].tolist())
                    resized.add(len(cells) - 1)

        # 1. Rows merged by the model, from the bottom up so that the rows still to process do not move
        num_rows = max(c["row_end"] for c in cells) + 1
        for r in range(num_rows - 1, -1, -1):
            in_row = [i for i, c in enumerate(cells) if c["row_start"] == r and c["row_end"] == r]
            filled = [i for i in in_row if cell_words[i]]
            if len(filled) < 2:
                continue
            words = [w for i in filled for w in cell_words[i]]
            line_of = dict(zip(words, cls._text_lines(np.array([center(w) for w in words]), tol).tolist()))
            n_lines = max(line_of.values()) + 1
            multi = [i for i in filled if len({line_of[w] for w in cell_words[i]}) > 1]
            if n_lines < 2 or len(multi) < 2 or 2 * len(multi) < len(filled):
                continue
            # The rows below move down, the cells spanning the row span its new rows too
            for c in cells:
                if c["row_start"] == r and c["row_end"] == r:
                    continue
                if c["row_start"] > r:
                    c["row_start"] += n_lines - 1
                if c["row_end"] >= r:
                    c["row_end"] += n_lines - 1
            for i in in_row:
                if not cell_words[i]:
                    cells[i]["row_end"] = r + n_lines - 1
                    continue
                cell_lines = np.array([line_of[w] for w in cell_words[i]])
                present = np.unique(cell_lines)
                # Local line indices of the cell -> rows of the table
                split_cell(i, np.searchsorted(present, cell_lines), [(r + line, r + line) for line in present])

        # 2. Cells spanning several rows whose text lines lie in different rows
        logic = np.array([[c["row_start"], c["row_end"]] for c in cells], dtype=int)
        if logic[:, 1].max() > 0:
            row_y0 = np.array([r[1] for r in rects])
            row_y1 = np.array([r[3] for r in rects])
            lines = cls._grid_lines(logic[:, 0], logic[:, 1], row_y0, row_y1, int(logic[:, 1].max()) + 1)
            for i in range(len(cells)):
                rs, re = cells[i]["row_start"], cells[i]["row_end"]
                if re == rs or len(cell_words[i]) < 2:
                    continue
                cell_lines = cls._text_lines(np.array([center(w) for w in cell_words[i]]), tol)
                n_lines = int(cell_lines.max()) + 1
                if n_lines < 2:
                    continue
                line_rows = []
                for line in range(n_lines):
                    members = [w for w, lid in zip(cell_words[i], cell_lines) if lid == line]
                    lo, hi = min(w_y0[w] for w in members), max(w_y1[w] for w in members)
                    row = int(np.clip(np.searchsorted(lines, (lo + hi) / 2, side="right") - 1, rs, re))
                    # The text of the other cells of the row lies on the same line
                    aligned = any(
                        cells[j]["row_start"] == row == cells[j]["row_end"]
                        and any(min(hi, w_y1[w]) - max(lo, w_y0[w]) > 0.5 * (hi - lo) for w in cell_words[j])
                        for j in range(len(cells))
                        if j != i and cell_words[j]
                    )
                    if not aligned:
                        break
                    line_rows.append(row)
                if len(line_rows) < n_lines or len(set(line_rows)) < n_lines:
                    continue
                spans = [
                    (rs if line == 0 else line_rows[line], re if line == n_lines - 1 else line_rows[line + 1] - 1)
                    for line in range(n_lines)
                ]
                split_cell(i, cell_lines, spans)

        if not resized:
            return cells, polys, cell_words
        new_polys = [
            cls._rotate_points(
                np.array([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], dtype=np.float64), angle, pivot
            )
            if idx in resized
            else polys[idx]
            for idx, r in enumerate(rects)
        ]
        return cells, np.stack(new_polys).astype(polys.dtype), cell_words

    def _build_tables(
        self,
        boxes: np.ndarray,
        word_preds: list[tuple[str, float]],
        page_table: dict[str, Any] | list[dict[str, Any]] | None,
        page_shape: tuple[int, int] | None = None,
    ) -> tuple[list[Table], np.ndarray]:
        """Assign detected words to table cells and build the page tables.

        Every word whose center lies in a table (its layout region when provided, the cells otherwise) is part of
        the table: the words outside the predicted cells complete the grid (cf. `_complete_grid`).

        Args:
            boxes: word boxes of the page, of shape (N, 4) or (N, 4, 2), in relative coordinates
            word_preds: list of (text, confidence) for each of the N words
            page_table: the table structure prediction(s) for the page. Either a single grid
                `{"cells": [{"geometry", "score", "row_start", "row_end", "col_start", "col_end"}], "num_rows",
                "num_cols", "geometry"}` (cell geometries in page-relative coordinates, the optional "geometry" being
                the (4, 2) polygon of the table region), a list of such grids, or None
            page_shape: the (height, width) of the page, to process the geometry in an isotropic space

        Returns:
            a tuple with the list of `Table` objects of the page (one per provided table) and a boolean mask of
            shape (N,) that is True for words assigned to a table (to be removed from `blocks`)
        """
        consumed = np.zeros(boxes.shape[0], dtype=bool)
        grids = self._table_grids(page_table)
        if len(grids) == 0:
            return [], consumed
        # The geometries follow the page's word geometry: straight 2-point boxes for (N, 4) word boxes, 4-point polygons
        # for (N, 4, 2) ones.
        straight = boxes.ndim != 3
        scale = np.ones(2) if page_shape is None else np.array([page_shape[1], page_shape[0]], dtype=np.float64)
        if boxes.ndim == 3:  # (N, 4, 2) polygons, ordered TL, TR, BR, BL
            word_polys = boxes.astype(np.float64) * scale
        else:  # (N, 4) straight boxes (xmin, ymin, xmax, ymax)
            x0, y0, x1, y1 = boxes.reshape(-1, 4).T.astype(np.float64)
            word_polys = np.stack([np.stack(pt, axis=-1) for pt in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))], 1) * scale
        centers = word_polys.mean(axis=1)
        heights = np.linalg.norm(word_polys[:, 3] - word_polys[:, 0], axis=1)  # rotation-invariant (left edge)

        tables: list[Table] = []
        for cells, num_rows, num_cols, region in grids:
            polys = np.stack([self._as_cell_polygon(cell["geometry"]) for cell in cells]) * scale  # (C, 4, 2)
            cell_words = self._assign_cell_words(centers, polys, consumed)
            top_edges = polys[:, 1] - polys[:, 0]  # TR - TL
            angle = 0.0 if straight else float(np.median(np.arctan2(top_edges[:, 1], top_edges[:, 0])))
            # The words of the table that lie in no cell: in its layout region, or in the cells' outline
            free = np.flatnonzero(~consumed)
            outline = cv2.boxPoints(cv2.minAreaRect(polys.reshape(-1, 2).astype(np.float32))).reshape(1, 4, 2)
            in_table = self._points_in_polygons(centers[free], outline)[:, 0]
            if region is not None:
                in_table |= self._points_in_polygons(centers[free], (region * scale)[None])[:, 0]
            leftover = free[in_table]
            if leftover.size > 0:
                cells, polys, cell_words = self._complete_grid(cells, polys, cell_words, leftover, word_polys, angle)
                consumed[leftover] = True
                cells, num_rows, num_cols = self._localize_logic(cells)
            cells, polys, cell_words = self._simplify_grid(cells, polys, cell_words, word_polys, angle)
            cells, polys, cell_words = self._split_rows(cells, polys, cell_words, word_polys, angle)
            cells, num_rows, num_cols = self._localize_logic(cells)
            # Reading order inside the cells: rows top to bottom, words left to right (table axes)
            order_centers = self._rotate_points(centers, -angle, centers.mean(axis=0)) if centers.size else centers
            table_cells = []
            for cell, poly, w_idcs in zip(cells, polys / scale, cell_words):
                if len(w_idcs) > 0:
                    ordered = self._order_cell_words(w_idcs, order_centers, heights)
                    value = " ".join(word_preds[i][0] for i in ordered)
                    confidence = float(np.mean([word_preds[i][1] for i in ordered]))
                else:
                    value, confidence = "", float(cell["score"] or 0.0)
                table_cells.append(
                    TableCell(
                        value=value,
                        confidence=confidence,
                        geometry=self._export_geometry(poly, straight),
                        row_start=int(cell["row_start"]),
                        row_end=int(cell["row_end"]),
                        col_start=int(cell["col_start"]),
                        col_end=int(cell["col_end"]),
                        objectness_score=None if cell["score"] is None else float(cell["score"]),
                    )
                )
            # Enclosing geometry of the whole table
            pts = (polys / scale).reshape(-1, 2)
            (xmin, ymin), (xmax, ymax) = pts.min(axis=0), pts.max(axis=0)
            corners = np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            scores = [float(cell["score"]) for cell in cells if cell["score"] is not None]
            tables.append(
                Table(
                    cells=table_cells,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    geometry=self._export_geometry(corners, straight),
                    confidence=float(np.mean(scores)) if scores else 0.0,
                )
            )
        return tables, consumed

    @staticmethod
    def _export_geometry(poly: np.ndarray, straight: bool) -> Any:
        """Export a (4, 2) polygon as a straight 2-point box or as a 4-point polygon, with Python floats"""
        if straight:
            return (
                (float(poly[:, 0].min()), float(poly[:, 1].min())),
                (float(poly[:, 0].max()), float(poly[:, 1].max())),
            )
        return tuple(tuple(float(c) for c in pt) for pt in poly.tolist())

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
            layout: optional layout regions of the page

        Returns:
            list of block elements
        """
        if boxes.shape[0] != len(word_preds):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(word_preds)}")

        if boxes.shape[0] == 0:
            return []

        straight_or_rotated = boxes if boxes.ndim == 3 else boxes[:, :4]
        if not self.resolve_lines:
            # One line for all boxes, one block for the line
            lines = [self._sort_boxes(straight_or_rotated, shape)[0].tolist()]
        else:
            word_regions = (
                self._word_regions(boxes, [elt.geometry for elt in layout], [elt.type for elt in layout], shape)
                if layout
                else None
            )
            lines = self._resolve_lines(straight_or_rotated, shape, word_regions)
        blocks = (
            self._resolve_blocks(straight_or_rotated, lines)
            if self.resolve_lines and self.resolve_blocks and len(lines) > 1
            else [lines]
        )

        def _word(idx: int) -> Word:
            geometry = (
                tuple(tuple(pt) for pt in boxes[idx].tolist())
                if boxes.ndim == 3
                else ((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3]))
            )
            return Word(*word_preds[idx], geometry, float(objectness_scores[idx]), crop_orientations[idx])  # type: ignore[arg-type]

        return [Block([Line([_word(idx) for idx in line]) for line in block]) for block in blocks]

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
            page_tables, consumed = self._build_tables(page_boxes, word_preds, page_table, shape)
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
        paragraph_break: relative length of the minimum space separating paragraphs (scaled up for large type)
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
