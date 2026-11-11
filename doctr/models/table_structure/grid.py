# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Assemble the cells of a table structure model and the words of the table into a consistent grid.

Boxes are `(x0, y0, x1, y1)` in the upright table frame, logical coordinates are `(row_start, row_end, col_start,
col_end)` (0-indexed, inclusive).
"""

from typing import Any

import numpy as np

__all__ = ["fill_table_grid", "assemble_page_table"]


def as_polygons(geometries: Any) -> np.ndarray:
    """Convert straight boxes or (TL, TR, BR, BL) polygons to (N, 4, 2) float32 polygons"""
    flat = np.asarray(geometries, dtype=np.float32)
    flat = flat.reshape(len(flat), -1) if flat.size else np.zeros((0, 4), dtype=np.float32)
    if flat.shape[1] == 4:
        x0, y0, x1, y1 = flat.T
        return np.stack([np.stack(pt, axis=-1) for pt in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))], axis=1)
    return flat.reshape(-1, 4, 2)


def _grid_lines(start: np.ndarray, end: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: int) -> np.ndarray:
    """Positions of the `size + 1` grid lines of an axis: the median of the cell edges on each line, interpolated
    where no edge lies"""
    edges: list[list[float]] = [[] for _ in range(size + 1)]
    for s, e, low, high in zip(start, end, lo, hi):
        edges[int(s)].append(float(low))
        edges[int(e) + 1].append(float(high))
    known = [k for k, values in enumerate(edges) if values]
    return np.maximum.accumulate(np.interp(np.arange(size + 1), known, [float(np.median(edges[k])) for k in known]))


def _group(values: np.ndarray, tol: float) -> np.ndarray:
    """Group index of each 1D position, a group gathering the positions closer than `tol` to its mean"""
    groups = np.zeros(values.size, dtype=np.int64)
    group = 0
    members: list[int] = []
    for j in np.argsort(values, kind="stable"):
        if members and values[j] - np.mean(values[members]) >= tol:
            group, members = group + 1, []
        groups[j] = group
        members.append(j)
    return groups


def _axis_slots(lo: np.ndarray, hi: np.ndarray, lines: np.ndarray, line_tol: float | None) -> np.ndarray:
    """Index of each word along an axis. The words beyond the grid get new indices, grouped by text line (rows,
    `line_tol`) or by overlapping extent (columns, `line_tol=None`)"""
    size, center = lines.size - 1, (lo + hi) / 2
    idx = np.clip(np.searchsorted(lines, center, side="right") - 1, 0, size - 1)
    for before, mask in ((True, center < lines[0]), (False, center > lines[-1])):
        members = np.flatnonzero(mask)
        if members.size == 0:
            continue
        if line_tol is not None:
            groups = _group(center[members], line_tol)
        else:  # chain the overlapping extents
            groups, group, reach = np.zeros(members.size, dtype=np.int64), -1, -np.inf
            for j in np.argsort(lo[members], kind="stable"):
                if lo[members[j]] > reach:
                    group += 1
                groups[j], reach = group, max(reach, hi[members[j]])
        idx[members] = groups - (groups.max() + 1) if before else size + groups
    return idx


class _Grid:
    """The cells of a table: boxes, logical coordinates, word indices and source cell (-1 if made from words)"""

    def __init__(self, boxes: np.ndarray, logic: np.ndarray, words: np.ndarray):
        self.boxes = [np.asarray(b, dtype=np.float64) for b in boxes]
        self.logic = [list(map(int, lc)) for lc in logic]
        self.cell_words: list[list[int]] = [[] for _ in self.boxes]
        self.origin = list(range(len(self.boxes)))
        self.words = words

    def add(self, box: np.ndarray, logic: list[int], words: list[int], origin: int = -1) -> None:
        self.boxes.append(np.asarray(box, dtype=np.float64))
        self.logic.append(logic)
        self.cell_words.append(words)
        self.origin.append(origin)

    def keep(self, mask: np.ndarray) -> None:
        for name in ("boxes", "logic", "cell_words", "origin"):
            setattr(self, name, [v for v, k in zip(getattr(self, name), mask) if k])

    def localize(self) -> None:
        """Shift the logical coordinates so that the first row and column are 0"""
        logic = np.asarray(self.logic)
        logic -= [logic[:, 0].min()] * 2 + [logic[:, 2].min()] * 2
        self.logic = logic.tolist()

    def text_box(self, c_idx: int) -> np.ndarray:
        boxes = self.words[self.cell_words[c_idx]]
        return np.concatenate([boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)])

    def lines(self, axis: int) -> np.ndarray:
        """The grid lines of the rows (axis 0) or columns (axis 1)"""
        logic, boxes = np.asarray(self.logic), np.asarray(self.boxes)
        lo, hi = (1, 3) if axis == 0 else (0, 2)
        size = int(logic[:, 2 * axis + 1].max()) + 1
        return _grid_lines(logic[:, 2 * axis], logic[:, 2 * axis + 1], boxes[:, lo], boxes[:, hi], size)


def _complete(grid: _Grid, leftover: np.ndarray, line_height: float) -> None:
    """Place the words lying in no cell: in the cell covering their slot, or in a new cell (a missed cell, or a new
    row / column beyond the grid)"""
    words = grid.words[leftover]
    row_lines, col_lines = grid.lines(0), grid.lines(1)
    rows = _axis_slots(words[:, 1], words[:, 3], row_lines, line_height / 2)
    cols = _axis_slots(words[:, 0], words[:, 2], col_lines, None)
    occupied: dict[tuple[int, int], int] = {}
    for c_idx, (rs, re, cs, ce) in enumerate(grid.logic):
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                occupied.setdefault((r, c), c_idx)

    new_cells: dict[tuple[int, int, int, int], list[int]] = {}
    in_grid = (rows >= 0) & (rows < row_lines.size - 1)
    for j in np.flatnonzero(in_grid):
        slot = (int(rows[j]), int(cols[j]))
        if slot in occupied:
            grid.cell_words[occupied[slot]].append(int(leftover[j]))
        else:
            new_cells.setdefault((slot[0], slot[0], slot[1], slot[1]), []).append(int(j))
    for r in np.unique(rows[~in_grid]):
        # New row: a word starts a new cell in another column, unless it continues a sentence
        members = np.flatnonzero(rows == r)
        groups: list[list[int]] = []
        for j in members[np.argsort(words[members, 0], kind="stable")]:
            prev = groups[-1][-1] if groups else None
            if prev is None or (cols[j] != cols[prev] and words[j, 0] - words[prev, 2] >= line_height / 2):
                groups.append([])
            groups[-1].append(int(j))
        spans: list[tuple[int, int, list[int]]] = []
        for group in groups:
            lo, hi = int(cols[group].min()), int(cols[group].max())
            while spans and lo <= spans[-1][1]:  # overlapping words: a column belongs to a single cell
                prev_lo, prev_hi, prev_group = spans.pop()
                lo, hi, group = min(lo, prev_lo), max(hi, prev_hi), prev_group + group
            spans.append((lo, hi, group))
        for lo, hi, group in spans:
            new_cells[(int(r), int(r), lo, hi)] = group

    def band(lines: np.ndarray, start: int, end: int) -> tuple[float, float]:
        # Unbounded beyond the grid
        size = lines.size - 1
        return (lines[start] if 0 <= start < size else np.inf, lines[end + 1] if 0 <= end < size else -np.inf)

    for (rs, re, cs, ce), positions in new_cells.items():
        text = words[positions]
        (x0, x1), (y0, y1) = band(col_lines, cs, ce), band(row_lines, rs, re)
        box = [min(x0, text[:, 0].min()), min(y0, text[:, 1].min())]
        box += [max(x1, text[:, 2].max()), max(y1, text[:, 3].max())]
        grid.add(np.array(box), [rs, re, cs, ce], [int(leftover[j]) for j in positions])
    grid.localize()


def _merge_and_prune(grid: _Grid) -> None:
    """Merge the adjacent rows / columns the model split (complementary text on the same text line / column), then
    remove the rows / columns in which no single-span cell holds text"""
    logic = np.asarray(grid.logic)
    filled = np.array([len(w) > 0 for w in grid.cell_words])
    if not filled.any():
        return
    text = np.array([grid.text_box(i) if filled[i] else np.full(4, np.nan) for i in range(len(grid.boxes))])
    boxes = np.asarray(grid.boxes)
    alive = np.ones(len(boxes), dtype=bool)

    def mergeable(axis: int, k: int) -> bool:
        start, end = logic[:, 2 * axis], logic[:, 2 * axis + 1]
        o_start, o_end = logic[:, 2 - 2 * axis], logic[:, 3 - 2 * axis]
        in_k, in_next = alive & (start <= k) & (end >= k), alive & (start <= k + 1) & (end >= k + 1)
        a, b = np.flatnonzero(in_k & ~in_next & filled), np.flatnonzero(in_next & ~in_k & filled)
        if a.size == 0 or b.size == 0:
            return False
        if np.any((o_start[a, None] <= o_end[None, b]) & (o_start[None, b] <= o_end[a, None])):
            return False  # both hold text in a same column / row
        lo, hi = (1, 3) if axis == 0 else (0, 2)
        a_lo, a_hi, b_lo, b_hi = text[a, lo].min(), text[a, hi].max(), text[b, lo].min(), text[b, hi].max()
        return bool(min(a_hi, b_hi) - max(a_lo, b_lo) >= 0.5 * min(a_hi - a_lo, b_hi - b_lo))

    def shift(axis: int, k: int, merge: bool) -> None:
        # Merge the index k + 1 into k, or remove the index k
        start, end = logic[:, 2 * axis], logic[:, 2 * axis + 1]
        end[alive & ((end > k) if merge else (end >= k))] -= 1
        start[alive & (start > k)] -= 1

    def absorb_overlaps() -> None:
        # A cell absorbs the empty cells now sharing a slot with it
        for i in np.flatnonzero(alive & filled).tolist() + np.flatnonzero(alive & ~filled).tolist():
            if not alive[i]:
                continue
            others = np.flatnonzero(alive & ~filled)
            others = others[others != i]
            hits = others[
                (logic[others, 0] <= logic[i, 1])
                & (logic[i, 0] <= logic[others, 1])
                & (logic[others, 2] <= logic[i, 3])
                & (logic[i, 2] <= logic[others, 3])
            ]
            if hits.size:
                boxes[i, :2] = np.minimum(boxes[i, :2], boxes[hits, :2].min(axis=0))
                boxes[i, 2:] = np.maximum(boxes[i, 2:], boxes[hits, 2:].max(axis=0))
                logic[i, ::2] = np.minimum(logic[i, ::2], logic[hits, ::2].min(axis=0))
                logic[i, 1::2] = np.maximum(logic[i, 1::2], logic[hits, 1::2].max(axis=0))
                alive[hits] = False

    changed = True
    while changed:
        changed = False
        for axis in (0, 1):
            for k in range(int(logic[alive, 2 * axis + 1].max())):
                if mergeable(axis, k):
                    shift(axis, k, merge=True)
                    absorb_overlaps()
                    changed = True
                    break
    for axis in (0, 1):
        start, end = logic[:, 2 * axis], logic[:, 2 * axis + 1]
        for k in range(int(end[alive].max()), -1, -1):
            if not np.any(alive & filled & (start == k) & (end == k)):
                alive[alive & (start == k) & (end == k)] = False
                shift(axis, k, merge=False)
    grid.boxes, grid.logic = list(boxes), logic.tolist()
    grid.keep(alive)
    grid.localize()


def _split_rows(grid: _Grid, line_height: float) -> None:
    """Split the rows the model merged along their text lines: a row most of whose filled cells hold several lines
    (a single cell wrapping its text does not split its row), and a spanning cell whose text lines each lie level
    with a different row"""
    centers = (grid.words[:, 1] + grid.words[:, 3]) / 2

    def split(c_idx: int, line_of: np.ndarray, spans: list[tuple[int, int]]) -> None:
        # One cell per text line, cut midway between the lines (the text may lie beyond the box of its cell)
        words, (x0, y0, x1, y1) = np.asarray(grid.cell_words[c_idx]), grid.boxes[c_idx]
        tops = [grid.words[words[line_of == k], 1].min() for k in range(len(spans))]
        bottoms = [grid.words[words[line_of == k], 3].max() for k in range(len(spans))]
        cuts = [min(y0, tops[0]), *[(b + t) / 2 for b, t in zip(bottoms[:-1], tops[1:])], max(y1, bottoms[-1])]
        _, _, cs, ce = grid.logic[c_idx]
        for k, (rs, re) in enumerate(spans):
            box, logic, members = (
                np.array([x0, cuts[k], x1, cuts[k + 1]]),
                [rs, re, cs, ce],
                words[line_of == k].tolist(),
            )
            if k == 0:
                grid.boxes[c_idx], grid.logic[c_idx], grid.cell_words[c_idx] = box, logic, members
            else:
                grid.add(box, logic, members, grid.origin[c_idx])

    # From the bottom up, so that the rows still to process do not move
    for r in range(max(lc[1] for lc in grid.logic), -1, -1):
        in_row = [i for i, lc in enumerate(grid.logic) if lc[0] == lc[1] == r]
        filled = [i for i in in_row if grid.cell_words[i]]
        words = [w for i in filled for w in grid.cell_words[i]]
        if len(filled) < 2:
            continue
        line_of = dict(zip(words, _group(centers[words], line_height / 2).tolist()))
        n_lines = max(line_of.values()) + 1
        multi = [i for i in filled if len({line_of[w] for w in grid.cell_words[i]}) > 1]
        if len(multi) < 2 or 2 * len(multi) < len(filled):
            continue
        for lc in grid.logic:  # the rows below move down, the cells spanning the row grow
            if not lc[0] == lc[1] == r:
                lc[0] += n_lines - 1 if lc[0] > r else 0
                lc[1] += n_lines - 1 if lc[1] >= r else 0
        for i in in_row:
            if not grid.cell_words[i]:
                grid.logic[i][1] = r + n_lines - 1
                continue
            lines = np.array([line_of[w] for w in grid.cell_words[i]])
            present = np.unique(lines)
            split(i, np.searchsorted(present, lines), [(r + k, r + k) for k in present])

    def level_with_row(c_idx: int, row: int, lo: float, hi: float) -> bool:
        # Whether a text line (lo to hi) is level with the text of another cell of the row
        return any(
            j != c_idx
            and grid.logic[j][0] == row == grid.logic[j][1]
            and any(min(hi, y1) - max(lo, y0) > 0.5 * (hi - lo) for _, y0, _, y1 in grid.words[grid.cell_words[j]])
            for j in range(len(grid.boxes))
        )

    row_lines = grid.lines(0)
    for i in range(len(grid.boxes)):
        rs, re = grid.logic[i][:2]
        if re == rs or len(grid.cell_words[i]) < 2:
            continue
        idcs = np.asarray(grid.cell_words[i])
        cell_lines = _group(centers[idcs], line_height / 2)
        rows: list[int] = []  # row of each text line
        for k in range(int(cell_lines.max()) + 1):
            lo, hi = grid.words[idcs[cell_lines == k], 1].min(), grid.words[idcs[cell_lines == k], 3].max()
            row = int(np.clip(np.searchsorted(row_lines, (lo + hi) / 2, side="right") - 1, rs, re))
            if row in rows or not level_with_row(i, row, lo, hi):
                rows = []
                break
            rows.append(row)
        if len(rows) > 1:
            # Each text line spans until the row of the next one
            starts, ends = [rs, *rows[1:]], [*(row - 1 for row in rows[1:]), re]
            split(i, cell_lines, list(zip(starts, ends)))


def _reading_order(words: np.ndarray, idcs: list[int], line_height: float) -> list[int]:
    """Order the words of a cell: text lines top to bottom, words left to right"""
    lines = _group((words[idcs, 1] + words[idcs, 3]) / 2, line_height / 2)
    return [idcs[j] for j in np.lexsort((words[idcs, 0], lines))]


def fill_table_grid(
    cells: np.ndarray, logic: np.ndarray, words: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[list[int]], list[int]]:
    """Make the grid of a table consistent with its text, and assign every word of the table to a cell.

    The words outside of every cell complete the grid, the rows / columns split by the model are merged, the ones
    without text are removed, and the rows merged by the model are split along their text lines.

    Args:
        cells: the (C, 4) cell boxes predicted by the table model
        logic: the (C, 4) logical coordinates of the cells
        words: the (W, 4) boxes of the words of the table, in the same frame

    Returns:
        the cell boxes, their local logical coordinates, the indices of their words in reading order, and the index of
        the predicted cell each one comes from (-1 for a cell made from words)
    """
    grid = _Grid(cells, logic, np.asarray(words, dtype=np.float64).reshape(-1, 4))
    if len(grid.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0, 4), dtype=np.int64), [], []
    grid.localize()
    w = grid.words
    if w.shape[0] > 0:
        centers = (w[:, :2] + w[:, 2:]) / 2
        boxes = np.asarray(grid.boxes)
        inside = (
            (boxes[None, :, 0] <= centers[:, None, 0])
            & (centers[:, None, 0] <= boxes[None, :, 2])
            & (boxes[None, :, 1] <= centers[:, None, 1])
            & (centers[:, None, 1] <= boxes[None, :, 3])
        )
        assigned = inside.any(axis=1)
        for w_idx, c_idx in zip(np.flatnonzero(assigned), inside[assigned].argmax(axis=1)):
            grid.cell_words[c_idx].append(int(w_idx))
        line_height = float(np.median(w[:, 3] - w[:, 1]))
        line_height = line_height if line_height > 0 else 1.0
        if not assigned.all():
            _complete(grid, np.flatnonzero(~assigned), line_height)
        _merge_and_prune(grid)
        _split_rows(grid, line_height)
        grid.localize()
        grid.cell_words = [_reading_order(w, idcs, line_height) if idcs else [] for idcs in grid.cell_words]
    return np.asarray(grid.boxes), np.asarray(grid.logic, dtype=np.int64), grid.cell_words, grid.origin


def _rotate(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    """Rotate (..., 2) points by `angle` radians around `pivot`"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x, y = points[..., 0] - pivot[0], points[..., 1] - pivot[1]
    return np.stack([pivot[0] + x * cos_a - y * sin_a, pivot[1] + x * sin_a + y * cos_a], axis=-1)


def assemble_page_table(
    cells: list[dict[str, Any]], word_boxes: np.ndarray, page_shape: tuple[int, int], taken: np.ndarray
) -> tuple[list[dict[str, Any]], int, int]:
    """Assemble a table given in page coordinates with the words of the page.

    The cells listing their words (`"words"`) keep them, otherwise the free words inside the outline of the cells are
    assigned with `fill_table_grid`.

    Args:
        cells: the cells `{"geometry", "score", "row_start", "row_end", "col_start", "col_end"}` of the table, with
            page-relative geometries (straight boxes or polygons)
        word_boxes: the (N, 4) or (N, 4, 2) page-relative word boxes
        page_shape: the (height, width) of the page
        taken: the (N,) mask of the words already assigned to a table, updated in place

    Returns:
        the cells (local logical coordinates, (4, 2) page-relative polygon, `"words"` in reading order), the number of
        rows and the number of columns
    """
    scale = np.array([page_shape[1], page_shape[0]], dtype=np.float64)
    polys = as_polygons([cell["geometry"] for cell in cells]) * scale
    logic = np.array([[c["row_start"], c["row_end"], c["col_start"], c["col_end"]] for c in cells], dtype=np.int64)
    if all("words" in cell for cell in cells):
        cell_words = [[int(i) for i in cell["words"] if not taken[i]] for cell in cells]
        origin = list(range(len(cells)))
        logic -= [logic[:, 0].min()] * 2 + [logic[:, 2].min()] * 2
    else:
        # Upright frame of the table rows
        top_edges = polys[:, 1] - polys[:, 0]
        angle, pivot = float(np.median(np.arctan2(top_edges[:, 1], top_edges[:, 0]))), polys.reshape(-1, 2).mean(0)
        upright = _rotate(polys, -angle, pivot)
        boxes = np.concatenate([upright.min(axis=1), upright.max(axis=1)], axis=1)
        free, word_coords = np.flatnonzero(~taken), np.zeros((0, 4))
        if free.size:
            words = _rotate(as_polygons(word_boxes[free]) * scale, -angle, pivot)
            centers = words.mean(axis=1)
            inside = np.all((centers >= boxes[:, :2].min(axis=0)) & (centers <= boxes[:, 2:].max(axis=0)), axis=1)
            free, words = free[inside], words[inside]
            word_coords = np.concatenate([words.min(axis=1), words.max(axis=1)], axis=1)
        boxes, logic, local_words, origin = fill_table_grid(boxes, logic, word_coords)
        cell_words = [[int(free[j]) for j in idcs] for idcs in local_words]
        corners = boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]]
        polys = _rotate(corners, angle, pivot)
    for idcs in cell_words:
        taken[idcs] = True
    out = []
    for poly, (rs, re, cs, ce), idcs, source in zip(polys / scale, logic, cell_words, origin):
        score = None if source < 0 else cells[source]["score"]
        out.append({
            "geometry": poly,
            "score": None if score is None else float(score),
            "row_start": int(rs),
            "row_end": int(re),
            "col_start": int(cs),
            "col_end": int(ce),
            "words": idcs,
        })
    return out, int(logic[:, 1].max()) + 1, int(logic[:, 3].max()) + 1
