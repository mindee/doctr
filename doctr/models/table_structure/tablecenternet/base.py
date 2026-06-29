# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: decode logic ported from https://github.com/dreamy-xay/TableCenterNet

import math

import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon

from doctr.models.core import BaseModel

__all__ = ["_TableCenterNet", "TableCenterNetPostProcessor"]


def _get_logic_coords(lc_logic: np.ndarray, col_span: int, row_span: int) -> tuple[int, int, int, int]:
    """Resolve a cell's logical coordinates (start/end column and row) from the per-corner logical
    predictions (`lc_logic` is a (4, 2) array of [col, row] for corners TL, TR, BR, BL) and the cell span.
    Pure numpy port of the reference `get_logic_coords`."""
    col_span = max(1, col_span)
    row_span = max(1, row_span)
    col_lc = [max(1, int(round(float(p)))) for p in lc_logic[:, 0]]
    row_lc = [max(1, int(round(float(p)))) for p in lc_logic[:, 1]]
    cols, rows = lc_logic[:, 0], lc_logic[:, 1]

    if col_lc[0] == col_lc[3]:
        start_col = col_lc[0]
        end_col = start_col + col_span - 1
    elif col_lc[1] == col_lc[2]:
        end_col = max(col_span + 1, col_lc[1]) - 1
        start_col = end_col + 1 - col_span
    elif abs(cols[0] - cols[3]) <= abs(cols[1] - cols[2]):
        start_col = max(1, int(round((cols[0] + cols[3]) / 2.0)))
        end_col = start_col + col_span - 1
    else:
        end_col = max(col_span + 1, int(round((cols[1] + cols[2]) / 2.0))) - 1
        start_col = end_col + 1 - col_span

    if row_lc[0] == row_lc[1]:
        start_row = row_lc[0]
        end_row = start_row + row_span - 1
    elif row_lc[2] == row_lc[3]:
        end_row = max(row_span + 1, row_lc[2]) - 1
        start_row = end_row + 1 - row_span
    elif abs(rows[0] - rows[1]) <= abs(rows[2] - rows[3]):
        start_row = max(1, int(round((rows[0] + rows[1]) / 2.0)))
        end_row = start_row + row_span - 1
    else:
        end_row = max(row_span + 1, int(round((rows[2] + rows[3]) / 2.0))) - 1
        start_row = end_row + 1 - row_span

    return start_col, end_col, start_row, end_row


def _bbox_overlap_query(center_polys: np.ndarray, corner_polys: np.ndarray) -> list[np.ndarray]:
    """For each center polygon, the indices of corner polygons whose axis-aligned bounding boxes overlap."""
    c_xmin, c_xmax = center_polys[:, 0::2].min(1), center_polys[:, 0::2].max(1)
    c_ymin, c_ymax = center_polys[:, 1::2].min(1), center_polys[:, 1::2].max(1)
    k_xmin, k_xmax = corner_polys[:, 0::2].min(1), corner_polys[:, 0::2].max(1)
    k_ymin, k_ymax = corner_polys[:, 1::2].min(1), corner_polys[:, 1::2].max(1)
    out = []
    for i in range(center_polys.shape[0]):
        x_ok = (k_xmin <= c_xmax[i]) & (k_xmax >= c_xmin[i])
        y_ok = (k_ymin <= c_ymax[i]) & (k_ymax >= c_ymin[i])
        out.append(np.nonzero(x_ok & y_ok)[0])
    return out


def _lookup_logic(lc_map: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample the (2, H, W) logical-coordinate map at a clamped pixel location."""
    h, w = lc_map.shape[1:]
    xi = 0 if x < 0 else (w - 1 if x >= w else int(x))
    yi = 0 if y < 0 else (h - 1 if y >= h else int(y))
    return lc_map[:, yi, xi]


def _ensure_simple_quads(polys: np.ndarray) -> np.ndarray:
    """Guarantee each predicted quad is a simple (non-self-intersecting) polygon.

    Args:
        polys: predicted quads, shape (N, 4, 2)

    Returns:
        the quads with every self-intersecting one reordered into a simple polygon, shape (N, 4, 2)
    """
    for i in range(polys.shape[0]):
        if not Polygon(polys[i]).is_valid:
            centroid = polys[i].mean(axis=0)
            angles = np.arctan2(polys[i, :, 1] - centroid[1], polys[i, :, 0] - centroid[0])
            polys[i] = polys[i][np.argsort(angles)]
    return polys


class TableCenterNetPostProcessor:
    """TableCenterNet post-processor turning the model's *decoded* key-points into table cells.

    The cell geometry is returned in **relative** coordinates ([0, 1] w.r.t. the model input), so the
    predictor can undo the pre-processor's padding/resize like the other docTR predictors. When
    `assume_straight_pages=True`, geometries are axis-aligned boxes of shape `(N, 4)`; otherwise they
    are quadrilaterals of shape `(N, 4, 2)`.

    Args:
        center_thresh: minimum score for a cell center to be kept
        corner_thresh: minimum score for a corner to be used during relocation
        not_relocate: if True, skip the corner-relocation step (faster, less accurate)
        assume_straight_pages: whether the pages are assumed to be straight (i.e., no rotation)
    """

    def __init__(
        self,
        center_thresh: float = 0.3,
        corner_thresh: float = 0.3,
        not_relocate: bool = False,
        assume_straight_pages: bool = True,
    ) -> None:
        self.center_thresh = center_thresh
        self.corner_thresh = corner_thresh
        self.not_relocate = not_relocate
        self.assume_straight_pages = assume_straight_pages
        # Cell score decay: cells optimised on <= 2 corners get their score scaled.
        self.cell_min_optimize_count = 2
        self.cell_decay_thresh = 0.4

    def _relocate(self, decoded: dict[str, np.ndarray], b: int):
        cp = decoded["center_polygons"][b].copy()  # (Kc, 8)
        cs = decoded["center_scores"][b].copy()  # (Kc,)
        spans = decoded["center_spans"][b]  # (Kc, 2)
        corner_polys = decoded["corner_polygons"][b]  # (Kn, 8)
        corner_scores = decoded["corner_scores"][b]  # (Kn,)
        corner_pts = decoded["corner_points"][b]  # (Kn, 2)
        corner_logics = decoded["corner_logics"][b]  # (Kn, 2)
        lc_map = decoded["lc"][b]  # (2, H, W)

        valid_c = np.nonzero(cs >= self.center_thresh)[0]
        valid_k = np.nonzero(corner_scores >= self.corner_thresh)[0]
        queries = (
            _bbox_overlap_query(cp[valid_c], corner_polys[valid_k])
            if valid_k.size
            else [np.array([], int)] * valid_c.size
        )

        logic = np.zeros((cp.shape[0], 4), dtype=np.int32)
        corner_count = np.zeros(cp.shape[0], dtype=np.int32)
        for qi, i in enumerate(valid_c):
            center_poly = Polygon(cp[i].reshape(4, 2))
            cell = cp[i].reshape(4, 2)
            origin = decoded["center_polygons"][b][i].reshape(4, 2)
            lc_logic: list[np.ndarray | None] = [None, None, None, None]
            n_used = n_repeat = 0
            for j in valid_k[queries[qi]]:
                cx, cy = corner_pts[j]
                if not any(Point(p).within(center_poly) for p in corner_polys[j].reshape(4, 2)):
                    continue
                # nearest corner index is computed on the ORIGINAL polygon
                idx = int(np.argmin(((origin - [cx, cy]) ** 2).sum(1)))
                ox, oy = origin[idx]
                px, py = cell[idx]
                if px == ox and py == oy:
                    n_used += 1
                    cell[idx] = [cx, cy]
                    lc_logic[idx] = corner_logics[j]
                elif (ox - px) ** 2 + (oy - py) ** 2 >= (ox - cx) ** 2 + (oy - cy) ** 2:
                    n_repeat += 1
                    cell[idx] = [cx, cy]
                    lc_logic[idx] = corner_logics[j]
            corner_count[i] = n_used + n_repeat
            for k in range(4):
                if lc_logic[k] is None:
                    lc_logic[k] = _lookup_logic(lc_map, cell[k][0], cell[k][1])
            col_span, row_span = int(round(float(spans[i][0]))), int(round(float(spans[i][1])))
            logic[i] = _get_logic_coords(np.stack(lc_logic), col_span, row_span)  # type: ignore[arg-type]
            cp[i] = cell.reshape(8)

        # Score decay for under-optimised cells, then re-sort
        keep_high = cs >= self.center_thresh
        decay = keep_high & (corner_count <= self.cell_min_optimize_count)
        cs[decay] *= self.cell_decay_thresh
        order = np.argsort(-cs)
        return cp[order], cs[order], logic[order]

    def _simple(self, decoded: dict[str, np.ndarray], b: int):
        cp = decoded["center_polygons"][b]
        cs = decoded["center_scores"][b]
        spans = decoded["center_spans"][b]
        lc_map = decoded["lc"][b]
        logic = np.zeros((cp.shape[0], 4), dtype=np.int32)
        for i in np.nonzero(cs >= self.center_thresh)[0]:
            cell = cp[i].reshape(4, 2)
            lc_logic = np.stack([_lookup_logic(lc_map, cell[k][0], cell[k][1]) for k in range(4)])
            col_span, row_span = int(round(float(spans[i][0]))), int(round(float(spans[i][1])))
            logic[i] = _get_logic_coords(lc_logic, col_span, row_span)
        return cp, cs, logic

    def __call__(self, decoded: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
        feat_h, feat_w = decoded["feat_size"]
        scale = np.array([feat_w, feat_h], dtype=np.float32)
        results: list[dict[str, np.ndarray]] = []
        for b in range(decoded["center_polygons"].shape[0]):
            cp, cs, logic = self._simple(decoded, b) if self.not_relocate else self._relocate(decoded, b)
            keep = cs >= self.center_thresh
            polys = cp[keep].reshape(-1, 4, 2) / scale  # relative coordinates
            polys = _ensure_simple_quads(np.clip(polys.astype(np.float32), 0, 1))
            cells = (
                np.concatenate([polys.min(axis=1), polys.max(axis=1)], axis=1).astype(np.float32)
                if self.assume_straight_pages
                else polys
            )
            results.append({
                "polygons": cells,  # (N, 4) boxes or (N, 4, 2) quads in relative coordinates
                "scores": cs[keep].astype(np.float32),
                "logical": (logic[keep] - 1).astype(np.int32),  # start_col, end_col, start_row, end_row (0-indexed)
            })
        return results


def _gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.7) -> float:
    height, width = det_size
    a1, b1, c1 = 1, height + width, width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1**2 - 4 * a1 * c1, 0))) / 2
    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2**2 - 4 * a2 * c2, 0))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3**2 - 4 * a3 * c3, 0))) / 2
    return min(r1, r2, r3)


def _gaussian_2d(shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    m, n = ((s - 1) / 2 for s in shape)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]  # type: ignore[misc]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def _draw_umich_gaussian(heatmap: np.ndarray, center: np.ndarray, radius: int, k: float = 1.0) -> None:
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _interpolate_polygons(polygons: list[list[tuple]], img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fill each polygon's interior with the linear interpolation of its per-corner value."""
    final_image = np.zeros(img_size, dtype=np.float32)
    mask = np.zeros(img_size, dtype=bool)
    areas = [_polygon_area([(x, y) for x, y, _ in poly]) for poly in polygons]
    for _, poly in sorted(zip(areas, polygons), key=lambda t: t[0]):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x_min, y_min = math.floor(min(xs)), math.floor(min(ys))
        x_max, y_max = math.ceil(max(xs)), math.ceil(max(ys))
        bw, bh = x_max - x_min + 1, y_max - y_min + 1
        if bw <= 0 or bh <= 0:
            continue
        points = np.array([(x - x_min, y - y_min) for x, y, _ in poly], dtype=np.float32)
        values = np.array([v for _, _, v in poly], dtype=np.float32)
        gx, gy = np.meshgrid(np.arange(bw), np.arange(bh))
        grid_points = np.vstack((gx.ravel(), gy.ravel())).T
        try:
            interp = griddata(points, values, grid_points, method="linear", fill_value=-1)
        except Exception:
            continue
        for (gi, gj), value in zip(grid_points, interp):
            i, j = gi + x_min, gj + y_min
            if value >= 0 and 0 <= j < img_size[0] and 0 <= i < img_size[1] and not mask[j, i]:
                final_image[j, i] = value
                mask[j, i] = True
    return final_image, mask


def _interpolate_logical_map(cells_logic_coords: list, output_size: tuple[int, int]):
    """Build the dense (2, H, W) logical-coordinate map (column map, row map) and its mask."""
    if not cells_logic_coords:
        return np.zeros((2, *output_size), np.float32), np.zeros((2, *output_size), np.float32)
    cols = [[(x, y, col) for (x, y, col, _row) in cell] for cell in cells_logic_coords]
    rows = [[(x, y, row) for (x, y, _col, row) in cell] for cell in cells_logic_coords]
    col_img, col_mask = _interpolate_polygons(cols, output_size)
    row_img, row_mask = _interpolate_polygons(rows, output_size)
    lc = np.stack([col_img, row_img], axis=0)
    lc_mask = np.stack([col_mask, row_mask], axis=0).astype(np.float32)
    return lc, lc_mask


def _build_table_target(
    cells: np.ndarray,
    logic: np.ndarray,
    output_size: tuple[int, int],
    max_objects: int = 300,
    max_corners: int = 1200,
) -> dict[str, np.ndarray]:
    """Render the dense TableCenterNet targets (for a single image) consumed by `TableCenterNet.compute_loss`.

    Args:
        cells: (N, 4, 2) cell quadrilaterals (corner order TL, TR, BR, BL) in **output-grid** coordinates
        logic: (N, 4) integer logical coordinates `[start_col, end_col, start_row, end_row]` (0-indexed)
        output_size: (H, W) of the model output grid (input size // down_ratio)
        max_objects: maximum number of cells
        max_corners: maximum number of distinct corners

    Returns:
        the dense target dictionary (numpy arrays) matching the reference schema
    """
    out_h, out_w = output_size
    hm = np.zeros((2, out_h, out_w), np.float32)
    reg = np.zeros((max_objects * 5, 2), np.float32)
    ct2cn = np.zeros((max_objects, 8), np.float32)
    cn2ct = np.zeros((max_corners, 8), np.float32)
    reg_ind = np.zeros((max_objects * 5,), np.int64)
    reg_mask = np.zeros((max_objects * 5,), np.float32)
    ct_ind = np.zeros((max_objects,), np.int64)
    ct_mask = np.zeros((max_objects,), np.float32)
    cn_ind = np.zeros((max_corners,), np.int64)
    cn_mask = np.zeros((max_corners,), np.float32)
    ct_cn_ind = np.zeros((max_objects * 4,), np.int64)
    lc_ind = np.zeros((max_objects, 4), np.int64)
    lc_span = np.zeros((max_objects, 2), np.float32)

    corner_dict: dict[str, int] = {}
    cells_logic_coords: list = []

    for i in range(min(len(cells), max_objects)):
        corners = cells[i].reshape(8).astype(np.float32).copy()
        corners[0::2] = np.clip(corners[0::2], 0, out_w - 1)
        corners[1::2] = np.clip(corners[1::2], 0, out_h - 1)
        if len(set(corners[0::2].tolist())) < 2 or len(set(corners[1::2].tolist())) < 2:
            continue  # not an effective quad
        xs, ys = corners[0::2], corners[1::2]
        max_x, min_x, max_y, min_y = xs.max(), xs.min(), ys.max(), ys.min()
        bbox_h, bbox_w = max_y - min_y, max_x - min_x
        if bbox_h <= 0 or bbox_w <= 0:
            continue

        radius = max(0, int(_gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))))
        center = np.array([(max_x + min_x) / 2.0, (max_y + min_y) / 2.0], np.float32)
        ci = center.astype(np.int32)
        flat = ci[1] * out_w + ci[0]
        reg[i] = center - ci
        reg_ind[i], reg_mask[i] = flat, 1
        ct_ind[i], ct_mask[i] = flat, 1
        _draw_umich_gaussian(hm[0], ci, radius)
        ct2cn[i] = center[[0, 1, 0, 1, 0, 1, 0, 1]] - corners

        start_col, end_col, start_row, end_row = (int(v) + 1 for v in logic[i])
        x1, y1, x2, y2, x3, y3, x4, y4 = corners.tolist()
        clc = [
            (x1, y1, start_col, start_row),
            (x2, y2, end_col + 1, start_row),
            (x3, y3, end_col + 1, end_row + 1),
            (x4, y4, start_col, end_row + 1),
        ]
        cells_logic_coords.append(clc)
        for j, (x, y, _c, _r) in enumerate(clc):
            lc_ind[i, j] = int(y) * out_w + int(x)
        lc_span[i] = (end_col - start_col + 1, end_row - start_row + 1)

        for j in range(4):
            si = j * 2
            corner = corners[si : si + 2]
            cint = corner.astype(np.int32)
            key = f"{cint[0]}_{cint[1]}"
            if key not in corner_dict:
                nc = len(corner_dict)
                if nc >= max_corners:
                    break
                corner_dict[key] = nc
                reg[max_objects + nc] = np.abs(corner - cint)
                reg_ind[max_objects + nc] = cint[1] * out_w + cint[0]
                reg_mask[max_objects + nc] = 1
                cn_ind[nc] = cint[1] * out_w + cint[0]
                cn_mask[nc] = 1
                _draw_umich_gaussian(hm[1], cint, 2)
                cn2ct[nc][si : si + 2] = corner - center
                ct_cn_ind[4 * i + j] = nc * 4 + j
            else:
                idx = corner_dict[key]
                cn2ct[idx][si : si + 2] = corner - center
                ct_cn_ind[4 * i + j] = idx * 4 + j

    lc, lc_mask = _interpolate_logical_map(cells_logic_coords, (out_h, out_w))
    return {
        "hm": hm,
        "reg": reg,
        "reg_ind": reg_ind,
        "reg_mask": reg_mask,
        "ct_ind": ct_ind,
        "ct_mask": ct_mask,
        "cn_ind": cn_ind,
        "cn_mask": cn_mask,
        "ct2cn": ct2cn,
        "cn2ct": cn2ct,
        "ct_cn_ind": ct_cn_ind,
        "lc": lc,
        "lc_mask": lc_mask,
        "lc_ind": lc_ind,
        "lc_span": lc_span,
    }


def _cells_to_polygons(cells: np.ndarray) -> np.ndarray:
    """Convert table cells to quadrilaterals.

    Args:
        cells: relative axis-aligned boxes of shape `(N, 4)` in `(xmin, ymin, xmax, ymax)` format,
            or quadrilaterals of shape `(N, 4, 2)`.

    Returns:
        Relative quadrilaterals of shape `(N, 4, 2)` in TL, TR, BR, BL order.
    """
    if cells.ndim == 3 and cells.shape[1:] == (4, 2):
        return cells
    if cells.ndim == 2 and cells.shape[1:] == (4,):
        xmin, ymin, xmax, ymax = cells.T
        return np.stack(
            [
                np.stack([xmin, ymin], axis=-1),
                np.stack([xmax, ymin], axis=-1),
                np.stack([xmax, ymax], axis=-1),
                np.stack([xmin, ymax], axis=-1),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    raise ValueError(f"cells are expected to have shape (N, 4) or (N, 4, 2), got {cells.shape}")


class _TableCenterNet(BaseModel):
    """TableCenterNet for table-structure recognition, as described in the official implementation
    `<https://github.com/dreamy-xay/TableCenterNet>`_.

    This base class holds the framework-agnostic target rendering (`build_target`): the dense maps consumed
    by `compute_loss` are produced here, while `TableCenterNetPostProcessor` decodes the model output.
    """

    max_objects: int = 300
    max_corners: int = 1200
    assume_straight_pages: bool = False

    def build_target(
        self,
        target: list[dict[str, np.ndarray]],
        output_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Render the dense training targets for a batch from per-image cell annotations.

        Args:
            target: one `{"cells": (N, 4) relative boxes or (N, 4, 2) relative polygons,
                "logic": (N, 4)}` dict per image
            output_shape: (H, W) of the model output grid (input size // down_ratio)

        Returns:
            the batched dense target dictionary (numpy arrays) matching the reference schema
        """
        cells_per_image: list[np.ndarray] = []
        for t in target:
            cells = np.asarray(t["cells"])
            if cells.dtype != np.float32:
                raise AssertionError("the expected dtype of target 'cells' entry is 'np.float32'.")
            cells = _cells_to_polygons(cells)
            if np.any((cells > 1) | (cells < 0)):
                raise ValueError("the 'cells' entry of the target is expected to take values between 0 & 1.")
            cells_per_image.append(cells)

        out_h, out_w = output_shape
        scale = np.array([out_w, out_h], dtype=np.float32)
        per_image = [
            _build_table_target(
                cells * scale,
                np.asarray(t["logic"], dtype=np.int64).reshape(-1, 4),
                (out_h, out_w),
                self.max_objects,
                self.max_corners,
            )
            for t, cells in zip(target, cells_per_image)
        ]
        return {k: np.stack([img[k] for img in per_image], axis=0) for k in per_image[0]}
