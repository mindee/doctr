# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from doctr.utils.geometry import estimate_page_angle, order_points
from doctr.utils.repr import NestedObject

__all__ = [
    "ReadingOrderPredictor",
    "assign_layout_labels",
    "deskew_reading_geometries",
    "detect_text_direction",
    "layout_label_role",
    "normalize_layout_label",
    "resolve_reading_segments",
    "sort_reading_order",
]

# Directions supported by the ordering functions
SUPPORTED_DIRECTIONS = ("auto", "ltr", "rtl", "ttb-rtl", "ttb-ltr")

# ISO 639 codes of languages predominantly written right-to-left (used as a fallback hint when no text is
# available, e.g. from the `langdetect` prediction attached to a page)
RTL_LANGUAGES = {"ar", "arc", "ckb", "dv", "fa", "he", "iw", "ks", "ku", "nqo", "ps", "sd", "syr", "ug", "ur", "yi"}

# Unicode bidirectional categories of strong right-to-left characters (Hebrew-like & Arabic-like)
_RTL_BIDI_CATEGORIES = {"R", "AL"}

# Role in the reading order of the normalized layout labels (DocLayNet-style); any other label is body text
_LABEL_ROLES = {
    **dict.fromkeys(("page_header", "header", "running_header"), "header"),
    **dict.fromkeys(("page_footer", "footer", "running_footer"), "footer"),
    "footnote": "footnote",
    "caption": "caption",
    **dict.fromkeys(("table", "picture", "figure", "image", "chart", "graphic"), "float"),
}
_FLOAT_LABELS = {label for label, role in _LABEL_ROLES.items() if role == "float"}
_FURNITURE_ROLES = ("header", "footer", "footnote")
# Normalized labels of the text lying in a float region without float element (e.g. a table without structure)
_FLOAT_TEXT_LABELS = {f"{label}_text" for label in _FLOAT_LABELS}

# Key-value rows (receipts, invoices, forms): a "column" of short values, each level with the text it belongs to and
# far from it, is not a text column. Below this number of column voters, such a split does not make a multi-column page
_KEY_VALUE_MAX_VOTERS = 60
# Maximum median aspect ratio (width / height) of the values: a number or a short label, not a line of running text
_KEY_VALUE_MAX_ASPECT = 8.0
# Minimum median gap between a value and the text level with it, in median element heights (a gutter is narrower)
_KEY_VALUE_MIN_GAP = 4.0


def normalize_layout_label(label: str | None) -> str:
    """Normalize a layout label to lower snake case (e.g. 'Page-header' -> 'page_header')

    Args:
        label: the layout label to normalize (None is mapped to an empty string)

    Returns:
        the normalized label
    """
    if label is None:
        return ""
    return "".join(c if c.isalnum() else "_" for c in label.strip().lower())


def layout_label_role(label: str | None) -> str:
    """Resolve the role of a layout label in the reading order.

    Args:
        label: the layout label (e.g. a DocLayNet class such as 'Page-header' or 'Caption')

    Returns:
        one of 'header', 'footer', 'footnote', 'caption', 'float' or 'body'
    """
    return _LABEL_ROLES.get(normalize_layout_label(label), "body")


def _float_text_label(label: str) -> str:
    """Return the label of a text element lying in a float region (e.g. 'Table' -> 'Table text')"""
    return f"{label} text"


def detect_text_direction(texts: Iterable[str], language: str | None = None) -> str:
    """Infer the horizontal base direction of a document from its text content.

    The detection relies on the Unicode bidirectional property of each character: strong right-to-left
    characters (bidirectional categories `R` and `AL`, covering among others Hebrew, Arabic, Syriac,
    Thaana, N'Ko) are counted against strong left-to-right characters (category `L`). This makes the
    detection language-independent and robust to mixed content. When no strong character is found, the
    optional ISO 639 language code is used as a fallback hint.

    >>> from doctr.models.reading_order import detect_text_direction
    >>> detect_text_direction(["Hello", "world"])
    'ltr'
    >>> detect_text_direction(["مرحبا", "بالعالم"])
    'rtl'

    Args:
        texts: iterable of text strings (e.g. all the words of a page)
        language: optional ISO 639 language code used as a fallback when the script is inconclusive

    Returns:
        either "ltr" or "rtl"
    """
    rtl_count = ltr_count = 0
    for text in texts:
        if not isinstance(text, str):
            continue
        for char in text:
            bidi = unicodedata.bidirectional(char)
            rtl_count += bidi in _RTL_BIDI_CATEGORIES
            ltr_count += bidi == "L"
    if rtl_count == 0 and ltr_count == 0:
        rtl_hint = isinstance(language, str) and language.split("-")[0].lower() in RTL_LANGUAGES
        return "rtl" if rtl_hint else "ltr"
    return "rtl" if rtl_count > ltr_count else "ltr"


# --- Geometry helpers -------------------------------------------------------------------------------------------------


def _geometry_to_xyxy(geometry: Any) -> tuple[float, float, float, float]:
    """Return the enclosing box (xmin, ymin, xmax, ymax) of a box or a polygon"""
    pts = np.asarray(geometry, dtype=np.float64).reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _to_boxes(geoms: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return the (N, 4) enclosing boxes of geometries in any docTR format"""
    if isinstance(geoms, np.ndarray) and geoms.ndim == 2 and geoms.shape[1] >= 4:
        return geoms[:, :4].astype(np.float64)
    if isinstance(geoms, np.ndarray) and geoms.ndim == 3:
        return np.concatenate((geoms.min(axis=1), geoms.max(axis=1)), axis=-1).astype(np.float64)
    if len(geoms) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray([_geometry_to_xyxy(geom) for geom in geoms], dtype=np.float64)


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _pairwise_overlaps(boxes: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Width and height of the intersection of each (N, 4) box with each (M, 4) box (0 when they do not overlap)"""
    inter_w = np.minimum(boxes[:, None, 2], others[None, :, 2]) - np.maximum(boxes[:, None, 0], others[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 3], others[None, :, 3]) - np.maximum(boxes[:, None, 1], others[None, :, 1])
    return np.clip(inter_w, 0, None), np.clip(inter_h, 0, None)


def _is_inside(boxes: np.ndarray, idx: int, target: int, min_coverage: float = 0.5) -> bool:
    """Check whether box `idx` lies inside box `target`, by at least `min_coverage` of its area"""
    x0, y0, x1, y1 = boxes[idx]
    tx0, ty0, tx1, ty1 = boxes[target]
    inter = max(min(x1, tx1) - max(x0, tx0), 0.0) * max(min(y1, ty1) - max(y0, ty0), 0.0)
    return inter >= min_coverage * max((x1 - x0) * (y1 - y0), 1e-9)


def _to_canonical_ltr(boxes: np.ndarray, direction: str) -> np.ndarray:
    """Map boxes to a canonical left-to-right, top-to-bottom space for the given reading direction.

    Mirroring & axis swapping are performed around the extent of the boxes themselves, so both relative and
    absolute coordinates are supported.
    """
    if direction == "ltr" or boxes.shape[0] == 0:
        return boxes
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    pivot = float(x0.min() + x1.max())
    if direction == "rtl":  # mirror horizontally
        return np.stack([pivot - x1, y0, pivot - x0, y1], axis=1)
    if direction == "ttb-ltr":  # vertical lines, columns ordered left to right: swap axes
        return np.stack([y0, x0, y1, x1], axis=1)
    if direction == "ttb-rtl":  # vertical lines, columns ordered right to left: swap axes & mirror
        return np.stack([y0, pivot - x1, y1, pivot - x0], axis=1)
    raise ValueError(f"invalid reading direction '{direction}', should be one of {SUPPORTED_DIRECTIONS[1:]}")


def _overlap_ratios(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Compute the pairwise 1D overlap of intervals, normalized by the length of the smaller interval"""
    starts = starts.astype(np.float32, copy=False)
    ends = ends.astype(np.float32, copy=False)
    lengths = ends - starts
    inter = np.minimum(ends[:, None], ends[None, :])
    inter -= np.maximum(starts[:, None], starts[None, :])
    np.clip(inter, 0, None, out=inter)
    min_len = np.minimum(lengths, lengths[:, None])
    np.clip(min_len, 1e-9, None, out=min_len)
    inter /= min_len
    return inter


def _strict_rank(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Return the rank of each element in the strict total order given by (primary, secondary, index)"""
    order = np.lexsort((np.arange(primary.shape[0]), secondary, primary))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.shape[0])
    return ranks


def deskew_reading_geometries(
    geoms: Sequence[Any] | np.ndarray,
    region_geoms: Sequence[Any] | np.ndarray | None = None,
    page_shape: tuple[int, int] | None = None,
    angle_geoms: Sequence[Any] | np.ndarray | None = None,
    min_angle: float = 1.0,
) -> tuple[list[Any], list[Any]]:
    """De-skew rotated geometries so the reading order can be computed in an upright frame.

    Args:
        geoms: geometries of the elements to order, in any docTR format (cf. `sort_reading_order`)
        region_geoms: optional geometries of the layout regions, de-skewed together with the elements
        page_shape: the page dimensions (height, width). Required for an exact angle on non-square pages with
            relative coordinates, since relative coordinates distort angles by the aspect ratio.
        angle_geoms: optional reading-oriented 4-point polygons used to estimate the page angle
        min_angle: minimum estimated angle (in degrees) to trigger the de-skew. Ordering is affected as soon
            as the drift along a line approaches the line height (about 1 degree for a page-wide line), and a
            small rigid rotation cannot change the order of an upright page, hence the low default. Beyond 45
            degrees the corner identification is ambiguous (an upstream orientation correction is needed) and
            nothing is done.

    Returns:
        the (possibly de-skewed) element and region geometries
    """
    region_geoms = list(region_geoms) if region_geoms is not None else []
    pts = [np.asarray(geom, dtype=np.float64).reshape(-1, 2) for geom in geoms]
    if len(pts) == 0 or any(p.shape[0] != 4 for p in pts):
        return list(geoms), region_geoms  # straight geometries: nothing to de-skew
    height, width = page_shape if page_shape is not None else (1, 1)
    scale = np.array([width, height], dtype=np.float64)
    angle_pts = [
        np.asarray(geom, dtype=np.float64).reshape(-1, 2) for geom in (angle_geoms if angle_geoms is not None else [])
    ]
    if len(angle_pts) > 0 and all(p.shape[0] == 4 for p in angle_pts):
        # Detection polygons are already reading-oriented (cf. `estimate_page_angle`): keep their vertex order
        angle = estimate_page_angle(np.stack(angle_pts) * scale)
    else:
        # Normalize the vertex order (TL, TR, BR, BL) so the estimation does not depend on the vertex
        angle = estimate_page_angle(np.stack([order_points(p * scale) for p in pts]))
    if not np.isfinite(angle) or abs(angle) < min_angle or abs(angle) >= 45:
        return list(geoms), region_geoms
    # Rigid rotation zeroing the estimated angle, in pixels; the center is irrelevant for ordering purposes
    theta = np.deg2rad(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = (np.concatenate(pts, axis=0) * scale).mean(axis=0)

    def _rotate(points: np.ndarray) -> np.ndarray:
        if points.shape[0] != 4:  # a straight region: rotate its 4 corners, not only its diagonal
            (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
            points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return ((points * scale - center) @ rot.T + center) / scale

    regions = [_rotate(np.asarray(geom, dtype=np.float64).reshape(-1, 2)) for geom in region_geoms]
    return [_rotate(p) for p in pts], regions


# --- Layout regions ---------------------------------------------------------------------------------------------------


def _covering_regions(
    boxes: np.ndarray,
    regions: np.ndarray,
    min_coverage: float,
    region_labels: Sequence[str | None] | None = None,
) -> list[int]:
    """Return the index of the region covering each (N, 4) box by at least `min_coverage`, -1 if none

    Regions covering a box (nearly) as much as the best one tie: the smallest one wins, unless it is a float tied with
    page furniture.
    """
    inter_w, inter_h = _pairwise_overlaps(boxes, regions)
    coverage = inter_w * inter_h / np.clip(_box_areas(boxes), 1e-9, None)[:, None]
    region_areas = _box_areas(regions)
    qualifies = coverage >= np.maximum(coverage.max(axis=1, keepdims=True) - 0.1, min_coverage)
    # Regions nested in the best region of a box also tie when they cover most of it
    reg_w, reg_h = _pairwise_overlaps(regions, regions)
    nested = reg_w * reg_h >= 0.9 * np.clip(region_areas, 1e-9, None)[:, None]  # nested[i, j]: region i lies inside j
    qualifies |= (coverage >= max(min_coverage, 0.75)) & nested[:, coverage.argmax(axis=1)].T
    ranked = np.where(qualifies, region_areas[None, :], np.inf)
    best = ranked.argmin(axis=1)
    result = [int(reg) if np.isfinite(ranked[idx, reg]) else -1 for idx, reg in enumerate(best)]
    if region_labels is None or len(region_labels) != regions.shape[0]:
        return result

    roles = [layout_label_role(label) for label in region_labels]
    is_float = np.array([role == "float" for role in roles], dtype=bool)
    is_furniture = np.array([role in _FURNITURE_ROLES for role in roles], dtype=bool)
    if not (is_float.any() and is_furniture.any()):
        return result
    # The text of a float tied with page furniture (e.g. a logo in the page header) is read with the furniture
    furniture_ranked = np.where(qualifies & is_furniture[None, :], region_areas[None, :], np.inf)
    for idx, reg in enumerate(result):
        if reg >= 0 and is_float[reg] and np.isfinite(furniture_ranked[idx].min()):
            result[idx] = int(furniture_ranked[idx].argmin())
    return result


def _region_groups(regions: np.ndarray, labels: Sequence[str | None], min_overlap: float = 0.1) -> list[int]:
    """Return the group of each (R, 4) layout region (the index of its first member), grouping the regions of the same
    label which overlap by more than `min_overlap` of the smaller one (e.g. two regions for a paragraph around a figure)
    """
    if regions.shape[0] == 0:
        return []
    norm = np.asarray([normalize_layout_label(label) for label in labels])
    inter_w, inter_h = _pairwise_overlaps(regions, regions)
    areas = _box_areas(regions)
    linked = (inter_w * inter_h > min_overlap * np.minimum(areas[:, None], areas[None, :])) & (
        norm[:, None] == norm[None, :]
    )
    component = connected_components(csr_matrix(linked), directed=False)[1]
    first: dict[int, int] = {}
    return [first.setdefault(int(comp), idx) for idx, comp in enumerate(component)]


def assign_layout_labels(
    geoms: Sequence[Any] | np.ndarray,
    layout_geoms: Sequence[Any] | np.ndarray,
    layout_labels: Sequence[str],
    min_coverage: float = 0.5,
    page_shape: tuple[int, int] | None = None,
    angle_geoms: Sequence[Any] | np.ndarray | None = None,
) -> list[str | None]:
    """Assign a layout label to each element based on its overlap with the detected layout regions.

    Each element receives the label of the region covering the largest share of its area, provided this share reaches
    `min_coverage`; otherwise its label is None (treated as regular body content). Among nested regions, the smallest
    one wins (e.g. 'Caption' over 'Picture'), except for a float nested in page furniture (e.g. a logo in the page
    header): its text takes the furniture label.

    Args:
        geoms: geometries of the elements to label, in any docTR format
        layout_geoms: geometries of the layout regions, in any docTR format
        layout_labels: labels of the layout regions (e.g. `[region.type for region in page.layout]`)
        min_coverage: minimum share of an element's area a region must cover to assign its label
        page_shape: the page dimensions (height, width), used to de-skew rotated pages exactly (cf.
            `deskew_reading_geometries`)
        angle_geoms: optional reading-oriented 4-point polygons (typically the page's word polygons) used to
            estimate the page angle on rotated pages (cf. `deskew_reading_geometries`)

    Returns:
        the label of each element (None when no region covers it enough)
    """
    # De-skew elements and regions together so the coverage is measured on tight boxes in the same frame
    geoms, layout_geoms = deskew_reading_geometries(geoms, layout_geoms, page_shape=page_shape, angle_geoms=angle_geoms)
    boxes, regions = _to_boxes(geoms), _to_boxes(layout_geoms)
    if len(layout_labels) != regions.shape[0]:
        raise ValueError(f"Incompatible number of labels ({len(layout_labels)}) and regions ({regions.shape[0]})")
    if boxes.shape[0] == 0 or regions.shape[0] == 0:
        return [None] * boxes.shape[0]
    covering = _covering_regions(boxes, regions, min_coverage, layout_labels)
    return [str(layout_labels[reg]) if reg >= 0 else None for reg in covering]


# --- Topological sort -------------------------------------------------------------------------------------------------


def _key_value_rows(
    boxes: np.ndarray, voters: np.ndarray, split: float, y_overlap: np.ndarray, y_overlap_threshold: float, span: float
) -> bool:
    """Check whether the voters on the right of a column split are the values of key-value rows rather than a column

    On a receipt or an invoice, the amounts form a narrow right-aligned "column": each amount is level with the item
    it belongs to, far from it, and the rows must be read one after the other. In a text column, the lines are long
    and the gutter separating them from the lines of the previous column is narrow.

    Args:
        boxes: the (N, 4) boxes in canonical LTR space
        voters: the mask of the boxes used to detect the columns
        split: the abscissa of the column split
        y_overlap: the (N, N) vertical overlap ratios of the boxes
        y_overlap_threshold: the vertical overlap above which two boxes lie on the same row
        span: the horizontal extent of the voters

    Returns:
        whether the split separates the keys from the values of key-value rows
    """
    if np.count_nonzero(voters) > _KEY_VALUE_MAX_VOTERS:
        return False
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    on_left = voters & ((x0 + x1) / 2 <= split)
    left, right = np.flatnonzero(on_left), np.flatnonzero(voters & ~on_left)
    height = float(np.median(y1[voters] - y0[voters]))
    if right.size == 0 or left.size == 0 or not height > 0:
        return False
    aspect = float(np.median((x1[right] - x0[right]) / np.maximum(y1[right] - y0[right], 1e-9)))
    if aspect >= _KEY_VALUE_MAX_ASPECT:
        return False
    # Gap between each value and the text level with it on its left
    level = y_overlap[np.ix_(right, left)] > y_overlap_threshold
    level &= x1[left][None, :] <= x0[right][:, None] + 0.02 * span
    paired = level.any(axis=1)
    if np.count_nonzero(paired) < 0.5 * right.size:
        return False
    ends = np.where(level, x1[left][None, :], -np.inf).max(axis=1)
    return float(np.median((x0[right][paired] - ends[paired]) / height)) >= _KEY_VALUE_MIN_GAP


def _column_gutter(
    boxes: np.ndarray, voters: np.ndarray, y_overlap: np.ndarray, y_overlap_threshold: float
) -> np.ndarray | None:
    """Detect a multi-column page: a vertical line separating the voters into two groups crossed by few of them

    Args:
        boxes: the (N, 4) boxes in canonical LTR space
        voters: the mask of the boxes used to detect the columns (all of them with fewer than 3 voters)
        y_overlap: the (N, N) vertical overlap ratios of the boxes
        y_overlap_threshold: the vertical overlap above which two boxes lie on the same row

    Returns:
        the mask of the boxes crossing the gutter for a multi-column page, None for a single-column page
    """
    if np.count_nonzero(voters) < 3:
        voters = np.ones(boxes.shape[0], dtype=bool)
    x0, x1 = boxes[:, 0], boxes[:, 2]
    vx0, vx1, num_voters = x0[voters], x1[voters], int(np.count_nonzero(voters))
    if num_voters < 3:
        return None
    # Measured on the voters only, so that a wide figure does not shift the search window
    left_edge = float(vx0.min())
    span = float(vx1.max()) - left_edge or 1.0
    tolerance = max(1, int(0.05 * num_voters))
    centers = (vx0 + vx1) / 2
    lo, hi = left_edge + 0.25 * span, left_edge + 0.75 * span
    for split in np.unique(vx1[(vx1 >= lo) & (vx1 <= hi)]):
        crossing = int(np.count_nonzero(np.minimum(vx1 - split, split - vx0) > 0.02 * span))
        left = int(np.count_nonzero(centers <= split))
        if crossing <= tolerance and left >= 0.25 * num_voters and num_voters - left >= 0.25 * num_voters:
            if _key_value_rows(boxes, voters, float(split), y_overlap, y_overlap_threshold, span):
                return None  # rows of keys and values (e.g. a receipt): read row after row
            return np.minimum(x1 - split, split - x0) > 0.02 * span
    return None


def _topological_order(
    boxes: np.ndarray, x_overlap_threshold: float, y_overlap_threshold: float, voters: np.ndarray | None = None
) -> list[int]:
    """Order boxes (already in canonical LTR space) with a column-following topological sort.

    Two families of "reads-before" relations are used (cf. Breuel 2003):

    * `i` precedes `j` when they overlap horizontally and `i` lies above `j`
    * `i` precedes `j` when they lie on the same visual row (vertical overlap, no horizontal overlap)
      and `i` is on the left of `j`

    The relations are resolved with Kahn's algorithm; among the available elements, the traversal favors the
    continuation of the current column (the closest element below the last emitted one with a horizontal
    overlap), which keeps multi-column bodies intact even for non-Manhattan layouts where recursive XY-cuts
    fail to find a valid split. `voters` optionally restricts the boxes used to detect a multi-column page. A split
    separating the values of key-value rows from their keys (e.g. the amounts of a receipt, cf. `_key_value_rows`)
    does not make the page multi-column.
    """
    num_boxes = boxes.shape[0]
    if num_boxes <= 1:
        return list(range(num_boxes))

    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_overlap = _overlap_ratios(x0, x1)
    y_overlap = _overlap_ratios(y0, y1)
    # Strict total orders on both axes, so that the induced relations cannot create 2-cycles
    x_rank, y_rank = _strict_rank(x0, x1), _strict_rank(y0, y1)

    # edges[i, j]: i must be read before j. The N x N matrices are combined in place, since materializing them all at
    # once dominates the memory of this function on dense pages.
    x_linked = x_overlap > x_overlap_threshold
    edges = y_rank[:, None] < y_rank[None, :]  # i is above j...
    edges &= x_linked  # ... in the same column
    same_row = y_overlap > y_overlap_threshold
    same_row &= ~x_linked
    same_row &= x_rank[:, None] < x_rank[None, :]  # i is on the left of j on the same row
    # An element shifted between two rows (e.g. an amount on a crumpled receipt) only waits for the row it overlaps most
    best_left = np.max(y_overlap, axis=0, where=same_row, initial=0.0)
    weak_i, weak_j = np.nonzero(same_row & (y_overlap < best_left[None, :] - 0.1))
    if weak_i.size > 0:
        cols = np.unique(weak_j)
        best = np.argmax(np.where(same_row[:, cols], y_overlap[:, cols], -1.0), axis=0)
        other_row = y_overlap[weak_i, best[np.searchsorted(cols, weak_j)]] <= y_overlap_threshold
        same_row[weak_i[other_row], weak_j[other_row]] = False
    edges |= same_row
    del same_row
    np.fill_diagonal(edges, False)

    # A multi-column page favors the continuation of the current column (cf. `_continuation`)
    spanning = (x1 - x0) > 0.5 * (float(x1.max() - x0.min()) or 1.0)
    crossing = _column_gutter(
        boxes, np.ones(num_boxes, dtype=bool) if voters is None else voters, y_overlap, y_overlap_threshold
    )
    multi_column = crossing is not None
    # Column components: elements connected through horizontal overlap, except those which must not merge the columns
    # (page-wide or crossing the gutter). This keeps the traversal inside a column even when the direct top-to-bottom
    # continuation is momentarily broken by a fragmented or misaligned OCR line.
    bridging = spanning | crossing if multi_column else spanning
    col_edges = np.triu(x_linked, 1)  # each pair once
    col_edges &= ~bridging[:, None]
    col_edges &= ~bridging[None, :]
    component = connected_components(csr_matrix(col_edges), directed=False, return_labels=True)[1]

    def _continuation(last: int, ready: np.ndarray) -> np.ndarray:
        """The available elements continuing the reading after `last`: below it in its column (multi-column page),
        else on its row, else in its column component"""
        if multi_column:
            below = ready[(x_overlap[last, ready] > x_overlap_threshold) & (y0[ready] >= y0[last])]
            if below.size > 0:
                return below
        level = ready[y_overlap[last, ready] > y_overlap_threshold]
        if level.size > 0 or spanning[last]:
            return level if level.size > 0 else ready
        same_column = ready[component[ready] == component[last]]
        return same_column if same_column.size > 0 else ready

    in_degree = edges.sum(axis=0)
    emitted = np.zeros(num_boxes, dtype=bool)
    order: list[int] = []
    while len(order) < num_boxes:
        ready = np.flatnonzero((in_degree == 0) & ~emitted)
        if ready.size == 0:  # cycle safety net (can only happen with degenerate overlapping geometries)
            candidates = np.flatnonzero(~emitted)
        else:
            candidates = _continuation(order[-1], ready) if order else ready
        # Topmost first, then leftmost (in canonical space)
        next_idx = int(candidates[np.lexsort((x0[candidates], y0[candidates]))[0]])
        order.append(next_idx)
        emitted[next_idx] = True
        in_degree = in_degree - edges[next_idx]
    return order


# --- Floats & captions ------------------------------------------------------------------------------------------------


def _outer_floats(idcs: list[int], boxes: np.ndarray, labels: list[str]) -> list[int]:
    """Return the floats among `idcs` that are not nested in a larger float"""
    areas = _box_areas(boxes)
    floats = [idx for idx in idcs if labels[idx] in _FLOAT_LABELS]
    return [
        idx
        for idx in floats
        if not any(areas[other] > areas[idx] and _is_inside(boxes, idx, other) for other in floats)
    ]


def _float_members(idcs: list[int], boxes: np.ndarray, labels: list[str]) -> dict[int, list[int]]:
    """Map each outer float to the elements inside it (e.g. the text of a figure)

    An element belongs to the smallest outer float covering at least half of it.
    """
    areas = _box_areas(boxes)
    outer = _outer_floats(idcs, boxes, labels)
    members: dict[int, list[int]] = {}
    for idx in idcs:
        if idx in outer:
            continue
        containers = [flt for flt in outer if areas[flt] > areas[idx] and _is_inside(boxes, idx, flt)]
        if containers:
            members.setdefault(min(containers, key=lambda flt: areas[flt]), []).append(idx)
    return members


def _stands_beside(boxes: np.ndarray, run: list[int], region: list[int]) -> bool:
    """Check whether an element of a region stands beside a run of floats (level with it, not overlapping it)

    Args:
        boxes: the (N, 4) boxes of all elements
        run: the floats of the run, with the elements read inside them
        region: the other elements of the interrupted region

    Returns:
        True if an element of the region stands beside the run
    """
    rx0, ry0 = boxes[run, 0].min(), boxes[run, 1].min()
    rx1, ry1 = boxes[run, 2].max(), boxes[run, 3].max()
    for x0, y0, x1, y1 in boxes[region]:
        level = min(y1, ry1) - max(y0, ry0) >= 0.5 * max(y1 - y0, 1e-9)  # at least half of the element height
        beside = min(x1, rx1) - max(x0, rx0) <= 0.1 * max(min(x1 - x0, rx1 - rx0), 1e-9)  # at most 10% overlap
        if level and beside:
            return True
    return False


def _defer_floats(
    order: list[int],
    labels: list[str],
    members: dict[int, list[int]],
    groups: Sequence[int],
    boxes: np.ndarray | None = None,
) -> list[int]:
    """Move the floats interrupting a layout region right after the end of that region

    The traversal can read a figure standing beside a paragraph before the last lines of that paragraph.

    Args:
        order: the body elements in reading order, the elements inside a float right after it
        labels: the normalized layout label of each element
        members: the elements inside each float
        groups: the layout region index of each element (-1 for none)
        boxes: the (N, 4) boxes of all elements, to only move the floats standing beside the region

    Returns:
        the reordered body elements
    """
    inner = {idx for idcs in members.values() for idx in idcs}

    def _is_float(idx: int) -> bool:
        return labels[idx] in _FLOAT_LABELS or labels[idx] in _FLOAT_TEXT_LABELS or idx in inner

    out = list(order)
    pos = 0
    while pos < len(out):
        if not _is_float(out[pos]):
            pos += 1
            continue
        # out[pos:end] is a run of floats, read after out[pos - 1]: does it interrupt the region of that element?
        end = pos
        while end < len(out) and _is_float(out[end]):
            end += 1
        group = int(groups[out[pos - 1]]) if pos > 0 else -1
        interrupts = group >= 0 and end < len(out) and int(groups[out[end]]) == group
        if interrupts and boxes is not None:
            region = [idx for idx in out if int(groups[idx]) == group and not _is_float(idx)]
            interrupts = _stands_beside(boxes, out[pos:end], region)
        if not interrupts:
            pos = end
            continue
        # Move the run after the rest of the region
        stop = end
        while stop < len(out) and not _is_float(out[stop]) and int(groups[out[stop]]) == group:
            stop += 1
        out[pos:stop] = out[end:stop] + out[pos:end]
        pos = stop - (end - pos)  # check the moved run again: it can interrupt the same region further down
    return out


def _caption_distance(caption: Sequence[float], target: Sequence[float]) -> float:
    """Distance from a caption box to a float box, penalizing horizontal shifts"""
    x_gap = max(target[0] - caption[2], caption[0] - target[2], 0.0)
    y_gap = max(target[1] - caption[3], caption[1] - target[3], 0.0)
    return y_gap + 2 * x_gap


def _caption_units(caption_idcs: list[int], region_groups: Sequence[int] | None) -> list[list[int]]:
    """Group the captions sharing a group id into units, in reading order (a caption without group is a unit)"""
    units: list[list[int]] = []
    by_group: dict[int, list[int]] = {}
    for cap in caption_idcs:
        group = -1 if region_groups is None else int(region_groups[cap])
        if group < 0:
            units.append([cap])
        elif group in by_group:
            by_group[group].append(cap)
        else:
            by_group[group] = [cap]
            units.append(by_group[group])
    return units


def _attach_captions(
    order: list[int],
    caption_idcs: list[int],
    boxes: np.ndarray,
    labels: list[str],
    max_distance: float,
    region_groups: Sequence[int] | None = None,
) -> list[int]:
    """Insert the captions right before (resp. after) the closest float they sit above (resp. below)

    The captions of a group (e.g. the lines of a caption region) move as a unit. A caption without a float within
    reach is read in its column, right after the closest element above it.
    """
    captions = set(caption_idcs)
    float_idcs = _outer_floats(order, boxes, labels)
    below: dict[int, set[int]] = {}  # captions already read after each element

    def _read_after(target: int, unit: list[int]) -> int:
        """Return the position after `target`, the elements inside it and the captions already read after it"""
        attached = below.setdefault(target, set())
        pos = order.index(target) + 1
        while pos < len(order) and (
            order[pos] in attached or (order[pos] not in captions and _is_inside(boxes, order[pos], target))
        ):
            pos += 1
        attached.update(unit)
        return pos

    for unit in _caption_units(caption_idcs, region_groups):
        cx0, cy0 = boxes[unit, 0].min(), boxes[unit, 1].min()
        cx1, cy1 = boxes[unit, 2].max(), boxes[unit, 3].max()
        target, best_dist = -1, float("inf")
        for flt in float_idcs:
            dist = _caption_distance((cx0, cy0, cx1, cy1), boxes[flt])
            if dist < best_dist:
                target, best_dist = flt, dist
        if target >= 0 and best_dist <= max_distance:
            # Read right before the float when above its center, otherwise after it
            above = (cy0 + cy1) / 2 <= (boxes[target, 1] + boxes[target, 3]) / 2
            pos = order.index(target) if above else _read_after(target, unit)
        else:
            in_column = [idx for idx in order if min(boxes[idx, 2], cx1) - max(boxes[idx, 0], cx0) > 0]
            above_idcs = [idx for idx in in_column if (boxes[idx, 1] + boxes[idx, 3]) / 2 < cy0]
            if above_idcs:
                pos = _read_after(max(above_idcs, key=lambda idx: boxes[idx, 3]), unit)
            else:  # at the top of its column
                fallback = next((i for i, idx in enumerate(order) if boxes[idx, 1] >= cy0), len(order))
                pos = min((order.index(idx) for idx in in_column), default=fallback)
        order[pos:pos] = unit
    return order


# --- Public ordering functions ----------------------------------------------------------------------------------------


def _canonical_boxes(
    geoms: Sequence[Any] | np.ndarray,
    direction: str,
    labels: Sequence[str | None] | None,
    region_groups: Sequence[int] | None,
    page_shape: tuple[int, int] | None,
    angle_geoms: Sequence[Any] | np.ndarray | None,
) -> np.ndarray:
    """Check the inputs of the ordering functions and return the de-skewed boxes in canonical LTR space"""
    if direction not in SUPPORTED_DIRECTIONS[1:]:
        raise ValueError(f"invalid reading direction '{direction}', should be one of {SUPPORTED_DIRECTIONS[1:]}")
    geoms, _ = deskew_reading_geometries(geoms, page_shape=page_shape, angle_geoms=angle_geoms)
    boxes = _to_boxes(geoms)
    if labels is not None and len(labels) != boxes.shape[0]:
        raise ValueError(f"Incompatible number of labels ({len(labels)}) and geometries ({boxes.shape[0]})")
    if region_groups is not None and len(region_groups) != boxes.shape[0]:
        raise ValueError(
            f"Incompatible number of region groups ({len(region_groups)}) and geometries ({boxes.shape[0]})"
        )
    return _to_canonical_ltr(boxes, direction)


def _sort(
    canonical: np.ndarray,
    labels: Sequence[str | None] | None,
    x_overlap_threshold: float,
    y_overlap_threshold: float,
    caption_max_distance: float,
    column_voters: Sequence[bool] | None,
    region_groups: Sequence[int] | None,
) -> list[int]:
    """Order boxes in canonical LTR space (cf. `sort_reading_order`)"""
    num_boxes = canonical.shape[0]
    if num_boxes <= 1:
        return list(range(num_boxes))
    voters = np.ones(num_boxes, dtype=bool) if column_voters is None else np.asarray(column_voters, dtype=bool)

    def _order(idcs: list[int]) -> list[int]:
        sub_order = _topological_order(canonical[idcs], x_overlap_threshold, y_overlap_threshold, voters[idcs])
        return [idcs[i] for i in sub_order]

    if labels is None:
        return _order(list(range(num_boxes)))

    norm_labels = [normalize_layout_label(label) for label in labels]
    groups: dict[str, list[int]] = {"header": [], "body": [], "caption": [], "footnote": [], "footer": []}
    for idx, label in enumerate(norm_labels):
        role = layout_label_role(label)
        groups["body" if role == "float" else role].append(idx)

    # Read the elements inside a float (e.g. the text of a figure) right after it
    members = _float_members(groups["body"], canonical, norm_labels)
    owner = {idx: flt for flt, idcs in members.items() for idx in idcs}
    body_order = _order(groups["body"])
    inside = {flt: [idx for idx in body_order if owner.get(idx) == flt] for flt in members}
    body_order = [elt for idx in body_order if idx not in owner for elt in (idx, *inside.get(idx, []))]
    if region_groups is not None:
        # A float standing beside a region must not interrupt it
        body_order = _defer_floats(body_order, norm_labels, members, region_groups, canonical)
    body_order = _attach_captions(
        body_order, _order(groups["caption"]), canonical, norm_labels, caption_max_distance, region_groups
    )
    return _order(groups["header"]) + body_order + _order(groups["footnote"]) + _order(groups["footer"])


def sort_reading_order(
    geoms: Sequence[Any] | np.ndarray,
    direction: str = "ltr",
    labels: Sequence[str | None] | None = None,
    x_overlap_threshold: float = 0.2,
    y_overlap_threshold: float = 0.5,
    caption_max_distance: float = 0.1,
    page_shape: tuple[int, int] | None = None,
    angle_geoms: Sequence[Any] | np.ndarray | None = None,
    column_voters: Sequence[bool] | None = None,
    region_groups: Sequence[int] | None = None,
) -> list[int]:
    """Compute the reading order of document elements from their geometries (and optionally, layout labels).

    >>> from doctr.models.reading_order import sort_reading_order
    >>> # A title spanning two columns of text
    >>> geoms = [
    ...     ((0.55, 0.2), (0.9, 0.8)),  # right column
    ...     ((0.1, 0.05), (0.9, 0.15)),  # title
    ...     ((0.1, 0.2), (0.45, 0.8)),  # left column
    ... ]
    >>> sort_reading_order(geoms)
    [1, 2, 0]

    Args:
        geoms: sequence of geometries in any docTR format (((xmin, ymin), (xmax, ymax)) boxes, 4-point
            polygons, or a (N, 4) / (N, 4, 2) numpy array), in relative or absolute coordinates
        direction: reading direction, one of 'ltr' (Latin, Cyrillic, CJK horizontal, ...), 'rtl' (Arabic,
            Hebrew, ...), 'ttb-rtl' (vertical lines with columns ordered right-to-left, e.g. traditional
            Chinese/Japanese) or 'ttb-ltr' (vertical lines with columns ordered left-to-right)
        labels: optional layout labels (one per geometry, e.g. DocLayNet classes from a docTR layout
            predictor). When provided, page headers are moved first, footnotes and page footers last, and
            captions are attached to the closest figure or table. Unknown or None labels are treated as body.
        x_overlap_threshold: minimum relative horizontal overlap for two elements to be considered vertically
            stacked (same column)
        y_overlap_threshold: minimum relative vertical overlap for two non-stacked elements to be considered
            on the same visual row
        caption_max_distance: maximum relative distance between a caption and a float (table or figure) for
            the caption to be attached to it
        page_shape: the page dimensions (height, width), used to de-skew rotated pages exactly (cf.
            `deskew_reading_geometries`)
        angle_geoms: optional reading-oriented 4-point polygons (typically the page's word polygons) used to
            estimate the page angle on rotated pages (cf. `deskew_reading_geometries`)
        column_voters: optional mask of the elements used to detect a multi-column page (all by default)
        region_groups: optional layout region index of each element (-1 for none), used to attach the captions of
            a region as a unit and to read a float standing beside a region after it

    Returns:
        the permutation of the input indices which sorts the elements in reading order
    """
    canonical = _canonical_boxes(geoms, direction, labels, region_groups, page_shape, angle_geoms)
    return _sort(
        canonical, labels, x_overlap_threshold, y_overlap_threshold, caption_max_distance, column_voters, region_groups
    )


class _SegmentJoins:
    """Criteria to join two consecutive elements of the reading order in the same segment (cf.
    `resolve_reading_segments`), in canonical LTR space"""

    def __init__(
        self,
        boxes: np.ndarray,
        norm_labels: list[str],
        region_groups: Sequence[int] | None,
        x_overlap_threshold: float,
        y_overlap_threshold: float,
    ) -> None:
        self.boxes, self.labels, self.groups = boxes, norm_labels, region_groups
        self.roles = [layout_label_role(label) for label in norm_labels]
        self.median_height = float(np.median(boxes[:, 3] - boxes[:, 1]))
        self.x_overlap_threshold, self.y_overlap_threshold = x_overlap_threshold, y_overlap_threshold

    @staticmethod
    def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
        """1D overlap of two intervals, relative to the shorter one"""
        inter = min(end, other_end) - max(start, other_start)
        return inter / max(min(end - start, other_end - other_start), 1e-9)

    def _same_region(self, prev: int, cur: int) -> bool:
        return self.groups is None or self.groups[prev] == self.groups[cur]

    def compatible(self, prev: int, cur: int) -> bool:
        """Same label (not a float) and same layout region"""
        return self.labels[prev] == self.labels[cur] and self.roles[cur] != "float" and self._same_region(prev, cur)

    def stacked(self, row: np.ndarray, cur: int, paragraph_gap: float) -> bool:
        """Right below the last row of the segment, in the same column"""
        box = self.boxes[cur]
        return (
            self._overlap(row[0], row[2], box[0], box[2]) > self.x_overlap_threshold
            and box[1] - row[3] <= paragraph_gap * self.median_height
        )

    def same_row(self, prev: int, cur: int, row_gap: float, aspect: float) -> bool:
        """On the row of the previous element, after it: the rest of a line split at a wide word gap (at most
        `row_gap` median heights, any gap within a region of body text)"""
        box, prev_box = self.boxes[cur], self.boxes[prev]
        return (
            self._overlap(prev_box[1], prev_box[3], box[1], box[3]) > self.y_overlap_threshold
            and box[0] > prev_box[0]
            and (
                (box[0] - prev_box[2]) * aspect <= row_gap * self.median_height
                or (
                    self.groups is not None
                    and self.groups[cur] >= 0
                    and self._same_region(prev, cur)
                    and self.roles[cur] == "body"
                )
            )
        )


def resolve_reading_segments(
    geoms: Sequence[Any] | np.ndarray,
    direction: str = "ltr",
    labels: Sequence[str | None] | None = None,
    x_overlap_threshold: float = 0.2,
    y_overlap_threshold: float = 0.5,
    caption_max_distance: float = 0.1,
    paragraph_gap: float = 0.8,
    page_shape: tuple[int, int] | None = None,
    angle_geoms: Sequence[Any] | np.ndarray | None = None,
    column_voters: Sequence[bool] | None = None,
    region_groups: Sequence[int] | None = None,
    row_gap: float = 1.5,
) -> list[list[int]]:
    """Order elements in reading order and group consecutive ones into segments (paragraphs or regions).

    This is typically used to regroup individual lines into paragraphs once they have been put in reading
    order: two consecutive elements are merged into the same segment when they share the same (possibly None)
    layout label and layout region, and either belong to the same column and are vertically close enough, or
    continue the same visual row.

    Args:
        geoms: sequence of geometries in any docTR format (cf. `sort_reading_order`)
        direction: reading direction, one of 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
        labels: optional layout labels (one per geometry), cf. `sort_reading_order`. Elements with a 'float'
            role (tables, pictures) are never merged with their neighbors.
        x_overlap_threshold: minimum relative horizontal overlap for two elements to be considered vertically
            stacked (same column)
        y_overlap_threshold: minimum relative vertical overlap for two non-stacked elements to be considered
            on the same visual row
        caption_max_distance: maximum relative distance between a caption and a float (table or figure) for
            the caption to be attached to it
        paragraph_gap: maximum vertical gap between two consecutive elements to belong to the same segment,
            as a multiple of the median element height
        page_shape: the page dimensions (height, width), used to de-skew rotated pages exactly (cf.
            `deskew_reading_geometries`)
        angle_geoms: optional reading-oriented 4-point polygons (typically the page's word polygons) used to
            estimate the page angle on rotated pages (cf. `deskew_reading_geometries`)
        column_voters: optional mask of the elements used to detect a multi-column page (all by default)
        region_groups: optional layout region index of each element (-1 for none), used to attach the captions of
            a region as a unit, to read a float standing beside a region after it, and to keep distinct regions in
            distinct segments
        row_gap: maximum horizontal gap between two consecutive elements of a visual row to join them, as a multiple
            of the median element height (not applied to body text of the same layout region)

    Returns:
        a partition of the input indices into reading-ordered segments (each segment being itself in
        reading order)
    """
    # The elements are ordered on their upright geometries, in which the gaps between them are measured as well
    upright, _ = deskew_reading_geometries(geoms, page_shape=page_shape, angle_geoms=angle_geoms)
    order = sort_reading_order(
        upright,
        direction=direction,
        labels=labels,
        x_overlap_threshold=x_overlap_threshold,
        y_overlap_threshold=y_overlap_threshold,
        caption_max_distance=caption_max_distance,
        column_voters=column_voters,
        region_groups=region_groups,
    )
    if len(order) == 0:
        return []
    canonical = _to_canonical_ltr(_to_boxes(upright), direction)
    aspect = 1.0  # horizontal distances in the height unit
    if page_shape is not None and page_shape[0] > 0 and page_shape[1] > 0:
        aspect = page_shape[0] / page_shape[1] if direction.startswith("ttb") else page_shape[1] / page_shape[0]
    norm_labels = [normalize_layout_label(labels[idx] if labels is not None else None) for idx in range(len(order))]
    joins = _SegmentJoins(canonical, norm_labels, region_groups, x_overlap_threshold, y_overlap_threshold)
    segments: list[list[int]] = [[order[0]]]
    row = canonical[order[0]].copy()  # extent of the last visual row of the current segment
    for prev, cur in zip(order[:-1], order[1:]):
        stacked, same_row = joins.stacked(row, cur, paragraph_gap), joins.same_row(prev, cur, row_gap, aspect)
        joined = joins.compatible(prev, cur) and (stacked or same_row)
        if joined:
            segments[-1].append(cur)
        else:
            segments.append([cur])
        box = canonical[cur]
        row = (
            np.concatenate((np.minimum(row[:2], box[:2]), np.maximum(row[2:], box[2:])))
            if joined and same_row and not stacked
            else box.copy()
        )
    return segments


class ReadingOrderPredictor(NestedObject):
    """Language-independent reading order estimation for document elements.

    >>> import numpy as np
    >>> from doctr.models.reading_order import ReadingOrderPredictor
    >>> predictor = ReadingOrderPredictor()
    >>> geoms = np.array([[0.55, 0.2, 0.9, 0.8], [0.1, 0.05, 0.9, 0.15], [0.1, 0.2, 0.45, 0.8]])
    >>> predictor(geoms, texts=["First words of each element", "of the page", "to detect the direction"])
    [1, 2, 0]

    Args:
        direction: reading direction, one of 'auto' (inferred from the text content), 'ltr', 'rtl', 'ttb-rtl'
            or 'ttb-ltr' (cf. `sort_reading_order`). Automatic detection covers horizontal scripts only.
        x_overlap_threshold: minimum relative horizontal overlap for two elements to be considered vertically
            stacked (same column)
        y_overlap_threshold: minimum relative vertical overlap for two non-stacked elements to be considered
            on the same visual row
        caption_max_distance: maximum relative distance between a caption and a float (table or figure) for
            the caption to be attached to it
    """

    def __init__(
        self,
        direction: str = "auto",
        x_overlap_threshold: float = 0.2,
        y_overlap_threshold: float = 0.5,
        caption_max_distance: float = 0.1,
    ) -> None:
        if direction not in SUPPORTED_DIRECTIONS:
            raise ValueError(f"invalid reading direction '{direction}', should be one of {SUPPORTED_DIRECTIONS}")
        self.direction = direction
        self.x_overlap_threshold = x_overlap_threshold
        self.y_overlap_threshold = y_overlap_threshold
        self.caption_max_distance = caption_max_distance

    def resolve_direction(self, texts: Iterable[str] | None = None, language: str | None = None) -> str:
        """Resolve the effective reading direction for the given content.

        Args:
            texts: text content used for the automatic detection (ignored when the direction is explicit)
            language: optional ISO 639 language code used as a fallback hint

        Returns:
            the effective reading direction ('ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr')
        """
        if self.direction != "auto":
            return self.direction
        return detect_text_direction(texts if texts is not None else [], language=language)

    def extra_repr(self) -> str:
        return f"direction='{self.direction}'"

    def __call__(
        self,
        geoms: Sequence[Any] | np.ndarray,
        texts: Sequence[str] | None = None,
        labels: Sequence[str | None] | None = None,
        language: str | None = None,
        page_shape: tuple[int, int] | None = None,
        angle_geoms: Sequence[Any] | np.ndarray | None = None,
    ) -> list[int]:
        """Compute the reading order of document elements.

        Args:
            geoms: sequence of geometries in any docTR format (cf. `sort_reading_order`)
            texts: optional text content of each element (or any text of the page), used for the automatic
                direction detection
            labels: optional layout labels (one per geometry), used to handle page furniture & captions
            language: optional ISO 639 language code used as a fallback hint for the direction detection
            page_shape: the page dimensions (height, width), used to de-skew rotated pages exactly on
                non-square pages
            angle_geoms: optional reading-oriented 4-point polygons (typically the page's word polygons) used
                to estimate the page angle on rotated pages

        Returns:
            the permutation of the input indices which sorts the elements in reading order
        """
        return sort_reading_order(
            geoms,
            direction=self.resolve_direction(texts, language=language),
            labels=labels,
            x_overlap_threshold=self.x_overlap_threshold,
            y_overlap_threshold=self.y_overlap_threshold,
            caption_max_distance=self.caption_max_distance,
            page_shape=page_shape,
            angle_geoms=angle_geoms,
        )
