# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

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

# Normalized layout labels (DocLayNet-style) grouped by their role in the reading order
_HEADER_LABELS = {"page_header", "header", "running_header"}
_FOOTER_LABELS = {"page_footer", "footer", "running_footer"}
_FOOTNOTE_LABELS = {"footnote"}
_CAPTION_LABELS = {"caption"}
_FLOAT_LABELS = {"table", "picture", "figure", "image", "chart", "graphic"}


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
    norm = normalize_layout_label(label)
    if norm in _HEADER_LABELS:
        return "header"
    if norm in _FOOTER_LABELS:
        return "footer"
    if norm in _FOOTNOTE_LABELS:
        return "footnote"
    if norm in _CAPTION_LABELS:
        return "caption"
    if norm in _FLOAT_LABELS:
        return "float"
    return "body"


def detect_text_direction(texts: Iterable[str], language: str | None = None) -> str:
    """Infer the horizontal base direction of a document from its text content.

    The detection relies on the Unicode bidirectional property of each character: strong right-to-left
    characters (bidirectional categories `R` and `AL`, covering among others Hebrew, Arabic, Syriac,
    Thaana, N'Ko) are counted against strong left-to-right characters (category `L`). This makes the
    detection language-independent and robust to mixed content. When no strong character is found, the
    optional ISO 639 language code is used as a fallback hint.

    >>> from doctr.utils.reading_order import detect_text_direction
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
    rtl_count = 0
    ltr_count = 0
    for text in texts:
        if not isinstance(text, str):
            continue
        for char in text:
            bidi = unicodedata.bidirectional(char)
            if bidi in _RTL_BIDI_CATEGORIES:
                rtl_count += 1
            elif bidi == "L":
                ltr_count += 1
    if rtl_count == 0 and ltr_count == 0:
        if isinstance(language, str) and language.split("-")[0].lower() in RTL_LANGUAGES:
            return "rtl"
        return "ltr"
    return "rtl" if rtl_count > ltr_count else "ltr"


def _geometry_to_xyxy(geometry: Any) -> tuple[float, float, float, float]:
    pts = np.asarray(geometry, dtype=np.float64).reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _to_boxes(geoms: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(geoms, np.ndarray) and geoms.ndim == 2 and geoms.shape[1] >= 4:
        return geoms[:, :4].astype(np.float64)
    if isinstance(geoms, np.ndarray) and geoms.ndim == 3:
        return np.concatenate((geoms.min(axis=1), geoms.max(axis=1)), axis=-1).astype(np.float64)
    if len(geoms) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray([_geometry_to_xyxy(geom) for geom in geoms], dtype=np.float64)


def _to_canonical_ltr(boxes: np.ndarray, direction: str) -> np.ndarray:
    """Map boxes to a canonical left-to-right, top-to-bottom space for the given reading direction.

    Mirroring & axis swapping are performed around the extent of the boxes themselves, so both relative and
    absolute coordinates are supported.
    """
    if direction == "ltr" or boxes.shape[0] == 0:
        return boxes
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if direction == "rtl":  # mirror horizontally
        pivot = float(x0.min() + x1.max())
        return np.stack([pivot - x1, y0, pivot - x0, y1], axis=1)
    if direction == "ttb-ltr":  # vertical lines, columns ordered left to right: swap axes
        return np.stack([y0, x0, y1, x1], axis=1)
    if direction == "ttb-rtl":  # vertical lines, columns ordered right to left: swap axes & mirror
        pivot = float(x0.min() + x1.max())
        return np.stack([y0, pivot - x1, y1, pivot - x0], axis=1)
    raise ValueError(f"invalid reading direction '{direction}', should be one of {SUPPORTED_DIRECTIONS[1:]}")


def _overlap_ratios(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Compute the pairwise 1D overlap of intervals, normalized by the length of the smaller interval"""
    inter = np.minimum(ends[:, None], ends[None, :]) - np.maximum(starts[:, None], starts[None, :])
    min_len = np.minimum(ends - starts, (ends - starts)[:, None])
    return np.clip(inter, 0, None) / np.clip(min_len, 1e-9, None)


def _strict_rank(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Return the rank of each element in the strict total order given by (primary, secondary, index)"""
    order = np.lexsort((np.arange(primary.shape[0]), secondary, primary))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.shape[0])
    return ranks


def _topological_order(boxes: np.ndarray, x_overlap_threshold: float, y_overlap_threshold: float) -> list[int]:
    """Order boxes (already in canonical LTR space) with a column-following topological sort.

    Two families of "reads-before" relations are used (cf. Breuel 2003):

    * `i` precedes `j` when they overlap horizontally and `i` lies above `j`
    * `i` precedes `j` when they lie on the same visual row (vertical overlap, no horizontal overlap)
      and `i` is on the left of `j`

    The relations are resolved with Kahn's algorithm; among the available elements, the traversal favors the
    continuation of the current column (the closest element below the last emitted one with a horizontal
    overlap), which keeps multi-column bodies intact even for non-Manhattan layouts where recursive XY-cuts
    fail to find a valid split.
    """
    num_boxes = boxes.shape[0]
    if num_boxes <= 1:
        return list(range(num_boxes))

    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_overlap = _overlap_ratios(x0, x1)
    y_overlap = _overlap_ratios(y0, y1)

    # Strict total orders on both axes, so that the induced relations cannot create 2-cycles
    x_rank = _strict_rank(x0, x1)
    y_rank = _strict_rank(y0, y1)
    is_above = y_rank[:, None] < y_rank[None, :]
    is_left = x_rank[:, None] < x_rank[None, :]

    # i -> j edges: i must be read before j
    edges = ((x_overlap > x_overlap_threshold) & is_above) | (
        (x_overlap <= x_overlap_threshold) & (y_overlap > y_overlap_threshold) & is_left
    )
    np.fill_diagonal(edges, False)

    in_degree = edges.sum(axis=0)
    emitted = np.zeros(num_boxes, dtype=bool)
    order: list[int] = []
    last = -1

    # Column components: elements connected through horizontal overlap, ignoring page-spanning lines (e.g.
    # titles) which would otherwise merge separate columns. This keeps the traversal inside a column even when
    # the direct top-to-bottom continuation is momentarily broken by a fragmented or misaligned OCR line.
    page_width = float(x1.max() - x0.min()) or 1.0
    spanning = (x1 - x0) > 0.5 * page_width
    parent = np.arange(num_boxes)

    def _find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    col_edges = (x_overlap > x_overlap_threshold) & ~spanning[:, None] & ~spanning[None, :]
    for i, j in np.argwhere(np.triu(col_edges, 1)):
        ri, rj = _find(int(i)), _find(int(j))
        if ri != rj:
            parent[ri] = rj
    component = np.array([_find(i) for i in range(num_boxes)])

    while len(order) < num_boxes:
        ready = np.flatnonzero((in_degree == 0) & ~emitted)
        if ready.size == 0:  # cycle safety net (can only happen with degenerate overlapping geometries)
            ready = np.flatnonzero(~emitted)
            candidates = ready
        else:
            # Favor the continuation of the current column: the closest available element located below the
            # last emitted one with a horizontal overlap
            candidates = (
                ready[(x_overlap[last, ready] > x_overlap_threshold) & (y0[ready] >= y0[last])]
                if last >= 0
                else np.empty(0, dtype=int)
            )
            if candidates.size == 0 and last >= 0:
                candidates = ready[y_overlap[last, ready] > y_overlap_threshold]
            if candidates.size == 0 and last >= 0 and not spanning[last]:
                # Continuation broken (gap, fragment): stay in the same column before switching to another
                same_column = ready[component[ready] == component[last]]
                candidates = same_column if same_column.size else ready
            elif candidates.size == 0:
                candidates = ready
        # Topmost first, then leftmost (in canonical space)
        next_idx = int(candidates[np.lexsort((x0[candidates], y0[candidates]))[0]])
        order.append(next_idx)
        emitted[next_idx] = True
        in_degree = in_degree - edges[next_idx]
        last = next_idx

    return order


def _attach_captions(
    order: list[int],
    caption_idcs: list[int],
    boxes: np.ndarray,
    labels: list[str],
    max_distance: float,
) -> list[int]:
    """Insert captions right before (resp. after) the closest float they sit above (resp. below).

    Captions without a float within reach keep their natural spatial position in the body.
    """
    float_idcs = [idx for idx in order if labels[idx] in _FLOAT_LABELS]
    for cap in caption_idcs:
        cx0, cy0, cx1, cy1 = boxes[cap]
        best_target, best_dist = -1, float("inf")
        for target in float_idcs:
            tx0, ty0, tx1, ty1 = boxes[target]
            x_gap = max(tx0 - cx1, cx0 - tx1, 0.0)
            y_gap = max(ty0 - cy1, cy0 - ty1, 0.0)
            # Captions are expected right above/below (or overlapping) their float: penalize horizontal shifts
            dist = y_gap + 2 * x_gap
            if dist < best_dist:
                best_target, best_dist = target, dist
        if best_target >= 0 and best_dist <= max_distance:
            pos = order.index(best_target)
            # A caption located above (the center of) its float is read before it, otherwise after
            above = (cy0 + cy1) / 2 <= (boxes[best_target, 1] + boxes[best_target, 3]) / 2
            order.insert(pos if above else pos + 1, cap)
        else:  # fallback: insert at the natural spatial position
            cap_y0 = boxes[cap, 1]
            pos = next((i for i, idx in enumerate(order) if boxes[idx, 1] >= cap_y0), len(order))
            order.insert(pos, cap)
    return order


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
    angle_source = angle_geoms if angle_geoms is not None else []
    angle_pts = [np.asarray(geom, dtype=np.float64).reshape(-1, 2) for geom in angle_source]
    if len(angle_pts) > 0 and all(p.shape[0] == 4 for p in angle_pts):
        # Detection polygons are already reading-oriented (cf. `estimate_page_angle`): keep their vertex order
        angle = estimate_page_angle(np.stack(angle_pts) * scale)
    else:
        # Normalize the vertex order (TL, TR, BR, BL) so the estimation does not depend on the vertex
        angle = estimate_page_angle(np.stack([order_points(p * scale) for p in pts]))
    if not np.isfinite(angle) or abs(angle) < min_angle or abs(angle) >= 45:
        return list(geoms), region_geoms
    # Rigid rotation zeroing the estimated angle; the center is irrelevant for ordering purposes
    theta = np.deg2rad(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    all_pts = np.concatenate(pts, axis=0) * scale
    center = all_pts.mean(axis=0)

    def _rotate(points: np.ndarray) -> np.ndarray:
        return ((points * scale - center) @ rot.T + center) / scale

    deskewed = [_rotate(p) for p in pts]

    def _corners(points: np.ndarray) -> np.ndarray:
        # Straight ((xmin, ymin), (xmax, ymax)) regions must be expanded to their 4 corners before rotating,
        # otherwise only the diagonal would be rotated and the resulting extent would be underestimated
        if points.shape[0] == 4:
            return points
        (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    region_pts = [np.asarray(geom, dtype=np.float64).reshape(-1, 2) for geom in region_geoms]
    regions_out = [_rotate(_corners(p)) for p in region_pts]
    return deskewed, regions_out


def sort_reading_order(
    geoms: Sequence[Any] | np.ndarray,
    direction: str = "ltr",
    labels: Sequence[str | None] | None = None,
    x_overlap_threshold: float = 0.2,
    y_overlap_threshold: float = 0.5,
    caption_max_distance: float = 0.1,
    page_shape: tuple[int, int] | None = None,
    angle_geoms: Sequence[Any] | np.ndarray | None = None,
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

    Returns:
        the permutation of the input indices which sorts the elements in reading order
    """
    if direction not in SUPPORTED_DIRECTIONS[1:]:
        raise ValueError(f"invalid reading direction '{direction}', should be one of {SUPPORTED_DIRECTIONS[1:]}")
    geoms, _ = deskew_reading_geometries(geoms, page_shape=page_shape, angle_geoms=angle_geoms)
    boxes = _to_boxes(geoms)
    num_boxes = boxes.shape[0]
    if labels is not None and len(labels) != num_boxes:
        raise ValueError(f"Incompatible number of labels ({len(labels)}) and geometries ({num_boxes})")
    if num_boxes <= 1:
        return list(range(num_boxes))

    canonical = _to_canonical_ltr(boxes, direction)

    def _order(idcs: list[int]) -> list[int]:
        if len(idcs) == 0:
            return []
        sub_order = _topological_order(canonical[idcs], x_overlap_threshold, y_overlap_threshold)
        return [idcs[i] for i in sub_order]

    if labels is None:
        return _order(list(range(num_boxes)))

    norm_labels = [normalize_layout_label(label) for label in labels]
    groups: dict[str, list[int]] = {"header": [], "body": [], "caption": [], "footnote": [], "footer": []}
    for idx, label in enumerate(norm_labels):
        role = layout_label_role(label)
        groups["body" if role == "float" else role].append(idx)

    body_order = _attach_captions(
        _order(groups["body"]), _order(groups["caption"]), canonical, norm_labels, caption_max_distance
    )
    return _order(groups["header"]) + body_order + _order(groups["footnote"]) + _order(groups["footer"])


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
) -> list[list[int]]:
    """Order elements in reading order and group consecutive ones into segments (paragraphs or regions).

    This is typically used to regroup individual lines into paragraphs once they have been put in reading
    order: two consecutive elements are merged into the same segment when they share the same (possibly None)
    layout label, belong to the same column and are vertically close enough.

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

    Returns:
        a partition of the input indices into reading-ordered segments (each segment being itself in
        reading order)
    """
    # De-skew once so the ordering and the vertical-gap measurement share the same upright frame
    geoms, _ = deskew_reading_geometries(geoms, page_shape=page_shape, angle_geoms=angle_geoms)
    order = sort_reading_order(
        geoms,
        direction=direction,
        labels=labels,
        x_overlap_threshold=x_overlap_threshold,
        y_overlap_threshold=y_overlap_threshold,
        caption_max_distance=caption_max_distance,
    )
    if len(order) == 0:
        return []
    canonical = _to_canonical_ltr(_to_boxes(geoms), direction)
    median_height = float(np.median(canonical[:, 3] - canonical[:, 1]))
    norm_labels = [normalize_layout_label(labels[idx] if labels is not None else None) for idx in range(len(order))]

    segments: list[list[int]] = [[order[0]]]
    for prev, cur in zip(order[:-1], order[1:]):
        x_gap = min(canonical[prev, 2], canonical[cur, 2]) - max(canonical[prev, 0], canonical[cur, 0])
        x_overlap = x_gap / max(
            min(canonical[prev, 2] - canonical[prev, 0], canonical[cur, 2] - canonical[cur, 0]), 1e-9
        )
        same_segment = (
            norm_labels[prev] == norm_labels[cur]
            and layout_label_role(norm_labels[cur]) != "float"
            and x_overlap > x_overlap_threshold
            and canonical[cur, 1] - canonical[prev, 3] <= paragraph_gap * median_height
        )
        if same_segment:
            segments[-1].append(cur)
        else:
            segments.append([cur])
    return segments


def assign_layout_labels(
    geoms: Sequence[Any] | np.ndarray,
    layout_geoms: Sequence[Any] | np.ndarray,
    layout_labels: Sequence[str],
    min_coverage: float = 0.5,
    page_shape: tuple[int, int] | None = None,
    angle_geoms: Sequence[Any] | np.ndarray | None = None,
) -> list[str | None]:
    """Assign a layout label to each element based on its overlap with the detected layout regions.

    Each element receives the label of the region covering the largest share of its area, provided this share
    reaches `min_coverage`; otherwise its label is None (treated as regular body content).

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

    inter_w = np.minimum(boxes[:, None, 2], regions[None, :, 2]) - np.maximum(boxes[:, None, 0], regions[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 3], regions[None, :, 3]) - np.maximum(boxes[:, None, 1], regions[None, :, 1])
    inter = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)
    areas = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9, None)
    coverage = inter / areas[:, None]

    best = coverage.argmax(axis=1)
    return [
        str(layout_labels[reg_idx]) if coverage[box_idx, reg_idx] >= min_coverage else None
        for box_idx, reg_idx in enumerate(best)
    ]


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
