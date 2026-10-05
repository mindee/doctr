# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import re
from html import escape as _html_escape
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element as ETElement
from xml.etree.ElementTree import SubElement

import numpy as np

import doctr
from doctr.io.figures import FigureEncoder, is_picture_label, picture_regions
from doctr.utils.common_types import BoundingBox

if TYPE_CHECKING:  # pragma: no cover
    from doctr.io.elements import Block, KIEPage, LayoutElement, Line, Page, Table

__all__ = [
    "AsciiDocExporter",
    "DocumentExportsMixin",
    "HTMLExporter",
    "KIEPageExportsMixin",
    "MarkdownExporter",
    "PageExportsMixin",
    "TextExporter",
    "XMLExporter",
    "page_reading_order",
]


def _export_as(exporters: dict[str, Any], format: str, **kwargs: Any) -> Any:
    fmt = format.strip().lower()
    if fmt not in exporters:
        raise ValueError(f"unsupported export format '{format}', should be one of {sorted(exporters)}")
    return exporters[fmt](**kwargs)


def to_json_safe(value: Any) -> Any:
    """Recursively convert NumPy containers and scalars into built-in Python types.

    Args:
        value: any exported value

    Returns:
        the same value with every NumPy array converted to nested tuples and every NumPy scalar to its
        Python equivalent
    """
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else tuple(to_json_safe(item) for item in value)
    if isinstance(value, np.generic):  # np.float32, np.int64, np.bool_, ...
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_json_safe(item) for item in value)
    if isinstance(value, (list, set, frozenset)):
        return [to_json_safe(item) for item in value]
    return value


_LIST_LABELS = {"list_item"}
# Characters / line markers that carry a structural meaning and are escaped to preserve the raw OCR text
_MD_SPECIAL_CHARS = "\\`*_[]|#<>"
_MD_LINE_MARKERS = "-+>#=`"
# Ordered list marker at the start of a line, e.g. "3." or "3)"
_MD_ORDERED_MARKER = re.compile(r"^(\s*\d{1,9})([.)])(?=\s|$)")
_ADOC_SPECIAL_CHARS = "\\`*_#^~|+{}<>"
_ADOC_LINE_MARKERS = "=*.-/+"


def _covering_region_indices(
    geoms: list[Any],
    region_geoms: list[Any],
    min_coverage: float = 0.5,
    region_labels: list[str] | None = None,
) -> list[int]:
    """Return the index of the layout region each geometry is assigned to (-1 for none)

    Same criterion as :func:`doctr.models.reading_order.assign_layout_labels`, on geometries in the same frame.
    """
    from doctr.models.reading_order.base import _covering_regions, _to_boxes

    if len(region_geoms) == 0 or len(geoms) == 0:
        return [-1] * len(geoms)
    return _covering_regions(_to_boxes(geoms), _to_boxes(region_geoms), min_coverage, region_labels)


def _xyxy(geometry: Any) -> tuple[float, float, float, float]:
    """Return the enclosing box (xmin, ymin, xmax, ymax) of a box or a polygon"""
    pts = np.asarray(geometry, dtype=np.float64).reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _join_caption_lines(lines: list[str]) -> str:
    """Join the lines of a caption, mending the words hyphenated at a line break ("dif-" + "ferent")"""
    text = ""
    for line in (line.strip() for line in lines):
        if not line:
            continue
        if text.endswith("-") and line[0].islower():
            fragment = text[:-1].rsplit(" ", 1)[-1]
            text = text[:-1] + line if fragment.isalpha() and fragment.islower() else text + line
        else:
            text = f"{text} {line}" if text else line
    return text


def _reading_order_signature(page: "Page", direction: str) -> tuple[Any, ...]:
    """Return a cheap fingerprint of a page, used to invalidate the reading-order cache

    It covers the direction and the identity of the blocks (with their line count), tables and layout regions.
    In-place edits of the words of a line are not detected: drop `_reading_order_cache` after such edits.
    """
    return (
        direction,
        tuple((id(block), len(block.lines)) for block in page.blocks),
        tuple(id(table) for table in getattr(page, "tables", ()) or ()),
        tuple(id(region) for region in getattr(page, "layout", ()) or ()),
    )


def _store_reading_order(page: "Page", signature: tuple[Any, ...], result: tuple[Any, ...]) -> None:
    """Memoize a reading-order result on the page, ignoring pages that reject attribute assignment."""
    try:
        page._reading_order_cache = (signature, result)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        pass


def page_reading_order(
    page: "Page", direction: str = "auto", include_figures: bool = False
) -> tuple[list[Any], list[str | None], str]:
    """Linearize the content of a page (blocks, tables and optionally figures) in reading order

    The result is cached on the page, so a page exported to several formats is ordered once. The figures always take
    part in the ordering, `include_figures` only controls whether they are returned.

    Args:
        page: the page to linearize
        direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
        include_figures: whether to return the figure regions too

    Returns:
        the ordered items, their layout label (None without layout) and the effective reading direction
    """
    from doctr.io.elements import Block, LayoutElement, Table
    from doctr.models.reading_order import (
        ReadingOrderPredictor,
        deskew_reading_geometries,
        layout_label_role,
        normalize_layout_label,
        resolve_reading_segments,
    )
    from doctr.models.reading_order.base import _FURNITURE_ROLES, _to_boxes

    signature = _reading_order_signature(page, direction)
    cached = getattr(page, "_reading_order_cache", None)
    if cached is not None and cached[0] == signature:
        items, labels, resolved = cached[1]
        return _select_items(list(items), list(labels), resolved, include_figures)

    texts = [word.value for block in page.blocks for line in block.lines for word in line.words]
    language = page.language.get("value") if isinstance(page.language, dict) else None
    direction = ReadingOrderPredictor(direction=direction).resolve_direction(texts, language=language)
    region_geoms = [region.geometry for region in page.layout]
    region_labels = [region.type for region in page.layout]

    lines = [line for block in page.blocks for line in block.lines]
    # Figures take part in the ordering as floats
    figures = picture_regions(page)
    elements: list[Any] = [*lines, *page.tables, *figures]
    if len(elements) == 0:
        _store_reading_order(page, signature, ([], [], direction))
        return [], [], direction
    # De-skew once so labeling, ordering and region grouping share the same upright frame; the page angle is
    # estimated from the word polygons, which carry the detection model's true orientation
    elt_geoms, region_geoms = deskew_reading_geometries(
        [elt.geometry for elt in elements],
        region_geoms,
        page_shape=page.dimensions,
        angle_geoms=[word.geometry for line in lines for word in line.words],
    )
    elt_labels: list[str | None] = [None] * len(elements)
    elt_regions = _covering_region_indices(elt_geoms, region_geoms, region_labels=region_labels)
    if len(region_geoms) > 0:
        elt_labels = [str(region_labels[reg]) if reg >= 0 else None for reg in elt_regions]

    # A figure lying in page furniture (e.g. a logo in the page header) is read with it, along with its text
    furniture = [reg for reg, label in enumerate(region_labels) if layout_label_role(label) in _FURNITURE_ROLES]
    fig_idcs = [idx for idx, elt in enumerate(elements) if isinstance(elt, LayoutElement)]
    fig_furniture = _covering_region_indices(
        [elt_geoms[idx] for idx in fig_idcs], [region_geoms[reg] for reg in furniture]
    )
    region_index = {id(region): reg for reg, region in enumerate(page.layout)}
    furniture_labels: dict[int, str] = {}  # figure region index -> furniture label
    for idx, reg in zip(fig_idcs, fig_furniture):
        elt_labels[idx] = str(region_labels[furniture[reg]]) if reg >= 0 else elements[idx].type
        if reg >= 0:
            furniture_labels[region_index[id(elements[idx])]] = str(region_labels[furniture[reg]])
    for idx, elt in enumerate(elements):
        if isinstance(elt, Table):
            elt_labels[idx] = "Table"
        elif not isinstance(elt, LayoutElement) and elt_regions[idx] in furniture_labels:
            elt_labels[idx] = furniture_labels[elt_regions[idx]]
    # Page-wide figures do not vote in the multi-column detection
    boxes = _to_boxes(elt_geoms)
    span = float(boxes[:, 2].max() - boxes[:, 0].min()) or 1.0
    column_voters = [
        not (isinstance(elt, LayoutElement) and box[2] - box[0] > 0.5 * span) for elt, box in zip(elements, boxes)
    ]
    segments = resolve_reading_segments(
        elt_geoms, direction=direction, labels=elt_labels, column_voters=column_voters, region_groups=elt_regions
    )

    def _is_float_item(idx: int) -> bool:
        return isinstance(elements[idx], (Table, LayoutElement))

    def _split_floats(segment: list[int]) -> list[list[int]]:
        """Split the tables and figures out of a segment (a figure in page furniture can share one)"""
        runs: list[list[int]] = []
        for idx in segment:
            if runs and not _is_float_item(idx) and not _is_float_item(runs[-1][-1]):
                runs[-1].append(idx)
            else:
                runs.append([idx])
        return runs

    segments = [run for segment in segments for run in _split_floats(segment)]

    items = []
    labels = []
    line_owner = {id(line): idx for idx, block in enumerate(page.blocks) for line in block.lines}
    pending_artefacts = {idx: list(block.artefacts) for idx, block in enumerate(page.blocks) if block.artefacts}

    def _claim_artefacts(block_lines: list[Any]) -> list[Any]:
        claimed: list[Any] = []
        for line in block_lines:
            owner = line_owner.get(id(line))
            if owner is not None and owner in pending_artefacts:
                claimed.extend(pending_artefacts.pop(owner))
        return claimed

    # Region index covering each element, used to group the lines of a wrapped list item under a single bullet
    region_idx = elt_regions
    open_list_region: int | None = None  # region of the list bullet currently being built (None outside a list)
    for segment in segments:
        first = elements[segment[0]]
        seg_label = elt_labels[segment[0]]
        if isinstance(first, (Table, LayoutElement)):
            items.append(first)
            labels.append("Table" if isinstance(first, Table) else seg_label)
            open_list_region = None
            continue
        if normalize_layout_label(seg_label) in _LIST_LABELS:
            # One bullet per list-item region: consecutive lines sharing the same region are one bullet, so a
            # list item wrapped over several visual lines renders as a single bullet point.
            for idx in segment:
                region = region_idx[idx]
                if open_list_region is not None and region == open_list_region and region != -1:
                    merged = [*items[-1].lines, elements[idx]]
                    items[-1] = Block(lines=merged, artefacts=[*items[-1].artefacts, *_claim_artefacts([merged[-1]])])
                else:
                    items.append(Block(lines=[elements[idx]], artefacts=_claim_artefacts([elements[idx]])))
                    labels.append(seg_label)
                    open_list_region = region
        else:
            block_lines = [elements[idx] for idx in segment]
            items.append(Block(lines=block_lines, artefacts=_claim_artefacts(block_lines)))
            labels.append(seg_label)
            open_list_region = None
    # Artefacts of blocks without any line stay attached to the page
    leftover = [artefact for artefacts in pending_artefacts.values() for artefact in artefacts]
    if leftover:
        last_block = next((item for item in reversed(items) if isinstance(item, Block)), None)
        if last_block is not None:
            last_block.artefacts = [*last_block.artefacts, *leftover]
    _store_reading_order(page, signature, (items, labels, direction))
    return _select_items(list(items), list(labels), direction, include_figures)


def _select_items(
    items: list[Any], labels: list[str | None], direction: str, include_figures: bool
) -> tuple[list[Any], list[str | None], str]:
    """Drop the figures from a linearization, unless requested"""
    from doctr.io.elements import LayoutElement

    if include_figures:
        return items, labels, direction
    kept = [(item, label) for item, label in zip(items, labels) if not isinstance(item, LayoutElement)]
    return [item for item, _ in kept], [label for _, label in kept], direction


def _line_render_direction(line: "Line", page_direction: str, auto: bool) -> str:
    """Resolve the direction used to order the words of a line.

    For vertical pages the words are always read top to bottom. For horizontal pages, when the page direction
    was inferred automatically, the base direction of each line is detected from its own text so that an
    embedded left-to-right run (e.g. a Latin quotation on an Arabic page) keeps its natural word order; when
    the direction is set explicitly, it is applied uniformly to every line.
    """
    if page_direction in ("ttb-rtl", "ttb-ltr") or not auto or len(line.words) <= 1:
        return page_direction
    from doctr.models.reading_order import detect_text_direction

    return detect_text_direction([word.render() for word in line.words])


def ordered_line_words(line: "Line", direction: str = "ltr", auto: bool = False) -> list[Any]:
    """Return the words of a line in reading order.

    Args:
        line: the line whose words should be ordered
        direction: the reading direction resolved for the page
        auto: whether the page direction was inferred (each line then gets its own base direction)

    Returns:
        the words of the line, ordered logically
    """
    direction = _line_render_direction(line, direction, auto)
    if direction in ("ttb-rtl", "ttb-ltr"):
        return sorted(line.words, key=lambda word: float(np.asarray(word.geometry, dtype=np.float64)[..., 1].mean()))
    if direction == "rtl":
        return sorted(line.words, key=lambda word: -float(np.asarray(word.geometry, dtype=np.float64)[..., 0].mean()))
    return list(line.words)


def predictions_in_reading_order(page: "KIEPage", predictions: list[Any], direction: str = "auto") -> list[Any]:
    """Sort the predictions of a single KIE detection class in reading order.

    Args:
        page: the KIE page the predictions belong to (used for its dimensions and detected language)
        predictions: the predictions of one detection class
        direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'

    Returns:
        the predictions, ordered logically
    """
    from doctr.models.reading_order import ReadingOrderPredictor

    if len(predictions) < 2:
        return list(predictions)
    language = page.language.get("value") if isinstance(page.language, dict) else None
    order = ReadingOrderPredictor(direction=direction)(
        [prediction.geometry for prediction in predictions],
        texts=[prediction.value for prediction in predictions],
        language=language,
        page_shape=page.dimensions,
    )
    return [predictions[idx] for idx in order]


class _FigurePlan(NamedTuple):
    """Figures of a page to export, by item index"""

    markup: dict[int, str]  # figure markup
    captions: set[int]  # captions rendered with their figure
    hidden: set[int]  # text left out of the export


class _PageTextExporter:
    """Shared logic of the reading-order-aware text exporters

    Subclasses define the format specifics: headings, bullets, escaping, line finalization, tables and figures.
    """

    headings: ClassVar[dict[str, str]] = {}
    bullet: ClassVar[str] = "- "
    block_break: ClassVar[str] = "\n\n"
    page_break: ClassVar[str] = "\n\n"
    # Whether the format can render figures
    supports_figures: ClassVar[bool] = False
    # Rendered for a figure without pixels
    figure_placeholder: ClassVar[str] = ""

    def escape_text(self, text: str) -> str:
        """Escape the characters carrying a structural meaning in the target format"""
        return text

    def finalize_line(self, line: str) -> str:
        """Neutralize the block-level markers a line must not start with in the target format"""
        return line

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a recognized table in the target format"""
        raise NotImplementedError

    def render_figure(self, source: str | None, caption: str | None = None, escape: bool = True) -> str:
        """Render a figure in the target format

        Args:
            source: the image source (data URI or relative path), None to render the placeholder
            caption: the unescaped caption of the figure, if any
            escape: whether to escape the characters carrying a structural meaning

        Returns:
            the figure markup
        """
        return self.figure_placeholder

    def render_heading(self, norm_label: str, lines: list[str], escape: bool = True) -> str:
        """Render a heading in the target format"""
        return self.headings[norm_label] + " ".join(lines)

    def render_list_item(self, lines: list[str], escape: bool = True) -> str:
        """Render a list item in the target format"""
        text = " ".join(lines)
        return self.bullet + (self.finalize_line(text) if escape else text)

    def render_list(self, items: list[str]) -> str:
        """Render a list from its rendered items in the target format"""
        return "\n".join(items)

    def render_paragraph(self, lines: list[str], escape: bool = True) -> str:
        """Render a paragraph in the target format"""
        return "\n".join(self.finalize_line(line) if escape else line for line in lines)

    def class_header(self, class_name: str, escape: bool = True) -> str:
        """Render the header of a detection class in a KIE export"""
        raise NotImplementedError

    def _line_text(self, line: "Line", direction: str, escape: bool) -> str:
        """Render the text of a line, ordering the words according to the reading direction."""
        text = " ".join(word.render() for word in ordered_line_words(line, direction))
        return self.escape_text(text) if escape else text

    def _block_lines(self, block: "Block", direction: str, escape: bool, auto: bool) -> list[str]:
        """Render the non-empty lines of a block"""
        lines = [self._line_text(line, _line_render_direction(line, direction, auto), escape) for line in block.lines]
        return [line for line in lines if line.strip()]

    def _plan_figures(
        self,
        page: "Page",
        items: list[Any],
        labels: list[str | None],
        encoder: FigureEncoder,
        direction: str,
        escape: bool,
        auto: bool,
        include_furniture: bool = True,
    ) -> "_FigurePlan":
        """Render the figures of a page, with their caption when they carry their pixels

        Args:
            page: the page to export
            items: the linearized page content, figures included
            labels: the layout label of each item
            encoder: the figure encoder
            direction: the effective reading direction
            escape: whether to escape the characters carrying a structural meaning
            auto: whether the reading direction was detected automatically
            include_furniture: whether page headers, page footers and footnotes are exported

        Returns:
            the figure markup, the captions rendered with their figure and the text to leave out
        """
        from doctr.io.elements import Block, LayoutElement, Table
        from doctr.models.reading_order import deskew_reading_geometries, layout_label_role, normalize_layout_label
        from doctr.models.reading_order.base import _FURNITURE_ROLES, _caption_distance

        figure_idcs = [idx for idx, item in enumerate(items) if isinstance(item, LayoutElement)]
        render = self.supports_figures and encoder.enabled
        if not figure_idcs or not (render or not include_furniture):
            return _FigurePlan({}, set(), set())

        # Same upright frame as the reading order
        upright, upright_regions = deskew_reading_geometries(
            [item.geometry for item in items],
            [region.geometry for region in page.layout],
            page_shape=page.dimensions,
            angle_geoms=[word.geometry for block in page.blocks for line in block.lines for word in line.words],
        )
        boxes = [_xyxy(geom) for geom in upright]

        def _coverage(box: tuple[float, ...], target: tuple[float, ...]) -> float:
            """Share of `box` covered by `target`"""
            inter = max(min(box[2], target[2]) - max(box[0], target[0]), 0.0) * max(
                min(box[3], target[3]) - max(box[1], target[1]), 0.0
            )
            return inter / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)

        def _figure_text(fig_idx: int) -> set[int]:
            """Blocks covered by a figure and labeled as part of it (or as furniture, for a figure in furniture)"""
            in_furniture = layout_label_role(labels[fig_idx]) in _FURNITURE_ROLES
            return {
                idx
                for idx, item in enumerate(items)
                if isinstance(item, Block)
                and (
                    is_picture_label(labels[idx])
                    or (in_furniture and layout_label_role(labels[idx]) in _FURNITURE_ROLES)
                )
                and _coverage(boxes[idx], boxes[fig_idx]) >= 0.5
            }

        hidden: set[int] = set()
        if not include_furniture:
            # Figures read with the page furniture are left out with it
            furniture = {idx for idx in figure_idcs if layout_label_role(labels[idx]) in _FURNITURE_ROLES}
            for idx in furniture:
                hidden |= _figure_text(idx)
            figure_idcs = [idx for idx in figure_idcs if idx not in furniture]
        if not render or not figure_idcs:
            return _FigurePlan({}, set(), hidden)

        sources = {idx: encoder.source(page, items[idx], rank) for rank, idx in enumerate(figure_idcs, start=1)}
        # Only a figure exported with its pixels makes its text redundant
        for idx in figure_idcs:
            if sources[idx] is not None:
                hidden |= _figure_text(idx)
        region_of = _covering_region_indices(upright, upright_regions, region_labels=[r.type for r in page.layout])

        def _is_caption(idx: int) -> bool:
            return isinstance(items[idx], Block) and normalize_layout_label(labels[idx]) == "caption"

        def _is_inner_text(idx: int, fig_idx: int) -> bool:
            return (
                isinstance(items[idx], Block) and not _is_caption(idx) and _coverage(boxes[idx], boxes[fig_idx]) >= 0.5
            )

        def _caption_run(start: int) -> tuple[int, ...]:
            # A caption region can span several blocks
            if region_of[start] == -1:
                return (start,)
            return tuple(idx for idx in range(len(items)) if _is_caption(idx) and region_of[idx] == region_of[start])

        def _run_box(run: tuple[int, ...]) -> tuple[float, ...]:
            return tuple(func(boxes[idx][k] for idx in run) for k, func in enumerate((min, min, max, max)))

        # Candidate captions: right before the figure, or right after the text inside it
        claims: dict[tuple[int, ...], list[int]] = {}
        for idx in figure_idcs:
            if sources[idx] is None:
                continue
            after = idx + 1
            while after < len(items) and _is_inner_text(after, idx):
                after += 1
            for cap in (idx - 1, after):
                if 0 <= cap < len(items) and _is_caption(cap):
                    claims.setdefault(_caption_run(cap), []).append(idx)

        def _table_without_caption(idx: int, run: tuple[int, ...]) -> bool:
            # A table with its own caption on its other side does not compete for this one
            if not (0 <= idx < len(items) and isinstance(items[idx], Table)):
                return False
            far = idx + 1 if idx > run[-1] else idx - 1
            return not (0 <= far < len(items) and _is_caption(far))

        # Each caption goes to its closest float, and each figure keeps its closest caption
        captions: dict[int, tuple[int, ...]] = {}
        for run, candidates in claims.items():
            cap_box = _run_box(run)
            tables = [idx for idx in (run[0] - 1, run[-1] + 1) if _table_without_caption(idx, run)]
            best = min([*candidates, *tables], key=lambda elt: _caption_distance(cap_box, boxes[elt]))
            if best in tables:
                continue
            dist = _caption_distance(cap_box, boxes[best])
            if best in captions and _caption_distance(_run_box(captions[best]), boxes[best]) <= dist:
                continue
            captions[best] = run

        markup: dict[int, str] = {}
        consumed: set[int] = set()
        for idx in figure_idcs:
            caption = None
            if idx in captions:
                lines = [
                    line
                    for cap_idx in captions[idx]
                    for line in self._block_lines(items[cap_idx], direction, False, auto)
                ]
                caption = _join_caption_lines(lines) or None
                if caption is not None:
                    consumed.update(captions[idx])
            markup[idx] = self.render_figure(sources[idx], caption, escape=escape)
        return _FigurePlan(markup, consumed, hidden)

    def export_page(
        self,
        page: "Page",
        direction: str = "auto",
        escape: bool = True,
        include_furniture: bool = True,
        block_break: str | None = None,
        images: "str | FigureEncoder | None" = "placeholder",
    ) -> str:
        """Export a page, with its content sorted in reading order.

        Args:
            page: the page to export
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters or markers carrying a structural meaning should be neutralized
            include_furniture: whether page headers, page footers and footnotes should be included
            block_break: the string inserted between two blocks (the format-specific default when None)
            images: how to render the figures: 'none', 'placeholder', 'embedded', or a
                :class:`~doctr.io.FigureEncoder` (required for 'referenced')

        Returns:
            the exported page as a string
        """
        from doctr.io.elements import LayoutElement, Table
        from doctr.models.reading_order import layout_label_role, normalize_layout_label
        from doctr.models.reading_order.base import _FURNITURE_ROLES

        auto = direction == "auto"
        encoder = FigureEncoder.resolve(images)
        items, labels, direction = page_reading_order(page, direction, include_figures=True)
        figures, absorbed_captions, figure_text = self._plan_figures(
            page, items, labels, encoder, direction, escape, auto, include_furniture=include_furniture
        )
        parts: list[str] = []
        list_group: list[str] = []

        def _flush_list() -> None:
            if list_group:
                parts.append(self.render_list(list_group))
                list_group.clear()

        for index, (item, label) in enumerate(zip(items, labels)):
            if not include_furniture and layout_label_role(label) in _FURNITURE_ROLES:
                continue
            if isinstance(item, LayoutElement):
                if figures.get(index):
                    _flush_list()
                    parts.append(figures[index])
                continue
            if index in absorbed_captions or index in figure_text:
                continue
            if isinstance(item, Table):
                _flush_list()
                rendered = self.render_table(item, escape=escape)
                if rendered:
                    parts.append(rendered)
                continue
            item_lines = self._block_lines(item, direction, escape, auto)
            if len(item_lines) == 0:
                continue
            norm_label = normalize_layout_label(label)
            if norm_label in self.headings:
                _flush_list()
                parts.append(self.render_heading(norm_label, item_lines, escape))
            elif norm_label in _LIST_LABELS:
                # A list item (possibly wrapped over several lines) renders as a single bullet
                list_group.append(self.render_list_item(item_lines, escape))
            else:
                _flush_list()
                parts.append(self.render_paragraph(item_lines, escape))
        _flush_list()
        return (self.block_break if block_break is None else block_break).join(parts)

    def export_kie_page(self, page: "KIEPage", direction: str = "auto", escape: bool = True) -> str:
        """Export a KIE page, with the predictions of each class sorted in reading order.

        Args:
            page: the KIE page to export
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters or markers carrying a structural meaning should be neutralized

        Returns:
            the exported page as a string, with one section per detection class
        """
        parts: list[str] = []
        for class_name, predictions in page.predictions.items():
            if len(predictions) == 0:
                continue
            values = "\n".join(
                self.bullet + (self.finalize_line(self.escape_text(prediction.value)) if escape else prediction.value)
                for prediction in predictions_in_reading_order(page, predictions, direction)
            )
            parts.append(f"{self.class_header(class_name, escape)}\n\n{values}")
        return "\n\n".join(parts)

    def export_document(self, document: Any, page_break: str | None = None, **kwargs: Any) -> str:
        """Export a document page by page.

        Args:
            document: the document to export
            page_break: the string inserted between two pages (a format-specific default when None)
            **kwargs: additional keyword arguments passed to the page export

        Returns:
            the exported document as a string
        """
        from doctr.io.elements import KIEPage

        page_break = self.page_break if page_break is None else page_break
        return page_break.join(
            self.export_kie_page(page, **kwargs) if isinstance(page, KIEPage) else self.export_page(page, **kwargs)
            for page in document.pages
        )


class TextExporter(_PageTextExporter):
    """Export OCR results to plain text, with the content sorted in reading order.

    >>> from doctr.io import TextExporter
    >>> text = TextExporter().export_page(page)
    """

    headings: ClassVar[dict[str, str]] = {}
    bullet: ClassVar[str] = ""
    block_break: ClassVar[str] = "\n\n"
    page_break: ClassVar[str] = "\n\n\n\n"

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a table as tab-separated values, one line per row"""
        return table.render()

    def class_header(self, class_name: str, escape: bool = True) -> str:
        return f"{class_name}:"


class MarkdownExporter(_PageTextExporter):
    """Export OCR results to Markdown, with the content sorted in reading order.

    >>> from doctr.io import MarkdownExporter
    >>> markdown = MarkdownExporter().export_page(page)
    """

    headings: ClassVar[dict[str, str]] = {"title": "# ", "section_header": "## "}
    bullet: ClassVar[str] = "- "
    page_break: ClassVar[str] = "\n\n---\n\n"
    supports_figures: ClassVar[bool] = True
    figure_placeholder: ClassVar[str] = "<!-- image -->"

    def escape_text(self, text: str) -> str:
        return "".join(f"\\{char}" if char in _MD_SPECIAL_CHARS else char for char in text)

    def finalize_line(self, line: str) -> str:
        stripped = line.lstrip()
        if stripped and stripped[0] in _MD_LINE_MARKERS:
            return f"\\{line}" if line[0] != "\\" else line
        # Escape the list delimiter: a backslash before a digit is not a Markdown escape
        return _MD_ORDERED_MARKER.sub(r"\1\\\2", line)

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a table as a GitHub-flavored Markdown table (first row used as header)"""
        grid = table.to_grid()
        if len(grid) == 0 or len(grid[0]) == 0:
            return ""

        def _cell(value: str) -> str:
            value = self.escape_text(value) if escape else value.replace("|", "\\|")
            return value.replace("\n", " ").strip()

        rows = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in grid]
        separator = "| " + " | ".join("---" for _ in grid[0]) + " |"
        return "\n".join([rows[0], separator, *rows[1:]])

    def render_figure(self, source: str | None, caption: str | None = None, escape: bool = True) -> str:
        """Render a figure as an image, with its caption as alt text and as a visible italic line below"""
        if source is None:
            return self.figure_placeholder
        caption = caption or ""
        # The link label delimiters are escaped in any case
        if escape:
            alt = self.escape_text(caption)
        else:
            alt = caption.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        image = f"![{alt}]({source})"
        if not caption:
            return image
        return f"{image}\n\n*{self.escape_text(caption) if escape else caption}*"

    def class_header(self, class_name: str, escape: bool = True) -> str:
        return f"**{self.escape_text(class_name) if escape else class_name}**"


class AsciiDocExporter(_PageTextExporter):
    """Export OCR results to AsciiDoc, with the content sorted in reading order.

    >>> from doctr.io import AsciiDocExporter
    >>> asciidoc = AsciiDocExporter().export_page(page)
    """

    headings: ClassVar[dict[str, str]] = {"title": "== ", "section_header": "=== "}
    bullet: ClassVar[str] = "* "
    page_break: ClassVar[str] = "\n\n<<<\n\n"
    supports_figures: ClassVar[bool] = True
    figure_placeholder: ClassVar[str] = "// image"

    def escape_text(self, text: str) -> str:
        return "".join(f"\\{char}" if char in _ADOC_SPECIAL_CHARS else char for char in text)

    def finalize_line(self, line: str) -> str:
        stripped = line.lstrip()
        if stripped and stripped[0] in _ADOC_LINE_MARKERS:
            return f"{{empty}}{line}"
        return line

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a table as an AsciiDoc table (first row used as header)"""
        grid = table.to_grid()
        if len(grid) == 0 or len(grid[0]) == 0:
            return ""

        def _row(row: list[str]) -> str:
            return " ".join(
                "|" + (self.escape_text(value) if escape else value.replace("|", "\\|")).replace("\n", " ").strip()
                for value in row
            )

        return "\n".join(["|===", _row(grid[0]), "", *[_row(row) for row in grid[1:]], "|==="])

    def render_figure(self, source: str | None, caption: str | None = None, escape: bool = True) -> str:
        """Render a figure as a block image macro, titled with its caption"""
        if source is None:
            return self.figure_placeholder
        if not caption:
            return f"image::{source}[]"
        title = self.escape_text(caption) if escape else caption
        if title.startswith("."):  # would open a literal block
            title = "{empty}" + title
        # Quoted, so that commas do not split the alt text into several attributes
        alt = caption.replace('"', '\\"')
        if escape:  # attribute references are also substituted in attribute values
            alt = alt.replace("{", "\\{").replace("}", "\\}")
        # The detected caption carries its own label (e.g. "Figure 3:"): disable the automatic "Figure N." prefix
        return f'[caption=""]\n.{title}\nimage::{source}["{alt}"]'

    def class_header(self, class_name: str, escape: bool = True) -> str:
        return f"*{self.escape_text(class_name) if escape else class_name}*"


class HTMLExporter(_PageTextExporter):
    """Export OCR results to semantic HTML, with the content sorted in reading order.

    Headings map to `<h1>`/`<h2>`, list items to `<ul><li>`, recognized tables to `<table>` and
    paragraphs to `<p>` (with `<br>` between the visual lines of a paragraph). The output is a
    fragment, not a full document: it carries no doctype, `<html>` or charset declaration.

    .. warning::
        The recognized text is HTML-escaped by default. Passing ``escape=False`` interpolates the OCR
        output into the markup verbatim, so a document containing markup yields active HTML.
        Only disable escaping for output that is never rendered in a browser.

    >>> from doctr.io import HTMLExporter
    >>> html = HTMLExporter().export_page(page)
    """

    headings: ClassVar[dict[str, str]] = {"title": "h1", "section_header": "h2"}
    block_break: ClassVar[str] = "\n"
    page_break: ClassVar[str] = "\n<hr>\n"
    supports_figures: ClassVar[bool] = True
    figure_placeholder: ClassVar[str] = "<!-- image -->"

    def escape_text(self, text: str) -> str:
        return _html_escape(text, quote=False)

    def render_heading(self, norm_label: str, lines: list[str], escape: bool = True) -> str:
        tag = self.headings[norm_label]
        return f"<{tag}>{' '.join(lines)}</{tag}>"

    def render_list_item(self, lines: list[str], escape: bool = True) -> str:
        return f"<li>{' '.join(lines)}</li>"

    def render_list(self, items: list[str]) -> str:
        return "<ul>\n" + "\n".join(items) + "\n</ul>"

    def render_paragraph(self, lines: list[str], escape: bool = True) -> str:
        return "<p>" + "<br>\n".join(lines) + "</p>"

    def render_figure(self, source: str | None, caption: str | None = None, escape: bool = True) -> str:
        """Render a figure as a `<figure>` element, with its caption as `<figcaption>`"""
        if source is None:
            return self.figure_placeholder
        alt = _html_escape(caption or "", quote=True)
        figcaption = f"\n<figcaption>{self.escape_text(caption) if escape else caption}</figcaption>" if caption else ""
        return f'<figure><img src="{_html_escape(source, quote=True)}" alt="{alt}">{figcaption}</figure>'

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a table as an HTML table (first row used as header)"""
        grid = table.to_grid()
        if len(grid) == 0 or len(grid[0]) == 0:
            return ""

        def _cell(value: str, tag: str) -> str:
            content = self.escape_text(value) if escape else value
            return f"<{tag}>{content.strip()}</{tag}>"

        head = "<tr>" + "".join(_cell(value, "th") for value in grid[0]) + "</tr>"
        body = "\n".join("<tr>" + "".join(_cell(value, "td") for value in row) + "</tr>" for row in grid[1:])
        return f"<table>\n{head}\n{body}\n</table>" if body else f"<table>\n{head}\n</table>"

    def export_kie_page(self, page: "KIEPage", direction: str = "auto", escape: bool = True) -> str:
        parts: list[str] = []
        for class_name, predictions in page.predictions.items():
            if len(predictions) == 0:
                continue
            values = "\n".join(
                f"<li>{self.escape_text(prediction.value) if escape else prediction.value}</li>"
                for prediction in predictions_in_reading_order(page, predictions, direction)
            )
            header = self.escape_text(class_name) if escape else class_name
            parts.append(f"<h3>{header}</h3>\n<ul>\n{values}\n</ul>")
        return "\n".join(parts)


def _resolve_hocr_language(language: dict[str, Any]) -> str:
    """Resolve the language code to use in the hOCR export, falling back to 'en'.

    Args:
        language: the page language dictionary `{"value": str | None, "confidence": float | None}`

    Returns:
        the detected language code when available, 'en' otherwise
    """
    lang_value = language.get("value") if isinstance(language, dict) else None
    return lang_value if isinstance(lang_value, str) and len(lang_value) > 0 else "en"


def _hocr_bbox(geometry: BoundingBox, width: int, height: int) -> str:
    """Format a relative straight bounding box as an absolute hOCR `bbox` property string.

    Args:
        geometry: the relative bounding box ((xmin, ymin), (xmax, ymax))
        width: the page width in pixels
        height: the page height in pixels

    Returns:
        the hOCR `bbox` property string
    """
    (xmin, ymin), (xmax, ymax) = geometry
    return (
        f"bbox {int(round(xmin * width))} {int(round(ymin * height))} "
        f"{int(round(xmax * width))} {int(round(ymax * height))}"
    )


def _hocr_text_size(geometry: BoundingBox, height: int, dpi: int = 72) -> tuple[int, int]:
    """Estimate the hOCR `x_size` and `x_fsize` properties from the height of a relative bounding box.

    Args:
        geometry: the relative bounding box ((xmin, ymin), (xmax, ymax))
        height: the page height in pixels
        dpi: the page resolution in dots per inch, used to convert the text height to font points

    Returns:
        a tuple of the text height in pixels (`x_size`), and the estimated font size in points (`x_fsize`)
    """
    (_, ymin), (_, ymax) = geometry
    x_size = int(round((ymax - ymin) * height))
    return x_size, int(round(x_size * 72 / dpi))


class XMLExporter:
    """hOCR (XML) exporter for pages, KIE pages and documents.
    See the hOCR 1.2 specification for the XML convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

    >>> from doctr.io import XMLExporter
    >>> xml_bytes, xml_tree = XMLExporter().export_page(page)
    """

    ocr_capabilities: ClassVar[str] = "ocr_page ocr_carea ocr_par ocr_line ocrx_word ocr_photo"

    def _new_document(self, file_title: str, language: str) -> tuple[ETElement, ETElement]:
        """Create the hOCR root element with its <head>, returning the root and its <body> element."""
        root = ETElement("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": str(language)})
        head = SubElement(root, "head")
        SubElement(head, "title").text = file_title
        SubElement(head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"})
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-system", "content": f"python-doctr {doctr.__version__}"},  # type: ignore[attr-defined]
        )
        SubElement(head, "meta", attrib={"name": "ocr-capabilities", "content": self.ocr_capabilities})
        return root, SubElement(root, "body")

    def _add_table(
        self, page_div: ETElement, table: "Table", width: int, height: int, table_count: int, dpi: int = 72
    ) -> int:
        """Serialize a recognized table as an hOCR text area, with one `ocr_line` per row.

        Args:
            page_div: the `ocr_page` element the table is appended to
            table: the table to serialize
            width: the page width in pixels
            height: the page height in pixels
            table_count: the 1-based index of the table on the page
            dpi: the page resolution in dots per inch, used to estimate font sizes

        Returns:
            the index of the next table
        """
        if len(table.geometry) != 2 or any(len(cell.geometry) != 2 for cell in table.cells):
            raise TypeError("XML export is only available for straight bounding boxes for now.")
        table_bbox = _hocr_bbox(table.geometry, width, height)  # type: ignore[arg-type]
        table_div = SubElement(
            page_div, "div", attrib={"class": "ocr_carea", "id": f"table_{table_count}", "title": table_bbox}
        )
        paragraph = SubElement(
            table_div, "p", attrib={"class": "ocr_par", "id": f"table_par_{table_count}", "title": table_bbox}
        )
        rows: dict[int, list[Any]] = {}
        for cell in table.cells:
            rows.setdefault(cell.row_start, []).append(cell)
        for row_idx in sorted(rows):
            cells = sorted(rows[row_idx], key=lambda cell: cell.col_start)
            xs = [coord for cell in cells for coord in (cell.geometry[0][0], cell.geometry[1][0])]
            ys = [coord for cell in cells for coord in (cell.geometry[0][1], cell.geometry[1][1])]
            row_geometry = ((min(xs), min(ys)), (max(xs), max(ys)))
            row_bbox = _hocr_bbox(row_geometry, width, height)
            row_x_size, row_x_fsize = _hocr_text_size(row_geometry, height, dpi)
            line_span = SubElement(
                paragraph,
                "span",
                attrib={
                    "class": "ocr_line",
                    "id": f"table_{table_count}_row_{row_idx + 1}",
                    "title": (
                        f"{row_bbox}; baseline 0 0; x_size {row_x_size}; x_fsize {row_x_fsize}; "
                        "x_descenders 0; x_ascenders 0"
                    ),
                },
            )
            for col_idx, cell in enumerate(cells):
                cell_span = SubElement(
                    line_span,
                    "span",
                    attrib={
                        "class": "ocrx_word",
                        "id": f"table_{table_count}_cell_{row_idx + 1}_{col_idx + 1}",
                        "title": (
                            f"{_hocr_bbox(cell.geometry, width, height)}; x_wconf {int(round(cell.confidence * 100))}"
                        ),
                    },
                )
                cell_span.text = cell.value
        return table_count + 1

    def _add_figure(
        self, page_div: ETElement, region: "LayoutElement", width: int, height: int, figure_count: int
    ) -> int:
        """Add a figure as an hOCR `ocr_photo` area, with its enclosing box

        Args:
            page_div: the `ocr_page` element
            region: the figure region
            width: page width in pixels
            height: page height in pixels
            figure_count: 1-based index of the figure on the page

        Returns:
            the index of the next figure
        """
        xmin, ymin, xmax, ymax = _xyxy(region.geometry)
        SubElement(
            page_div,
            "div",
            attrib={
                "class": "ocr_photo",
                "id": f"figure_{figure_count}",
                "title": _hocr_bbox(((xmin, ymin), (xmax, ymax)), width, height),
            },
        )
        return figure_count + 1

    def export_page(
        self,
        page: "Page",
        file_title: str = "docTR - XML export (hOCR)",
        direction: str = "auto",
        reading_order: bool = True,
        dpi: int = 72,
    ) -> tuple[bytes, ET.ElementTree]:
        """Export a page as hOCR XML, with its content sorted in reading order.

        Figures are exported as `ocr_photo` areas.

        Args:
            page: the page to export
            file_title: the title of the XML file
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            reading_order: whether the content should be linearized in reading order. Pass False to serialize
                `page.blocks` then `page.tables` in their raw order.
            dpi: the page resolution in dots per inch, used to estimate font sizes (`x_size`, `x_fsize`)

        Returns:
            a tuple of the XML byte string, and its ElementTree
        """
        from doctr.io.elements import LayoutElement, Table

        block_count: int = 1
        line_count: int = 1
        word_count: int = 1
        table_count: int = 1
        figure_count: int = 1
        height, width = page.dimensions
        page_hocr, body = self._new_document(file_title, _resolve_hocr_language(page.language))
        page_div = SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": f"page_{page.page_idx + 1}",
                "title": f"image; bbox 0 0 {width} {height}; ppageno 0",
            },
        )
        auto = direction == "auto"
        if reading_order:
            items, _, direction = page_reading_order(page, direction, include_figures=True)
        else:
            items = [*page.blocks, *page.tables, *picture_regions(page)]
        # iterate over the blocks / lines / words and create the XML elements line by line with the attributes
        for item in items:
            if isinstance(item, Table):
                table_count = self._add_table(page_div, item, width, height, table_count, dpi=dpi)
                continue
            if isinstance(item, LayoutElement):
                figure_count = self._add_figure(page_div, item, width, height, figure_count)
                continue
            block = item
            if len(block.geometry) != 2:
                raise TypeError("XML export is only available for straight bounding boxes for now.")
            block_bbox = _hocr_bbox(block.geometry, width, height)
            block_div = SubElement(
                page_div,
                "div",
                attrib={"class": "ocr_carea", "id": f"block_{block_count}", "title": block_bbox},
            )
            paragraph = SubElement(
                block_div,
                "p",
                attrib={"class": "ocr_par", "id": f"par_{block_count}", "title": block_bbox},
            )
            block_count += 1
            for line in block.lines:
                # NOTE: baseline, x_descenders, x_ascenders are currently initialized to 0,
                # while x_size and x_fsize are estimated from the line box height
                x_size, x_fsize = _hocr_text_size(line.geometry, height, dpi)
                line_span = SubElement(
                    paragraph,
                    "span",
                    attrib={
                        "class": "ocr_line",
                        "id": f"line_{line_count}",
                        "title": (
                            f"{_hocr_bbox(line.geometry, width, height)}; "
                            f"baseline 0 0; x_size {x_size}; x_fsize {x_fsize}; x_descenders 0; x_ascenders 0"
                        ),
                    },
                )
                line_count += 1
                for word in ordered_line_words(line, direction, auto):
                    word_div = SubElement(
                        line_span,
                        "span",
                        attrib={
                            "class": "ocrx_word",
                            "id": f"word_{word_count}",
                            "title": (
                                f"{_hocr_bbox(word.geometry, width, height)}; "
                                f"x_wconf {int(round(word.confidence * 100))}"
                            ),
                        },
                    )
                    word_div.text = word.value
                    word_count += 1
        return ET.tostring(page_hocr, encoding="utf-8", method="xml"), ET.ElementTree(page_hocr)

    def export_kie_page(
        self,
        page: "KIEPage",
        file_title: str = "docTR - XML export (hOCR)",
        direction: str = "auto",
        reading_order: bool = True,
        dpi: int = 72,
    ) -> tuple[bytes, ET.ElementTree]:
        """Export a KIE page as hOCR XML, with the predictions of each class sorted in reading order.

        Args:
            page: the KIE page to export
            file_title: the title of the XML file
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            reading_order: whether the predictions of each class should be sorted in reading order
            dpi: the page resolution in dots per inch, used to estimate font sizes (`x_size`, `x_fsize`)

        Returns:
            a tuple of the XML byte string, and its ElementTree
        """
        prediction_count: int = 1
        height, width = page.dimensions
        page_hocr, body = self._new_document(file_title, _resolve_hocr_language(page.language))
        page_div = SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": f"page_{page.page_idx + 1}",
                "title": f"image; bbox 0 0 {width} {height}; ppageno 0",
            },
        )
        # iterate over the predictions and create the XML elements line by line with the attributes
        for class_name, predictions in page.predictions.items():
            ordered = predictions_in_reading_order(page, predictions, direction) if reading_order else predictions
            for prediction in ordered:
                if len(prediction.geometry) != 2:
                    raise TypeError("XML export is only available for straight bounding boxes for now.")
                prediction_bbox = _hocr_bbox(prediction.geometry, width, height)  # type: ignore[arg-type]
                x_size, x_fsize = _hocr_text_size(prediction.geometry, height, dpi)  # type: ignore[arg-type]
                prediction_div = SubElement(
                    page_div,
                    "div",
                    attrib={
                        "class": "ocr_carea",
                        "id": f"{class_name}_prediction_{prediction_count}",
                        "title": prediction_bbox,
                    },
                )
                # NOTE: ocr_par, ocr_line and ocrx_word are the same because the KIE predictions contain only words
                # This is a workaround to make it PDF/A compatible
                par_div = SubElement(
                    prediction_div,
                    "p",
                    attrib={
                        "class": "ocr_par",
                        "id": f"{class_name}_par_{prediction_count}",
                        "title": prediction_bbox,
                    },
                )
                line_span = SubElement(
                    par_div,
                    "span",
                    attrib={
                        "class": "ocr_line",
                        "id": f"{class_name}_line_{prediction_count}",
                        "title": (
                            f"{prediction_bbox}; baseline 0 0; x_size {x_size}; x_fsize {x_fsize}; "
                            "x_descenders 0; x_ascenders 0"
                        ),
                    },
                )
                word_div = SubElement(
                    line_span,
                    "span",
                    attrib={
                        "class": "ocrx_word",
                        "id": f"{class_name}_word_{prediction_count}",
                        "title": f"{prediction_bbox}; x_wconf {int(round(prediction.confidence * 100))}",
                    },
                )
                word_div.text = prediction.value
                prediction_count += 1
        return ET.tostring(page_hocr, encoding="utf-8", method="xml"), ET.ElementTree(page_hocr)

    def export_document(self, document: Any, **kwargs: Any) -> list[tuple[bytes, ET.ElementTree]]:
        """Export a document as a list of hOCR pages.

        Args:
            document: the document to export
            **kwargs: additional keyword arguments passed to the page export

        Returns:
            list of tuple of (bytes, ElementTree), one per page
        """
        from doctr.io.elements import KIEPage

        return [
            self.export_kie_page(page, **kwargs) if isinstance(page, KIEPage) else self.export_page(page, **kwargs)
            for page in document.pages
        ]


class PageExportsMixin:
    """Export functionality of a :class:`~doctr.io.elements.Page`"""

    if TYPE_CHECKING:  # structural attributes provided by the element class
        page: np.ndarray
        blocks: list["Block"]
        page_idx: int
        dimensions: tuple[int, int]
        orientation: dict[str, Any]
        language: dict[str, Any]
        layout: list[Any]
        tables: list["Table"]

    def render(self, block_break: str = "\n\n", direction: str = "auto", include_furniture: bool = True) -> str:
        """Renders the full text of the page, with its content sorted in reading order.

        Args:
            block_break: the string inserted between two blocks
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            include_furniture: whether page headers, page footers and footnotes should be included

        Returns:
            the text of the page
        """
        return TextExporter().export_page(
            cast("Page", self), direction=direction, include_furniture=include_furniture, block_break=block_break
        )

    def export(self, reading_order: bool = True) -> dict[str, Any]:
        """Export the page into a nested dict, with its content sorted in reading order.

        Args:
            reading_order: whether the blocks should be linearized in reading order, exactly like the
                Markdown / HTML / AsciiDoc / hOCR exports. Pass False to serialize `page.blocks` as stored.

        Returns:
            a JSON-serializable dict
        """
        from doctr.io.elements import Block, Element

        export_dict = Element.export(cast("Element", self))
        if reading_order:
            # Tables and layout regions have their own export keys
            blocks = [item for item in page_reading_order(cast("Page", self))[0] if isinstance(item, Block)]
            if blocks:  # an empty linearization (no line on the page) leaves the stored blocks untouched
                export_dict["blocks"] = [block.export() for block in blocks]
        return export_dict

    def export_as_xml(
        self,
        file_title: str = "docTR - XML export (hOCR)",
        direction: str = "auto",
        reading_order: bool = True,
        dpi: int = 72,
    ) -> tuple[bytes, ET.ElementTree]:
        """Export the page as XML (hOCR-format), with its content sorted in reading order
        convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

        Args:
            file_title: the title of the XML file
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            reading_order: whether the content should be linearized in reading order
            dpi: the page resolution in dots per inch, used to estimate font sizes (`x_size`, `x_fsize`)

        Returns:
            a tuple of the XML byte string, and its ElementTree
        """
        return XMLExporter().export_page(
            cast("Page", self), file_title=file_title, direction=direction, reading_order=reading_order, dpi=dpi
        )

    def items_in_reading_order(
        self, direction: str = "auto", include_figures: bool = False
    ) -> list["Block | Table | LayoutElement"]:
        """Return the content of the page (blocks, tables and optionally figures) sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            include_figures: whether to return the figure regions too

        Returns:
            list of blocks, tables and, if requested, figures in reading order
        """
        return page_reading_order(cast("Page", self), direction, include_figures=include_figures)[0]

    def export_as_markdown(
        self,
        direction: str = "auto",
        escape: bool = True,
        include_furniture: bool = True,
        images: "str | FigureEncoder | None" = "placeholder",
    ) -> str:
        """Export the page as Markdown, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters carrying a structural meaning in Markdown should be escaped
            include_furniture: whether page headers, page footers and footnotes should be included
            images: how to render the figures: 'none', 'placeholder', 'embedded', or a
                :class:`~doctr.io.FigureEncoder` (required for 'referenced')

        Returns:
            a Markdown string
        """
        return MarkdownExporter().export_page(
            cast("Page", self),
            direction=direction,
            escape=escape,
            include_furniture=include_furniture,
            images=images,
        )

    def export_as_asciidoc(
        self,
        direction: str = "auto",
        escape: bool = True,
        include_furniture: bool = True,
        images: "str | FigureEncoder | None" = "placeholder",
    ) -> str:
        """Export the page as AsciiDoc, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters and line markers carrying a structural meaning in AsciiDoc should
                be escaped
            include_furniture: whether page headers, page footers and footnotes should be included
            images: how to render the figures: 'none', 'placeholder', 'embedded', or a
                :class:`~doctr.io.FigureEncoder` (required for 'referenced')

        Returns:
            an AsciiDoc string
        """
        return AsciiDocExporter().export_page(
            cast("Page", self),
            direction=direction,
            escape=escape,
            include_furniture=include_furniture,
            images=images,
        )

    def export_as_html(
        self,
        direction: str = "auto",
        include_furniture: bool = True,
        images: "str | FigureEncoder | None" = "placeholder",
    ) -> str:
        """Export the page as semantic HTML, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            include_furniture: whether page headers, page footers and footnotes should be included
            images: how to render the figures: 'none', 'placeholder', 'embedded', or a
                :class:`~doctr.io.FigureEncoder` (required for 'referenced')

        Returns:
            an HTML string
        """
        return HTMLExporter().export_page(
            cast("Page", self), direction=direction, include_furniture=include_furniture, images=images
        )

    def export_as(self, format: str, **kwargs: Any) -> Any:
        """Export the page in the requested format.

        Args:
            format: one of 'markdown'/'md', 'asciidoc'/'adoc', 'html', 'text'/'txt', 'json'/'dict',
                'xml'/'hocr'
            **kwargs: additional keyword arguments passed to the format-specific export method

        Returns:
            the exported page
        """
        exporters: dict[str, Any] = {
            "markdown": self.export_as_markdown,
            "md": self.export_as_markdown,
            "asciidoc": self.export_as_asciidoc,
            "adoc": self.export_as_asciidoc,
            "html": self.export_as_html,
            "text": self.render,
            "txt": self.render,
            "json": self.export,
            "dict": self.export,
            "xml": self.export_as_xml,
            "hocr": self.export_as_xml,
        }
        return _export_as(exporters, format, **kwargs)


class KIEPageExportsMixin:
    """Export functionality of a :class:`~doctr.io.elements.KIEPage`"""

    if TYPE_CHECKING:  # structural attributes provided by the element class
        page: np.ndarray
        predictions: dict[str, list[Any]]
        page_idx: int
        dimensions: tuple[int, int]
        orientation: dict[str, Any]
        language: dict[str, Any]

    def export(self, reading_order: bool = True) -> dict[str, Any]:
        """Export the KIE page into a nested dict, with the predictions of each class in reading order.

        Args:
            reading_order: whether the predictions of each class should be sorted in reading order

        Returns:
            a JSON-serializable dict
        """
        from doctr.io.elements import Element

        export_dict = Element.export(cast("Element", self))
        if reading_order:
            export_dict["predictions"] = {
                class_name: [
                    prediction.export()
                    for prediction in predictions_in_reading_order(cast("KIEPage", self), predictions)
                ]
                for class_name, predictions in self.predictions.items()
            }
        return export_dict

    def render(self, prediction_break: str = "\n\n", direction: str = "auto") -> str:
        """Renders the full text of the page, with the predictions of each class sorted in reading order.

        Args:
            prediction_break: the string inserted between two predictions
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'

        Returns:
            the text of the page, one section per detection class with its predictions in reading order
        """
        parts: list[str] = []
        for class_name, predictions in self.predictions.items():
            parts.extend(
                f"{class_name}: {prediction.render()}"
                for prediction in predictions_in_reading_order(cast("KIEPage", self), predictions, direction)
            )
        return prediction_break.join(parts)

    def export_as_xml(
        self,
        file_title: str = "docTR - XML export (hOCR)",
        direction: str = "auto",
        reading_order: bool = True,
        dpi: int = 72,
    ) -> tuple[bytes, ET.ElementTree]:
        """Export the page as XML (hOCR-format), with the predictions of each class in reading order
        convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

        Args:
            file_title: the title of the XML file
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            reading_order: whether the predictions of each class should be sorted in reading order
            dpi: the page resolution in dots per inch, used to estimate font sizes (`x_size`, `x_fsize`)

        Returns:
            a tuple of the XML byte string, and its ElementTree
        """
        return XMLExporter().export_kie_page(
            cast("KIEPage", self), file_title=file_title, direction=direction, reading_order=reading_order, dpi=dpi
        )

    def export_as_markdown(self, direction: str = "auto", escape: bool = True) -> str:
        """Export the KIE page as Markdown, with the predictions of each class sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters carrying a structural meaning in Markdown should be escaped

        Returns:
            a Markdown string with one section per detection class
        """
        return MarkdownExporter().export_kie_page(cast("KIEPage", self), direction=direction, escape=escape)

    def export_as_asciidoc(self, direction: str = "auto", escape: bool = True) -> str:
        """Export the KIE page as AsciiDoc, with the predictions of each class sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters and line markers carrying a structural meaning in AsciiDoc should
                be escaped

        Returns:
            an AsciiDoc string with one section per detection class
        """
        return AsciiDocExporter().export_kie_page(cast("KIEPage", self), direction=direction, escape=escape)

    def export_as_html(self, direction: str = "auto") -> str:
        """Export the KIE page as semantic HTML, with the predictions of each class sorted in reading order"""
        return HTMLExporter().export_kie_page(cast("KIEPage", self), direction=direction)

    def export_as(self, format: str, **kwargs: Any) -> Any:
        """Export the KIE page in the requested format ('markdown'/'md', 'asciidoc'/'adoc', 'html',
        'text'/'txt', 'json'/'dict', 'xml'/'hocr')."""
        exporters: dict[str, Any] = {
            "markdown": self.export_as_markdown,
            "md": self.export_as_markdown,
            "asciidoc": self.export_as_asciidoc,
            "adoc": self.export_as_asciidoc,
            "html": self.export_as_html,
            "text": self.render,
            "txt": self.render,
            "json": self.export,
            "dict": self.export,
            "xml": self.export_as_xml,
            "hocr": self.export_as_xml,
        }
        return _export_as(exporters, format, **kwargs)


class DocumentExportsMixin:
    """Export functionality of a :class:`~doctr.io.elements.Document` (also used by `KIEDocument`)"""

    if TYPE_CHECKING:  # structural attributes provided by the element class
        pages: list[Any]
        _exported_keys: list[str]

    def render(self, page_break: str = "\n\n\n\n", **kwargs: Any) -> str:
        """Renders the full text of the document, with the content of each page sorted in reading order.

        Args:
            page_break: the string inserted between two pages
            **kwargs: additional keyword arguments passed to the `Page.render` / `KIEPage.render` method

        Returns:
            the text of the document
        """
        return page_break.join(page.render(**kwargs) for page in self.pages)

    def export(self, reading_order: bool = True) -> dict[str, Any]:
        """Export the document into a nested dict, with the content of each page sorted in reading order.

        Args:
            reading_order: whether the content of each page should be linearized in reading order

        Returns:
            a JSON-serializable dict
        """
        export_dict: dict[str, Any] = {key: to_json_safe(getattr(self, key)) for key in self._exported_keys}
        export_dict["pages"] = [page.export(reading_order=reading_order) for page in self.pages]
        return export_dict

    def export_as_xml(self, **kwargs: Any) -> list[tuple[bytes, ET.ElementTree]]:
        """Export the document as XML (hOCR-format)

        Args:
            **kwargs: additional keyword arguments passed to the XML page export

        Returns:
            list of tuple of (bytes, ElementTree)
        """
        return XMLExporter().export_document(self, **kwargs)

    def export_as_markdown(self, page_break: str = "\n\n---\n\n", **kwargs: Any) -> str:
        """Export the document as Markdown, with the content of each page sorted in reading order.

        Args:
            page_break: the string inserted between two pages (a thematic break by default)
            **kwargs: additional keyword arguments passed to the `Page.export_as_markdown` method (e.g. `images`,
                not supported by KIE pages)

        Returns:
            a Markdown string
        """
        return page_break.join(page.export_as_markdown(**kwargs) for page in self.pages)

    def export_as_asciidoc(self, page_break: str = "\n\n<<<\n\n", **kwargs: Any) -> str:
        """Export the document as AsciiDoc, with the content of each page sorted in reading order.

        Args:
            page_break: the string inserted between two pages (an AsciiDoc page break by default)
            **kwargs: additional keyword arguments passed to the `Page.export_as_asciidoc` method (e.g. `images`,
                not supported by KIE pages)

        Returns:
            an AsciiDoc string
        """
        return page_break.join(page.export_as_asciidoc(**kwargs) for page in self.pages)

    def export_as_html(self, page_break: str = "<hr>", **kwargs: Any) -> str:
        """Export the document as semantic HTML, with the content of each page sorted in reading order.

        Args:
            page_break: the HTML snippet inserted between two pages
            **kwargs: additional keyword arguments passed to the page export (e.g. `images`, not supported by
                KIE pages)

        Returns:
            an HTML string
        """
        return page_break.join(page.export_as_html(**kwargs) for page in self.pages)

    def export_as(self, format: str, **kwargs: Any) -> Any:
        """Export the document in the requested format ('markdown'/'md', 'asciidoc'/'adoc', 'html',
        'text'/'txt', 'json'/'dict', 'xml'/'hocr')."""
        exporters: dict[str, Any] = {
            "markdown": self.export_as_markdown,
            "md": self.export_as_markdown,
            "asciidoc": self.export_as_asciidoc,
            "adoc": self.export_as_asciidoc,
            "html": self.export_as_html,
            "text": self.render,
            "txt": self.render,
            "json": self.export,
            "dict": self.export,
            "xml": self.export_as_xml,
            "hocr": self.export_as_xml,
        }
        return _export_as(exporters, format, **kwargs)
