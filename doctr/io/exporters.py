# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from html import escape as _html_escape
from typing import TYPE_CHECKING, Any, ClassVar, cast
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element as ETElement
from xml.etree.ElementTree import SubElement

import numpy as np

import doctr
from doctr.utils.common_types import BoundingBox

if TYPE_CHECKING:  # pragma: no cover
    from doctr.io.elements import Block, KIEPage, Line, Page, Table

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
_ADOC_SPECIAL_CHARS = "\\`*_#^~|+{}<>"
_ADOC_LINE_MARKERS = "=*.-/+"


def _covering_region_indices(geoms: list[Any], region_geoms: list[Any], min_coverage: float = 0.5) -> list[int]:
    """For each element geometry, the index of the layout region covering the largest share of its area.

    Uses the same area-coverage criterion as :func:`doctr.models.reading_order.assign_layout_labels`, and
    returns -1 when no region covers the element by at least `min_coverage`. The geometries are expected to
    be in the same (upright) frame.
    """
    from doctr.models.reading_order.base import _to_boxes

    if len(region_geoms) == 0 or len(geoms) == 0:
        return [-1] * len(geoms)
    boxes, regions = _to_boxes(geoms), _to_boxes(region_geoms)
    inter_w = np.minimum(boxes[:, None, 2], regions[None, :, 2]) - np.maximum(boxes[:, None, 0], regions[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 3], regions[None, :, 3]) - np.maximum(boxes[:, None, 1], regions[None, :, 1])
    inter = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)
    areas = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9, None)
    coverage = inter / areas[:, None]
    best = coverage.argmax(axis=1)
    return [int(reg) if coverage[i, reg] >= min_coverage else -1 for i, reg in enumerate(best)]


def _reading_order_signature(page: "Page", direction: str) -> tuple[Any, ...]:
    """A cheap structural fingerprint of a page, used to invalidate the reading-order cache.

    Covers the requested direction and the identity (plus line count) of every block and table, so
    replacing or re-grouping the page content invalidates the cache. In-place edits to a `Line`'s words
    are not detected; callers mutating a page that deeply should drop `_reading_order_cache` themselves.
    """
    return (
        direction,
        tuple((id(block), len(block.lines)) for block in page.blocks),
        tuple(id(table) for table in getattr(page, "tables", ()) or ()),
    )


def _store_reading_order(page: "Page", signature: tuple[Any, ...], result: tuple[Any, ...]) -> None:
    """Memoize a reading-order result on the page, ignoring pages that reject attribute assignment."""
    try:
        page._reading_order_cache = (signature, result)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        pass


def page_reading_order(page: "Page", direction: str = "auto") -> tuple[list[Any], list[str | None], str]:
    """Linearize the content of a page (blocks & tables) in reading order.

    The result is memoized on the page: every exporter calls this, so a page exported to several formats
    (or built with `keep_reading_order=True` and then exported) orders its content once.

    Args:
        page: the page to linearize
        direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'

    Returns:
        a tuple with the ordered items (blocks & tables), their layout label (None without layout) and the
        effective reading direction
    """
    from doctr.io.elements import Block, Table
    from doctr.models.reading_order import (
        ReadingOrderPredictor,
        assign_layout_labels,
        deskew_reading_geometries,
        normalize_layout_label,
        resolve_reading_segments,
    )

    signature = _reading_order_signature(page, direction)
    cached = getattr(page, "_reading_order_cache", None)
    if cached is not None and cached[0] == signature:
        items, labels, resolved = cached[1]
        return list(items), list(labels), resolved

    texts = [word.value for block in page.blocks for line in block.lines for word in line.words]
    language = page.language.get("value") if isinstance(page.language, dict) else None
    direction = ReadingOrderPredictor(direction=direction).resolve_direction(texts, language=language)
    region_geoms = [region.geometry for region in page.layout]
    region_labels = [region.type for region in page.layout]

    lines = [line for block in page.blocks for line in block.lines]
    elements: list[Any] = [*lines, *page.tables]
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
    if len(region_geoms) > 0:
        elt_labels = assign_layout_labels(elt_geoms, region_geoms, region_labels)
    elt_labels = ["Table" if isinstance(elt, Table) else label for elt, label in zip(elements, elt_labels)]
    segments = resolve_reading_segments(elt_geoms, direction=direction, labels=elt_labels)

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
    region_idx = _covering_region_indices(elt_geoms, region_geoms) if len(region_geoms) > 0 else [-1] * len(elements)
    open_list_region: int | None = None  # region of the list bullet currently being built (None outside a list)
    for segment in segments:
        first = elements[segment[0]]
        seg_label = elt_labels[segment[0]]
        if isinstance(first, Table):
            items.append(first)
            labels.append("Table")
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
    return items, labels, direction


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


class _PageTextExporter:
    """Shared logic of the reading-order-aware text exporters.

    Subclasses define the format specifics: heading prefixes (per normalized layout label), the bullet
    prefix, character escaping, line finalization (neutralizing markers a line must not start with) and the
    table rendering.
    """

    headings: ClassVar[dict[str, str]] = {}
    bullet: ClassVar[str] = "- "
    block_break: ClassVar[str] = "\n\n"
    page_break: ClassVar[str] = "\n\n"

    def escape_text(self, text: str) -> str:
        """Escape the characters carrying a structural meaning in the target format"""
        return text

    def finalize_line(self, line: str) -> str:
        """Neutralize the block-level markers a line must not start with in the target format"""
        return line

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a recognized table in the target format"""
        raise NotImplementedError

    def class_header(self, class_name: str, escape: bool = True) -> str:
        """Render the header of a detection class in a KIE export"""
        raise NotImplementedError

    def _line_text(self, line: "Line", direction: str, escape: bool) -> str:
        """Render the text of a line, ordering the words according to the reading direction."""
        text = " ".join(word.render() for word in ordered_line_words(line, direction))
        return self.escape_text(text) if escape else text

    def export_page(
        self,
        page: "Page",
        direction: str = "auto",
        escape: bool = True,
        include_furniture: bool = True,
        block_break: str | None = None,
    ) -> str:
        """Export a page, with its content sorted in reading order.

        Args:
            page: the page to export
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters or markers carrying a structural meaning should be neutralized
            include_furniture: whether page headers, page footers and footnotes should be included
            block_break: the string inserted between two blocks (the format-specific default when None)

        Returns:
            the exported page as a string
        """
        from doctr.io.elements import Table
        from doctr.models.reading_order import layout_label_role, normalize_layout_label

        auto = direction == "auto"
        items, labels, direction = page_reading_order(page, direction)
        parts: list[str] = []
        list_group: list[str] = []

        def _flush_list() -> None:
            if list_group:
                parts.append("\n".join(list_group))
                list_group.clear()

        for item, label in zip(items, labels):
            if not include_furniture and layout_label_role(label) in ("header", "footer", "footnote"):
                continue
            if isinstance(item, Table):
                _flush_list()
                rendered = self.render_table(item, escape=escape)
                if rendered:
                    parts.append(rendered)
                continue
            item_lines = [
                self._line_text(line, _line_render_direction(line, direction, auto), escape) for line in item.lines
            ]
            item_lines = [line for line in item_lines if line.strip()]
            if len(item_lines) == 0:
                continue
            norm_label = normalize_layout_label(label)
            if norm_label in self.headings:
                _flush_list()
                parts.append(self.headings[norm_label] + " ".join(item_lines))
            elif norm_label in _LIST_LABELS:
                # A list item (possibly wrapped over several lines) renders as a single bullet
                text = " ".join(item_lines)
                list_group.append(self.bullet + (self.finalize_line(text) if escape else text))
            else:
                _flush_list()
                parts.append("\n".join(self.finalize_line(line) if escape else line for line in item_lines))
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

    def escape_text(self, text: str) -> str:
        return "".join(f"\\{char}" if char in _MD_SPECIAL_CHARS else char for char in text)

    def finalize_line(self, line: str) -> str:
        stripped = line.lstrip()
        if stripped and (stripped[0] in _MD_LINE_MARKERS or stripped.split(" ")[0].rstrip(".").isdigit()):
            return f"\\{line}" if line[0] != "\\" else line
        return line

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

    def escape_text(self, text: str) -> str:
        return _html_escape(text, quote=False)

    def export_page(
        self,
        page: "Page",
        direction: str = "auto",
        escape: bool = True,
        include_furniture: bool = True,
        block_break: str | None = None,
    ) -> str:
        from doctr.io.elements import Table
        from doctr.models.reading_order import layout_label_role, normalize_layout_label

        auto = direction == "auto"
        items, labels, direction = page_reading_order(page, direction)
        parts: list[str] = []
        list_group: list[str] = []

        def _flush_list() -> None:
            if list_group:
                parts.append("<ul>\n" + "\n".join(list_group) + "\n</ul>")
                list_group.clear()

        for item, label in zip(items, labels):
            if not include_furniture and layout_label_role(label) in ("header", "footer", "footnote"):
                continue
            if isinstance(item, Table):
                _flush_list()
                rendered = self.render_table(item, escape=escape)
                if rendered:
                    parts.append(rendered)
                continue
            item_lines = [
                self._line_text(line, _line_render_direction(line, direction, auto), escape) for line in item.lines
            ]
            item_lines = [line for line in item_lines if line.strip()]
            if len(item_lines) == 0:
                continue
            norm_label = normalize_layout_label(label)
            if norm_label in self.headings:
                _flush_list()
                tag = self.headings[norm_label]
                parts.append(f"<{tag}>{' '.join(item_lines)}</{tag}>")
            elif norm_label in _LIST_LABELS:
                list_group.append(f"<li>{' '.join(item_lines)}</li>")
            else:
                _flush_list()
                parts.append("<p>" + "<br>\n".join(item_lines) + "</p>")
        _flush_list()
        return (self.block_break if block_break is None else block_break).join(parts)

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

    ocr_capabilities: ClassVar[str] = "ocr_page ocr_carea ocr_par ocr_line ocrx_word"

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

    def export_page(
        self,
        page: "Page",
        file_title: str = "docTR - XML export (hOCR)",
        direction: str = "auto",
        reading_order: bool = True,
        dpi: int = 72,
    ) -> tuple[bytes, ET.ElementTree]:
        """Export a page as hOCR XML, with its content sorted in reading order.

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
        from doctr.io.elements import Table

        block_count: int = 1
        line_count: int = 1
        word_count: int = 1
        table_count: int = 1
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
            items, _, direction = page_reading_order(page, direction)
        else:
            items = [*page.blocks, *page.tables]
        # iterate over the blocks / lines / words and create the XML elements line by line with the attributes
        for item in items:
            if isinstance(item, Table):
                table_count = self._add_table(page_div, item, width, height, table_count, dpi=dpi)
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
        SubElement(
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
                    body,
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
        from doctr.io.elements import Element, Table

        export_dict = Element.export(cast("Element", self))
        if reading_order:
            blocks = [item for item in page_reading_order(cast("Page", self))[0] if not isinstance(item, Table)]
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

    def items_in_reading_order(self, direction: str = "auto") -> list["Block | Table"]:
        """Return the content of the page (blocks & tables) sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'

        Returns:
            list of blocks & tables in reading order
        """
        return page_reading_order(cast("Page", self), direction)[0]

    def export_as_markdown(self, direction: str = "auto", escape: bool = True, include_furniture: bool = True) -> str:
        """Export the page as Markdown, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters carrying a structural meaning in Markdown should be escaped
            include_furniture: whether page headers, page footers and footnotes should be included

        Returns:
            a Markdown string
        """
        return MarkdownExporter().export_page(
            cast("Page", self), direction=direction, escape=escape, include_furniture=include_furniture
        )

    def export_as_asciidoc(self, direction: str = "auto", escape: bool = True, include_furniture: bool = True) -> str:
        """Export the page as AsciiDoc, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters and line markers carrying a structural meaning in AsciiDoc should
                be escaped
            include_furniture: whether page headers, page footers and footnotes should be included

        Returns:
            an AsciiDoc string
        """
        return AsciiDocExporter().export_page(
            cast("Page", self), direction=direction, escape=escape, include_furniture=include_furniture
        )

    def export_as_html(self, direction: str = "auto", include_furniture: bool = True) -> str:
        """Export the page as semantic HTML, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            include_furniture: whether page headers, page footers and footnotes should be included

        Returns:
            an HTML string
        """
        return HTMLExporter().export_page(cast("Page", self), direction=direction, include_furniture=include_furniture)

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
            **kwargs: additional keyword arguments passed to the `Page.export_as_markdown` method

        Returns:
            a Markdown string
        """
        return page_break.join(page.export_as_markdown(**kwargs) for page in self.pages)

    def export_as_asciidoc(self, page_break: str = "\n\n<<<\n\n", **kwargs: Any) -> str:
        """Export the document as AsciiDoc, with the content of each page sorted in reading order.

        Args:
            page_break: the string inserted between two pages (an AsciiDoc page break by default)
            **kwargs: additional keyword arguments passed to the `Page.export_as_asciidoc` method

        Returns:
            an AsciiDoc string
        """
        return page_break.join(page.export_as_asciidoc(**kwargs) for page in self.pages)

    def export_as_html(self, page_break: str = "<hr>", **kwargs: Any) -> str:
        """Export the document as semantic HTML, with the content of each page sorted in reading order.

        Args:
            page_break: the HTML snippet inserted between two pages
            **kwargs: additional keyword arguments passed to the page export

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
