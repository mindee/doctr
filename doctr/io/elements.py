# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element as ETElement
from xml.etree.ElementTree import SubElement

import numpy as np

import doctr
from doctr.file_utils import requires_package
from doctr.utils.common_types import BoundingBox
from doctr.utils.geometry import resolve_enclosing_bbox, resolve_enclosing_rbbox
from doctr.utils.reconstitution import synthesize_kie_page, synthesize_page
from doctr.utils.repr import NestedObject

try:  # optional dependency for visualization
    from doctr.utils.visualization import visualize_kie_page, visualize_page
except ModuleNotFoundError:
    pass

__all__ = [
    "Element",
    "Word",
    "Artefact",
    "Line",
    "Prediction",
    "Block",
    "Page",
    "KIEPage",
    "KIEDocument",
    "Document",
    "LayoutElement",
    "TableCell",
    "Table",
]


class Element(NestedObject):
    """Implements an abstract document element with exporting and text rendering capabilities"""

    _children_names: list[str] = []
    _exported_keys: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self._children_names:
                setattr(self, k, v)
            else:
                raise KeyError(f"{self.__class__.__name__} object does not have any attribute named '{k}'")

    def export(self) -> dict[str, Any]:
        """Exports the object into a nested dict format"""
        export_dict = {k: getattr(self, k) for k in self._exported_keys}
        for children_name in self._children_names:
            if children_name in ["predictions"]:
                export_dict[children_name] = {
                    k: [item.export() for item in c] for k, c in getattr(self, children_name).items()
                }
            else:
                export_dict[children_name] = [c.export() for c in getattr(self, children_name)]

        return export_dict

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


class Word(Element):
    """Implements a word element

    Args:
        value: the text string of the word
        confidence: the confidence associated with the text prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size
        objectness_score: the objectness score of the detection
        crop_orientation: the general orientation of the crop in degrees and its confidence
    """

    _exported_keys: list[str] = ["value", "confidence", "geometry", "objectness_score", "crop_orientation"]
    _children_names: list[str] = []

    def __init__(
        self,
        value: str,
        confidence: float,
        geometry: BoundingBox | np.ndarray,
        objectness_score: float,
        crop_orientation: dict[str, Any],
    ) -> None:
        super().__init__()
        self.value = value
        self.confidence = confidence
        self.geometry = geometry
        self.objectness_score = objectness_score
        self.crop_orientation = crop_orientation

    def render(self) -> str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) -> str:
        return f"value='{self.value}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


class Artefact(Element):
    """Implements a non-textual element

    Args:
        artefact_type: the type of artefact
        confidence: the confidence of the type prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size.
    """

    _exported_keys: list[str] = ["geometry", "type", "confidence"]
    _children_names: list[str] = []

    def __init__(self, artefact_type: str, confidence: float, geometry: BoundingBox) -> None:
        super().__init__()
        self.geometry = geometry
        self.type = artefact_type
        self.confidence = confidence

    def render(self) -> str:
        """Renders the region as a tag"""
        return f"<[{self.type.upper()}]>"

    def extra_repr(self) -> str:
        return f"type='{self.type}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


class LayoutElement(Element):
    """Implements a layout region predicted by a layout detection model

    Args:
        layout_type: the predicted region class (e.g. 'Title', 'Text', 'Table', 'Page-header')
        confidence: the confidence of the region prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size
    """

    _exported_keys: list[str] = ["geometry", "type", "confidence"]
    _children_names: list[str] = []

    def __init__(self, layout_type: str, confidence: float, geometry: BoundingBox | np.ndarray) -> None:
        super().__init__()
        self.geometry = geometry
        self.type = layout_type
        self.confidence = confidence

    def render(self) -> str:
        """Renders the region as a tag"""
        return f"<[{self.type.upper()}]>"

    def extra_repr(self) -> str:
        return f"type='{self.type}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(layout_type=kwargs["type"], confidence=kwargs["confidence"], geometry=kwargs["geometry"])


class TableCell(Element):
    """Implements a single cell of a recognized table

    Args:
        value: the text content of the cell (words assigned to the cell, joined together)
        confidence: the mean recognition confidence of the words assigned to the cell
        geometry: bounding box of the cell in format ((xmin, ymin), (xmax, ymax)) or a (4, 2) polygon,
            with coordinates relative to the page's size
        row_start: index of the first row spanned by the cell (0-indexed)
        row_end: index of the last row spanned by the cell (0-indexed, inclusive)
        col_start: index of the first column spanned by the cell (0-indexed)
        col_end: index of the last column spanned by the cell (0-indexed, inclusive)
    """

    _exported_keys: list[str] = [
        "geometry",
        "value",
        "confidence",
        "row_start",
        "row_end",
        "col_start",
        "col_end",
    ]
    _children_names: list[str] = []

    def __init__(
        self,
        value: str,
        confidence: float,
        geometry: BoundingBox | np.ndarray,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> None:
        super().__init__()
        self.value = value
        self.confidence = confidence
        self.geometry = geometry
        self.row_start = row_start
        self.row_end = row_end
        self.col_start = col_start
        self.col_end = col_end

    @property
    def row_span(self) -> int:
        """Number of rows spanned by the cell"""
        return self.row_end - self.row_start + 1

    @property
    def col_span(self) -> int:
        """Number of columns spanned by the cell"""
        return self.col_end - self.col_start + 1

    def render(self) -> str:
        """Renders the cell text"""
        return self.value

    def extra_repr(self) -> str:
        return f"value='{self.value}', rows=({self.row_start}, {self.row_end}), cols=({self.col_start}, {self.col_end})"

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


class Table(Element):
    """Implements a table recognized on a page as a grid of cells

    The recognized text of the words falling inside the table is regrouped here and removed from the
    regular `blocks` output of the page, so it is not duplicated. The structured content can be loaded
    directly into pandas, e.g. `pd.DataFrame(table.to_grid())`.

    Args:
        cells: list of `TableCell` objects composing the table
        num_rows: number of rows of the table
        num_cols: number of columns of the table
        geometry: bounding box enclosing the whole table, with coordinates relative to the page's size
        confidence: the confidence of the table structure prediction
    """

    _exported_keys: list[str] = ["geometry", "num_rows", "num_cols", "confidence"]
    _children_names: list[str] = ["cells"]
    cells: list[TableCell] = []

    def __init__(
        self,
        cells: list[TableCell],
        num_rows: int,
        num_cols: int,
        geometry: BoundingBox | np.ndarray,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(cells=cells)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.geometry = geometry
        self.confidence = confidence

    def to_grid(self) -> list[list[str]]:
        """Return the table content as a dense `num_rows` x `num_cols` grid of strings.

        Cells spanning several rows/columns have their value placed at their top-left position; the
        remaining positions they span are left empty. The result is directly loadable into pandas via
        `pd.DataFrame(table.to_grid())`.

        Returns:
            a list of `num_rows` lists, each of length `num_cols`
        """
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for cell in self.cells:
            if 0 <= cell.row_start < self.num_rows and 0 <= cell.col_start < self.num_cols:
                grid[cell.row_start][cell.col_start] = cell.value
        return grid

    def render(self, row_break: str = "\n", col_break: str = "\t") -> str:
        """Renders the table as plain text (tab-separated values)"""
        return row_break.join(col_break.join(row) for row in self.to_grid())

    def extra_repr(self) -> str:
        return f"num_rows={self.num_rows}, num_cols={self.num_cols}, confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs["cells"] = [TableCell.from_dict(cell) for cell in save_dict["cells"]]
        return cls(**kwargs)


class Line(Element):
    """Implements a line element as a collection of words

    Args:
        words: list of word elements
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all words in it.
    """

    _exported_keys: list[str] = ["geometry", "objectness_score"]
    _children_names: list[str] = ["words"]
    words: list[Word] = []

    def __init__(
        self,
        words: list[Word],
        geometry: BoundingBox | np.ndarray | None = None,
        objectness_score: float | None = None,
    ) -> None:
        # Compute the objectness score of the line
        if objectness_score is None:
            objectness_score = float(np.mean([w.objectness_score for w in words]))
        # Resolve the geometry using the smallest enclosing bounding box
        if geometry is None:
            # Check whether this is a rotated or straight box
            box_resolution_fn = resolve_enclosing_rbbox if len(words[0].geometry) == 4 else resolve_enclosing_bbox
            geometry = box_resolution_fn([w.geometry for w in words])  # type: ignore[misc]

        super().__init__(words=words)
        self.geometry = geometry
        self.objectness_score = objectness_score

    def render(self) -> str:
        """Renders the full text of the element"""
        return " ".join(w.render() for w in self.words)

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            "words": [Word.from_dict(_dict) for _dict in save_dict["words"]],
        })
        return cls(**kwargs)


class Prediction(Word):
    """Implements a prediction element"""

    def render(self) -> str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) -> str:
        return f"value='{self.value}', confidence={self.confidence:.2}, bounding_box={self.geometry}"


class Block(Element):
    """Implements a block element as a collection of lines and artefacts

    Args:
        lines: list of line elements
        artefacts: list of artefacts
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all lines and artefacts in it.
    """

    _exported_keys: list[str] = ["geometry", "objectness_score"]
    _children_names: list[str] = ["lines", "artefacts"]
    lines: list[Line] = []
    artefacts: list[Artefact] = []

    def __init__(
        self,
        lines: list[Line] = [],
        artefacts: list[Artefact] = [],
        geometry: BoundingBox | np.ndarray | None = None,
        objectness_score: float | None = None,
    ) -> None:
        # Compute the objectness score of the line
        if objectness_score is None:
            objectness_score = float(np.mean([w.objectness_score for line in lines for w in line.words]))
        # Resolve the geometry using the smallest enclosing bounding box
        if geometry is None:
            line_boxes = [word.geometry for line in lines for word in line.words]
            artefact_boxes = [artefact.geometry for artefact in artefacts]
            box_resolution_fn = (
                resolve_enclosing_rbbox if isinstance(lines[0].geometry, np.ndarray) else resolve_enclosing_bbox
            )
            geometry = box_resolution_fn(line_boxes + artefact_boxes)  # type: ignore

        super().__init__(lines=lines, artefacts=artefacts)
        self.geometry = geometry
        self.objectness_score = objectness_score

    def render(self, line_break: str = "\n") -> str:
        """Renders the full text of the element"""
        return line_break.join(line.render() for line in self.lines)

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            "lines": [Line.from_dict(_dict) for _dict in save_dict["lines"]],
            "artefacts": [Artefact.from_dict(_dict) for _dict in save_dict["artefacts"]],
        })
        return cls(**kwargs)


class Page(Element):
    """Implements a page element as a collection of blocks

    Args:
        page: image encoded as a numpy array in uint8
        blocks: list of block elements
        page_idx: the index of the page in the input raw document
        dimensions: the page size in pixels in format (height, width)
        orientation: a dictionary with the value of the rotation angle in degress and confidence of the prediction
        language: a dictionary with the language value and confidence of the prediction
        layout: optional list of layout regions detected on the page
        tables: optional list of tables recognized on the page. Words assigned to a table are removed from `blocks`.
    """

    _exported_keys: list[str] = ["page_idx", "dimensions", "orientation", "language"]
    _children_names: list[str] = ["blocks", "layout", "tables"]
    blocks: list[Block] = []
    layout: list[LayoutElement] = []
    tables: list[Table] = []

    def __init__(
        self,
        page: np.ndarray,
        blocks: list[Block],
        page_idx: int,
        dimensions: tuple[int, int],
        orientation: dict[str, Any] | None = None,
        language: dict[str, Any] | None = None,
        layout: list[LayoutElement] | None = None,
        tables: list[Table] | None = None,
    ) -> None:
        super().__init__(
            blocks=blocks,
            layout=layout if layout is not None else [],
            tables=tables if tables is not None else [],
        )
        self.page = page
        self.page_idx = page_idx
        self.dimensions = dimensions
        self.orientation = orientation if isinstance(orientation, dict) else dict(value=None, confidence=None)
        self.language = language if isinstance(language, dict) else dict(value=None, confidence=None)

    def render(self, block_break: str = "\n\n") -> str:
        """Renders the full text of the element"""
        return block_break.join(b.render() for b in self.blocks)

    def extra_repr(self) -> str:
        return f"dimensions={self.dimensions}"

    def show(self, interactive: bool = True, preserve_aspect_ratio: bool = False, **kwargs) -> None:
        """Overlay the result on a given image

        Args:
            interactive: whether the display should be interactive
            preserve_aspect_ratio: pass True if you passed True to the predictor
            **kwargs: additional keyword arguments passed to the matplotlib.pyplot.show method
                (e.g. `display_layout=False` to hide detected layout regions)
        """
        requires_package("matplotlib", "`.show()` requires matplotlib & mplcursors installed")
        requires_package("mplcursors", "`.show()` requires matplotlib & mplcursors installed")
        import matplotlib.pyplot as plt

        show_kwargs = {k: kwargs.pop(k) for k in ("words_only", "display_artefacts", "display_layout") if k in kwargs}
        visualize_page(
            self.export(),
            self.page,
            interactive=interactive,
            preserve_aspect_ratio=preserve_aspect_ratio,
            **show_kwargs,
        )
        plt.show(**kwargs)

    def synthesize(self, **kwargs) -> np.ndarray:
        """Synthesize the page from the predictions

        Args:
            **kwargs: keyword arguments passed to the `synthesize_page` method

        Returns:
            synthesized page
        """
        return synthesize_page(self.export(), **kwargs)

    def export_as_xml(self, file_title: str = "docTR - XML export (hOCR)") -> tuple[bytes, ET.ElementTree]:
        """Export the page as XML (hOCR-format)
        convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

        Args:
            file_title: the title of the XML file

        Returns:
            a tuple of the XML byte string, and its ElementTree
        """
        p_idx = self.page_idx
        block_count: int = 1
        line_count: int = 1
        word_count: int = 1
        height, width = self.dimensions
        language = self.language if "language" in self.language.keys() else "en"
        # Create the XML root element
        page_hocr = ETElement("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": str(language)})
        # Create the header / SubElements of the root element
        head = SubElement(page_hocr, "head")
        SubElement(head, "title").text = file_title
        SubElement(head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"})
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-system", "content": f"python-doctr {doctr.__version__}"},  # type: ignore[attr-defined]
        )
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-capabilities", "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word"},
        )
        # Create the body
        body = SubElement(page_hocr, "body")
        page_div = SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": f"page_{p_idx + 1}",
                "title": f"image; bbox 0 0 {width} {height}; ppageno 0",
            },
        )
        # iterate over the blocks / lines / words and create the XML elements in body line by line with the attributes
        for block in self.blocks:
            if len(block.geometry) != 2:
                raise TypeError("XML export is only available for straight bounding boxes for now.")
            (xmin, ymin), (xmax, ymax) = block.geometry
            block_div = SubElement(
                page_div,
                "div",
                attrib={
                    "class": "ocr_carea",
                    "id": f"block_{block_count}",
                    "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                    {int(round(xmax * width))} {int(round(ymax * height))}",
                },
            )
            paragraph = SubElement(
                block_div,
                "p",
                attrib={
                    "class": "ocr_par",
                    "id": f"par_{block_count}",
                    "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                    {int(round(xmax * width))} {int(round(ymax * height))}",
                },
            )
            block_count += 1
            for line in block.lines:
                (xmin, ymin), (xmax, ymax) = line.geometry
                # NOTE: baseline, x_size, x_descenders, x_ascenders is currently initalized to 0
                line_span = SubElement(
                    paragraph,
                    "span",
                    attrib={
                        "class": "ocr_line",
                        "id": f"line_{line_count}",
                        "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                        {int(round(xmax * width))} {int(round(ymax * height))}; \
                        baseline 0 0; x_size 0; x_descenders 0; x_ascenders 0",
                    },
                )
                line_count += 1
                for word in line.words:
                    (xmin, ymin), (xmax, ymax) = word.geometry
                    conf = word.confidence
                    word_div = SubElement(
                        line_span,
                        "span",
                        attrib={
                            "class": "ocrx_word",
                            "id": f"word_{word_count}",
                            "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                            {int(round(xmax * width))} {int(round(ymax * height))}; \
                            x_wconf {int(round(conf * 100))}",
                        },
                    )
                    # set the text
                    word_div.text = word.value
                    word_count += 1

        return (ET.tostring(page_hocr, encoding="utf-8", method="xml"), ET.ElementTree(page_hocr))

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            "blocks": [Block.from_dict(block_dict) for block_dict in save_dict["blocks"]],
            "layout": [LayoutElement.from_dict(region_dict) for region_dict in save_dict.get("layout", [])],
            "tables": [Table.from_dict(table_dict) for table_dict in save_dict.get("tables", [])],
        })
        return cls(**kwargs)


class KIEPage(Element):
    """Implements a KIE page element as a collection of predictions

    Args:
        predictions: Dictionary with list of block elements for each detection class
        page: image encoded as a numpy array in uint8
        page_idx: the index of the page in the input raw document
        dimensions: the page size in pixels in format (height, width)
        orientation: a dictionary with the value of the rotation angle in degress and confidence of the prediction
        language: a dictionary with the language value and confidence of the prediction
        layout: optional list of layout regions detected on the page
    """

    _exported_keys: list[str] = ["page_idx", "dimensions", "orientation", "language"]
    _children_names: list[str] = ["predictions", "layout"]
    predictions: dict[str, list[Prediction]] = {}
    layout: list[LayoutElement] = []

    def __init__(
        self,
        page: np.ndarray,
        predictions: dict[str, list[Prediction]],
        page_idx: int,
        dimensions: tuple[int, int],
        orientation: dict[str, Any] | None = None,
        language: dict[str, Any] | None = None,
        layout: list[LayoutElement] | None = None,
    ) -> None:
        super().__init__(predictions=predictions, layout=layout if layout is not None else [])
        self.page = page
        self.page_idx = page_idx
        self.dimensions = dimensions
        self.orientation = orientation if isinstance(orientation, dict) else dict(value=None, confidence=None)
        self.language = language if isinstance(language, dict) else dict(value=None, confidence=None)

    def render(self, prediction_break: str = "\n\n") -> str:
        """Renders the full text of the element"""
        return prediction_break.join(
            f"{class_name}: {p.render()}" for class_name, predictions in self.predictions.items() for p in predictions
        )

    def extra_repr(self) -> str:
        return f"dimensions={self.dimensions}"

    def show(self, interactive: bool = True, preserve_aspect_ratio: bool = False, **kwargs) -> None:
        """Overlay the result on a given image

        Args:
            interactive: whether the display should be interactive
            preserve_aspect_ratio: pass True if you passed True to the predictor
            **kwargs: keyword arguments passed to the matplotlib.pyplot.show method
        """
        requires_package("matplotlib", "`.show()` requires matplotlib & mplcursors installed")
        requires_package("mplcursors", "`.show()` requires matplotlib & mplcursors installed")
        import matplotlib.pyplot as plt

        show_kwargs = {k: kwargs.pop(k) for k in ("words_only", "display_artefacts", "display_layout") if k in kwargs}
        visualize_kie_page(
            self.export(),
            self.page,
            interactive=interactive,
            preserve_aspect_ratio=preserve_aspect_ratio,
            **show_kwargs,
        )
        plt.show(**kwargs)

    def synthesize(self, **kwargs) -> np.ndarray:
        """Synthesize the page from the predictions

        Args:
            **kwargs: keyword arguments passed to the `synthesize_kie_page` method

        Returns:
            synthesized page
        """
        return synthesize_kie_page(self.export(), **kwargs)

    def export_as_xml(self, file_title: str = "docTR - XML export (hOCR)") -> tuple[bytes, ET.ElementTree]:
        """Export the page as XML (hOCR-format)
        convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

        Args:
            file_title: the title of the XML file

        Returns:
            a tuple of the XML byte string, and its ElementTree
        """
        p_idx = self.page_idx
        prediction_count: int = 1
        height, width = self.dimensions
        language = self.language if "language" in self.language.keys() else "en"
        # Create the XML root element
        page_hocr = ETElement("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": str(language)})
        # Create the header / SubElements of the root element
        head = SubElement(page_hocr, "head")
        SubElement(head, "title").text = file_title
        SubElement(head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"})
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-system", "content": f"python-doctr {doctr.__version__}"},  # type: ignore[attr-defined]
        )
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-capabilities", "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word"},
        )
        # Create the body
        body = SubElement(page_hocr, "body")
        SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": f"page_{p_idx + 1}",
                "title": f"image; bbox 0 0 {width} {height}; ppageno 0",
            },
        )
        # iterate over the blocks / lines / words and create the XML elements in body line by line with the attributes
        for class_name, predictions in self.predictions.items():
            for prediction in predictions:
                if len(prediction.geometry) != 2:
                    raise TypeError("XML export is only available for straight bounding boxes for now.")
                (xmin, ymin), (xmax, ymax) = prediction.geometry
                prediction_div = SubElement(
                    body,
                    "div",
                    attrib={
                        "class": "ocr_carea",
                        "id": f"{class_name}_prediction_{prediction_count}",
                        "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                        {int(round(xmax * width))} {int(round(ymax * height))}",
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
                        "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                        {int(round(xmax * width))} {int(round(ymax * height))}",
                    },
                )
                line_span = SubElement(
                    par_div,
                    "span",
                    attrib={
                        "class": "ocr_line",
                        "id": f"{class_name}_line_{prediction_count}",
                        "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                        {int(round(xmax * width))} {int(round(ymax * height))}; \
                        baseline 0 0; x_size 0; x_descenders 0; x_ascenders 0",
                    },
                )
                word_div = SubElement(
                    line_span,
                    "span",
                    attrib={
                        "class": "ocrx_word",
                        "id": f"{class_name}_word_{prediction_count}",
                        "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                        {int(round(xmax * width))} {int(round(ymax * height))}; \
                        x_wconf {int(round(prediction.confidence * 100))}",
                    },
                )
                word_div.text = prediction.value
                prediction_count += 1

        return ET.tostring(page_hocr, encoding="utf-8", method="xml"), ET.ElementTree(page_hocr)

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            "predictions": {
                class_name: [Prediction.from_dict(pred) for pred in preds]
                for class_name, preds in save_dict["predictions"].items()
            },
            "layout": [LayoutElement.from_dict(region_dict) for region_dict in save_dict.get("layout", [])],
        })
        return cls(**kwargs)


class Document(Element):
    """Implements a document element as a collection of pages

    Args:
        pages: list of page elements
    """

    _children_names: list[str] = ["pages"]
    pages: list[Page] = []

    def __init__(
        self,
        pages: list[Page],
    ) -> None:
        super().__init__(pages=pages)

    def render(self, page_break: str = "\n\n\n\n") -> str:
        """Renders the full text of the element"""
        return page_break.join(p.render() for p in self.pages)

    def show(self, **kwargs) -> None:
        """Overlay the result on a given image"""
        for result in self.pages:
            result.show(**kwargs)

    def synthesize(self, **kwargs) -> list[np.ndarray]:
        """Synthesize all pages from their predictions

        Args:
            **kwargs: keyword arguments passed to the `Page.synthesize` method

        Returns:
            list of synthesized pages
        """
        return [page.synthesize(**kwargs) for page in self.pages]

    def export_as_xml(self, **kwargs) -> list[tuple[bytes, ET.ElementTree]]:
        """Export the document as XML (hOCR-format)

        Args:
            **kwargs: additional keyword arguments passed to the Page.export_as_xml method

        Returns:
            list of tuple of (bytes, ElementTree)
        """
        return [page.export_as_xml(**kwargs) for page in self.pages]

    @classmethod
    def from_dict(cls, save_dict: dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({"pages": [Page.from_dict(page_dict) for page_dict in save_dict["pages"]]})
        return cls(**kwargs)


class KIEDocument(Document):
    """Implements a document element as a collection of pages

    Args:
        pages: list of page elements
    """

    _children_names: list[str] = ["pages"]
    pages: list[KIEPage] = []  # type: ignore[assignment]

    def __init__(
        self,
        pages: list[KIEPage],
    ) -> None:
        super().__init__(pages=pages)  # type: ignore[arg-type]
