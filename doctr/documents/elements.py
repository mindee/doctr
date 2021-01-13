# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple, Dict, List, Any, Optional
from doctr.utils.geometry import resolve_enclosing_bbox
from doctr.utils.typing import BoundingBox, Polygon4P

__all__ = ['Element', 'Word', 'Line', 'Block', 'Page', 'Document']


class Element:
    """Implements an abstract document element with exporting and text rendering capabilities

    Args:
        children: list of children elements
    """

    _children_name: str = "children"
    _exported_keys: List[str] = []

    def __init__(self, children: List[Any] = []) -> None:
        self.children = children

    def export(self) -> Dict[str, Any]:
        """Exports the object into a nested dict format"""

        export_dict = {k: getattr(self, k) for k in self._exported_keys}
        if any(self.children):
            export_dict[self._children_name] = [c.export() for c in self.children]

        return export_dict

    def render(self) -> str:
        raise NotImplementedError


class Word(Element):
    """Implements a word element

    Args:
        value: the text string of the word
        confidence: the confidence associated with the text prediction
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates
        are relative to the page's size
    """

    _exported_keys: List[str] = ["value", "confidence", "geometry"]

    def __init__(self, value: str, confidence: float, geometry: BoundingBox) -> None:
        super().__init__()
        self.value = value
        self.confidence = confidence
        self.geometry = geometry

    def render(self) -> str:
        """Renders the full text of the element"""
        return self.value


class Line(Element):
    """Implements a line element as a collection of words

    Args:
        words: list of word elements
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates
        are relative to the page's size. If not specified, it will be resolved by default to the smallest bounding box
        enclosing all words in it.
    """

    _children_name: str = "words"
    _exported_keys: List[str] = ["geometry"]

    def __init__(
        self,
        words: List[Word],
        geometry: Optional[BoundingBox] = None,
    ) -> None:
        # Resolve the geometry using the smallest enclosing bounding box
        if geometry is None:
            geometry = resolve_enclosing_bbox([w.geometry for w in words])

        super().__init__(words)
        self.geometry = geometry

    def render(self) -> str:
        """Renders the full text of the element"""
        return " ".join(w.render() for w in self.children)


class Block(Element):
    """Implements a block element as a collection of lines

    Args:
        lines: list of line elements
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates
        are relative to the page's size. If not specified, it will be resolved by default to the smallest bounding box
        enclosing all blocks in it.
    """

    _children_name: str = "lines"
    _exported_keys: List[str] = ["geometry"]

    def __init__(
        self,
        lines: List[Word],
        geometry: Optional[BoundingBox] = None,
    ) -> None:
        # Resolve the geometry using the smallest enclosing bounding box
        if geometry is None:
            geometry = resolve_enclosing_bbox([word.geometry for line in lines for word in line.children])
        super().__init__(lines)
        self.geometry = geometry

    def render(self, line_break: str = '\n') -> str:
        """Renders the full text of the element"""
        return line_break.join(line.render() for line in self.children)


class Page(Element):
    """Implements a page element as a collection of blocks

    Args:
        blocks: list of block elements
        page_idx: the index of the page in the input raw document
        dimensions: the page size in pixels in format (width, height)
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates
        are relative to the page's size. If not specified, it will be resolved by default to the smallest bounding box
        enclosing all blocks in it.
    """

    _children_name: str = "blocks"
    _exported_keys: List[str] = ["page_idx", "dimensions", "orientation", "language"]

    def __init__(
        self,
        blocks: List[Block],
        page_idx: int,
        dimensions: Tuple[int, int],
        orientation: Dict[str, Any],
        language: Dict[str, Any],
    ) -> None:
        super().__init__(blocks)
        self.page_idx = page_idx
        self.dimensions = dimensions
        self.orientation = orientation
        self.language = language

    def render(self, block_break: str = '\n\n') -> str:
        """Renders the full text of the element"""
        return block_break.join(b.render() for b in self.children)


class Document(Element):

    _children_name: str = "pages"

    def __init__(
        self,
        pages: List[Page],
    ) -> None:
        super().__init__(pages)

    def render(self, page_break: str = '\n\n\n\n') -> str:
        """Renders the full text of the element"""
        return page_break.join(p.render() for p in self.children)
