# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any, Optional, Union

from doctr.utils.geometry import resolve_enclosing_bbox, resolve_enclosing_rbbox
from doctr.utils.visualization import visualize_page, synthesize_page
from doctr.utils.common_types import BoundingBox, RotatedBbox
from doctr.utils.repr import NestedObject

__all__ = ['Element', 'Word', 'Artefact', 'Line', 'Block', 'Page', 'Document']


class Element(NestedObject):
    """Implements an abstract document element with exporting and text rendering capabilities"""

    _children_names: List[str] = []
    _exported_keys: List[str] = []

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self._children_names:
                setattr(self, k, v)
            else:
                raise KeyError(f"{self.__class__.__name__} object does not have any attribute named '{k}'")

    def export(self) -> Dict[str, Any]:
        """Exports the object into a nested dict format"""

        export_dict = {k: getattr(self, k) for k in self._exported_keys}
        for children_name in self._children_names:
            export_dict[children_name] = [c.export() for c in getattr(self, children_name)]

        return export_dict

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
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
    """

    _exported_keys: List[str] = ["value", "confidence", "geometry"]
    _children_names: List[str] = []

    def __init__(self, value: str, confidence: float, geometry: Union[BoundingBox, RotatedBbox]) -> None:
        super().__init__()
        self.value = value
        self.confidence = confidence
        self.geometry = geometry

    def render(self) -> str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) -> str:
        return f"value='{self.value}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
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

    _exported_keys: List[str] = ["geometry", "type", "confidence"]
    _children_names: List[str] = []

    def __init__(self, artefact_type: str, confidence: float, geometry: BoundingBox) -> None:
        super().__init__()
        self.geometry = geometry
        self.type = artefact_type
        self.confidence = confidence

    def render(self) -> str:
        """Renders the full text of the element"""
        return f"[{self.type.upper()}]"

    def extra_repr(self) -> str:
        return f"type='{self.type}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        return cls(**kwargs)


class Line(Element):
    """Implements a line element as a collection of words

    Args:
        words: list of word elements
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all words in it.
    """

    _exported_keys: List[str] = ["geometry"]
    _children_names: List[str] = ['words']
    words: List[Word] = []

    def __init__(
        self,
        words: List[Word],
        geometry: Optional[Union[BoundingBox, RotatedBbox]] = None,
    ) -> None:
        # Resolve the geometry using the smallest enclosing bounding box
        if geometry is None:
            # Check whether this is a rotated or straight box
            box_resolution_fn = resolve_enclosing_rbbox if len(words[0].geometry) == 5 else resolve_enclosing_bbox
            geometry = box_resolution_fn([w.geometry for w in words])  # type: ignore[operator, misc]

        super().__init__(words=words)
        self.geometry = geometry

    def render(self) -> str:
        """Renders the full text of the element"""
        return " ".join(w.render() for w in self.words)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            'words': [Word.from_dict(_dict) for _dict in save_dict['words']],
        })
        return cls(**kwargs)


class Block(Element):
    """Implements a block element as a collection of lines and artefacts

    Args:
        lines: list of line elements
        artefacts: list of artefacts
        geometry: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all lines and artefacts in it.
    """

    _exported_keys: List[str] = ["geometry"]
    _children_names: List[str] = ['lines', 'artefacts']
    lines: List[Line] = []
    artefacts: List[Artefact] = []

    def __init__(
        self,
        lines: List[Line] = [],
        artefacts: List[Artefact] = [],
        geometry: Optional[Union[BoundingBox, RotatedBbox]] = None,
    ) -> None:
        # Resolve the geometry using the smallest enclosing bounding box
        if geometry is None:
            line_boxes = [word.geometry for line in lines for word in line.words]
            artefact_boxes = [artefact.geometry for artefact in artefacts]
            box_resolution_fn = resolve_enclosing_rbbox if len(lines[0].geometry) == 5 else resolve_enclosing_bbox
            geometry = box_resolution_fn(line_boxes + artefact_boxes)  # type: ignore[operator, arg-type]

        super().__init__(lines=lines, artefacts=artefacts)
        self.geometry = geometry

    def render(self, line_break: str = '\n') -> str:
        """Renders the full text of the element"""
        return line_break.join(line.render() for line in self.lines)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            'lines': [Line.from_dict(_dict) for _dict in save_dict['lines']],
            'artefacts': [Artefact.from_dict(_dict) for _dict in save_dict['artefacts']],
        })
        return cls(**kwargs)


class Page(Element):
    """Implements a page element as a collection of blocks

    Args:
        blocks: list of block elements
        page_idx: the index of the page in the input raw document
        dimensions: the page size in pixels in format (width, height)
        orientation: a dictionary with the value of the rotation angle in degress and confidence of the prediction
        language: a dictionary with the language value and confidence of the prediction
    """

    _exported_keys: List[str] = ["page_idx", "dimensions", "orientation", "language"]
    _children_names: List[str] = ['blocks']
    blocks: List[Block] = []

    def __init__(
        self,
        blocks: List[Block],
        page_idx: int,
        dimensions: Tuple[int, int],
        orientation: Optional[Dict[str, Any]] = None,
        language: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(blocks=blocks)
        self.page_idx = page_idx
        self.dimensions = dimensions
        self.orientation = orientation if isinstance(orientation, dict) else dict(value=None, confidence=None)
        self.language = language if isinstance(language, dict) else dict(value=None, confidence=None)

    def render(self, block_break: str = '\n\n') -> str:
        """Renders the full text of the element"""
        return block_break.join(b.render() for b in self.blocks)

    def extra_repr(self) -> str:
        return f"dimensions={self.dimensions}"

    def show(
        self, page: np.ndarray, interactive: bool = True, **kwargs
    ) -> None:
        """Overlay the result on a given image

        Args:
            page: image encoded as a numpy array in uint8
            interactive: whether the display should be interactive
        """
        visualize_page(self.export(), page, interactive=interactive)
        plt.show(**kwargs)

    def synthesize(self, **kwargs) -> np.ndarray:
        """Synthesize the page from the predictions

        Returns:
            synthesized page
        """

        return synthesize_page(self.export(), **kwargs)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({'blocks': [Block.from_dict(block_dict) for block_dict in save_dict['blocks']]})
        return cls(**kwargs)


class Document(Element):
    """Implements a document element as a collection of pages

    Args:
        pages: list of page elements
    """

    _children_names: List[str] = ['pages']
    pages: List[Page] = []

    def __init__(
        self,
        pages: List[Page],
    ) -> None:
        super().__init__(pages=pages)

    def render(self, page_break: str = '\n\n\n\n') -> str:
        """Renders the full text of the element"""
        return page_break.join(p.render() for p in self.pages)

    def show(self, pages: List[np.ndarray], **kwargs) -> None:
        """Overlay the result on a given image

        Args:
            pages: list of images encoded as numpy arrays in uint8
        """
        for img, result in zip(pages, self.pages):
            result.show(img, **kwargs)

    def synthesize(self, **kwargs) -> List[np.ndarray]:
        """Synthesize all pages from their predictions

        Returns:
            list of synthesized pages
        """

        return [page.synthesize() for page in self.pages]

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({'pages': [Page.from_dict(page_dict) for page_dict in save_dict['pages']]})
        return cls(**kwargs)
