import numpy as np
import pytest

from doctr.io import Document
from doctr.models import builder


def test_documentbuilder():

    words_per_page = 10
    num_pages = 2

    # Don't resolve lines
    doc_builder = builder.DocumentBuilder(resolve_lines=False, resolve_blocks=False)
    boxes = np.random.rand(words_per_page, 6)
    boxes[:2] *= boxes[2:4]

    # Arg consistency check
    with pytest.raises(ValueError):
        doc_builder([boxes, boxes], [('hello', 1.0)] * 3, [(100, 200), (100, 200)])
    out = doc_builder([boxes, boxes], [[('hello', 1.0)] * words_per_page] * num_pages, [(100, 200), (100, 200)])
    assert isinstance(out, Document)
    assert len(out.pages) == num_pages
    # 1 Block & 1 line per page
    assert len(out.pages[0].blocks) == 1 and len(out.pages[0].blocks[0].lines) == 1
    assert len(out.pages[0].blocks[0].lines[0].words) == words_per_page

    # Resolve lines
    doc_builder = builder.DocumentBuilder(resolve_lines=True, resolve_blocks=True)
    out = doc_builder([boxes, boxes], [[('hello', 1.0)] * words_per_page] * num_pages, [(100, 200), (100, 200)])

    # No detection
    boxes = np.zeros((0, 5))
    out = doc_builder([boxes, boxes], [[], []], [(100, 200), (100, 200)])
    assert len(out.pages[0].blocks) == 0

    # Rotated boxes to export à straight boxes
    boxes = np.array([
        [0.25, 0.25, np.sqrt(2) / 4, np.sqrt(2) / 4, 45, 0.99],
        [0.75, 0.75, np.sqrt(2) / 4, np.sqrt(2) / 4, 45, 0.99],
    ])
    doc_builder_2 = builder.DocumentBuilder(
        resolve_blocks=False,
        resolve_lines=False,
        export_as_straight_boxes=True
    )
    out = doc_builder_2([boxes], [[("hello", 0.99), ("world", 0.99)]], [(100, 100)])
    assert out.pages[0].blocks[0].lines[0].words[-1].geometry == ((0.5, 0.5), (1.0, 1.0))

    # Repr
    assert repr(doc_builder) == "DocumentBuilder(resolve_lines=True, " \
                                "resolve_blocks=True, paragraph_break=0.035, export_as_straight_boxes=False)"


@pytest.mark.parametrize(
    "input_boxes, sorted_idxs",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # diagonal
        [[[0, 0.5, 0.1, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [0, 1, 2]],  # same line, 2p
        [[[0, 0.5, 0.1, 0.6], [0.2, 0.49, 0.35, 0.59], [0.8, 0.52, 0.9, 0.63]], [0, 1, 2]],  # ~same line
        [[[0, 0.3, 0.4, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
    ],
)
def test_sort_boxes(input_boxes, sorted_idxs):

    doc_builder = builder.DocumentBuilder()
    assert doc_builder._sort_boxes(np.asarray(input_boxes)).tolist() == sorted_idxs


@pytest.mark.parametrize(
    "input_boxes, lines",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [[2], [1], [0]]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [[2], [1], [0]]],  # diagonal
        [[[0, 0.5, 0.14, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [[0, 1], [2]]],  # same line, 2p
        [[[0, 0.5, 0.18, 0.6], [0.2, 0.48, 0.35, 0.58], [0.8, 0.52, 0.9, 0.63]], [[0, 1], [2]]],  # ~same line
        [[[0, 0.3, 0.48, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [[0, 1], [2]]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [[0], [1], [2]]],  # 2 lines
    ],
)
def test_resolve_lines(input_boxes, lines):

    doc_builder = builder.DocumentBuilder()
    assert doc_builder._resolve_lines(np.asarray(input_boxes)) == lines
