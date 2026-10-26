import json

import numpy as np
import pytest

from doctr.file_utils import CLASS_NAME
from doctr.io import Document
from doctr.io.elements import KIEDocument, LayoutElement, Table
from doctr.models import builder

words_per_page = 10


def _orientations(num):
    return [{"value": 0, "confidence": None}] * num


def _word_args(words, boxes, shape=(1000, 1000), confidences=None):
    """Pack builder arguments for a single page from word values and boxes"""
    confidences = [0.95] * len(words) if confidences is None else confidences
    return (
        [np.zeros((*shape, 3), dtype=np.uint8)],
        [np.asarray(boxes, dtype=np.float32)],
        [np.full(len(words), 0.9, dtype=np.float32)],
        [list(zip(words, confidences))],
        [shape],
        [_orientations(len(words))],
    )


def _words(page):
    return [word.value for block in page.blocks for line in block.lines for word in line.words]


def _rot_poly(x0, y0, x1, y1, deg, cx=0.5, cy=0.5):
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return [
        [cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca]
        for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    ]


def test_documentbuilder():
    num_pages = 2

    # Don't resolve lines
    doc_builder = builder.DocumentBuilder(resolve_lines=False, resolve_blocks=False)
    pages = [np.zeros((100, 200, 3))] * num_pages
    boxes = np.random.rand(words_per_page, 6)  # array format
    boxes[:2] *= boxes[2:4]
    objectness_scores = np.array([0.9] * words_per_page)
    # Arg consistency check
    with pytest.raises(ValueError):
        doc_builder(
            pages,
            [boxes, boxes],
            [objectness_scores, objectness_scores],
            [("hello", 1.0)] * 3,
            [(100, 200)] * num_pages,
            _orientations(3),
        )
    args = (
        pages,
        [boxes, boxes],
        [objectness_scores, objectness_scores],
        [[("hello", 1.0)] * words_per_page] * num_pages,
        [(100, 200)] * num_pages,
        [_orientations(words_per_page)] * num_pages,
    )
    out = doc_builder(*args)
    assert isinstance(out, Document)
    assert len(out.pages) == num_pages
    assert all(isinstance(page.page, np.ndarray) and page.page.shape == (100, 200, 3) for page in out.pages)
    # 1 Block & 1 line per page
    assert len(out.pages[0].blocks) == 1 and len(out.pages[0].blocks[0].lines) == 1
    assert len(out.pages[0].blocks[0].lines[0].words) == words_per_page

    # Resolve lines
    doc_builder = builder.DocumentBuilder(resolve_lines=True, resolve_blocks=True)
    assert isinstance(doc_builder(*args), Document)

    # No detection
    out = doc_builder(
        pages, [np.zeros((0, 4))] * 2, [np.zeros([0])] * 2, [[], []], [(100, 200)] * num_pages, [[]] * num_pages
    )
    assert len(out.pages[0].blocks) == 0

    # Rotated boxes to export as straight boxes
    boxes = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
        [[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]],
    ])
    doc_builder_2 = builder.DocumentBuilder(resolve_blocks=False, resolve_lines=False, export_as_straight_boxes=True)
    out = doc_builder_2(
        [np.zeros((100, 100, 3))],
        [boxes],
        [np.array([0.99, 0.99])],
        [[("hello", 0.99), ("word", 0.99)]],
        [(100, 100)],
        [_orientations(2)],
    )
    assert out.pages[0].blocks[0].lines[0].words[-1].geometry == ((0.45, 0.5), (0.6, 0.65))
    assert out.pages[0].blocks[0].lines[0].words[-1].objectness_score == 0.99

    # A small `paragraph_break` still splits table cells separated by one word height
    small_break = builder.DocumentBuilder(paragraph_break=0.01)
    assert len(small_break._resolve_lines(np.asarray(_table_rows()), (2339, 1654))) == 12

    # Repr
    assert (
        repr(doc_builder) == "DocumentBuilder(resolve_lines=True, "
        "resolve_blocks=True, paragraph_break=0.035, export_as_straight_boxes=False, keep_reading_order=False)"
    )

    # A single mismatched argument must raise, whichever one it is
    boxes = np.random.rand(words_per_page, 6)
    boxes[:2] *= boxes[2:4]
    kwargs = dict(
        pages=pages,
        boxes=[boxes[:, :4]] * num_pages,
        objectness_scores=[np.array([0.9] * words_per_page)] * num_pages,
        text_preds=[[("hello", 0.99)] * words_per_page] * num_pages,
        page_shapes=[(100, 200)] * num_pages,
        crop_orientations=[_orientations(words_per_page)] * num_pages,
    )
    # sanity: consistent arguments build fine
    assert isinstance(builder.DocumentBuilder()(**kwargs), Document)
    for key in ("pages", "objectness_scores", "text_preds", "page_shapes", "crop_orientations"):
        with pytest.raises(ValueError):
            builder.DocumentBuilder()(**{**kwargs, key: kwargs[key] + [kwargs[key][0]]})


def test_kiedocumentbuilder():
    num_pages = 2

    # Don't resolve lines
    doc_builder = builder.KIEDocumentBuilder(resolve_lines=False, resolve_blocks=False)
    pages = [np.zeros((100, 200, 3))] * num_pages
    predictions = {CLASS_NAME: np.random.rand(words_per_page, 6)}  # dict format
    predictions[CLASS_NAME][:2] *= predictions[CLASS_NAME][2:4]
    objectness_scores = {CLASS_NAME: np.array([0.9] * words_per_page)}
    # Arg consistency check
    with pytest.raises(ValueError):
        doc_builder(
            pages,
            [predictions, predictions],
            [objectness_scores, objectness_scores],
            [{CLASS_NAME: ("hello", 1.0)}] * 3,
            [(100, 200), (100, 200)],
            [{CLASS_NAME: _orientations(3)}],
        )
    args = (
        pages,
        [predictions, predictions],
        [objectness_scores, objectness_scores],
        [{CLASS_NAME: [("hello", 1.0)] * words_per_page}] * num_pages,
        [(100, 200), (100, 200)],
        [{CLASS_NAME: _orientations(words_per_page)}] * num_pages,
    )
    out = doc_builder(*args)
    assert isinstance(out, KIEDocument)
    assert len(out.pages) == num_pages
    assert all(isinstance(page.page, np.ndarray) and page.page.shape == (100, 200, 3) for page in out.pages)
    assert len(out.pages[0].predictions) == 1
    assert len(out.pages[0].predictions[CLASS_NAME]) == words_per_page

    # Resolve lines
    doc_builder = builder.KIEDocumentBuilder(resolve_lines=True, resolve_blocks=True)
    assert isinstance(doc_builder(*args), KIEDocument)

    # No detection
    predictions = {CLASS_NAME: np.zeros((0, 4))}
    out = doc_builder(
        pages,
        [predictions, predictions],
        [{CLASS_NAME: np.zeros(1)}] * 2,
        [{CLASS_NAME: []}, {CLASS_NAME: []}],
        [(100, 200), (100, 200)],
        [{CLASS_NAME: []}, {CLASS_NAME: []}],
    )
    assert len(out.pages[0].predictions[CLASS_NAME]) == 0

    # Rotated boxes to export as straight boxes
    predictions = {
        CLASS_NAME: np.array([
            [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
            [[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]],
        ])
    }
    doc_builder_2 = builder.KIEDocumentBuilder(resolve_blocks=False, resolve_lines=False, export_as_straight_boxes=True)
    out = doc_builder_2(
        [np.zeros((100, 100, 3))],
        [predictions],
        [{CLASS_NAME: np.array([0.99, 0.99])}],
        [{CLASS_NAME: [("hello", 0.99), ("word", 0.99)]}],
        [(100, 100)],
        [{CLASS_NAME: _orientations(2)}],
    )
    assert out.pages[0].predictions[CLASS_NAME][0].geometry == ((0.05, 0.1), (0.2, 0.25))
    assert out.pages[0].predictions[CLASS_NAME][1].geometry == ((0.45, 0.5), (0.6, 0.65))
    assert out.pages[0].predictions[CLASS_NAME][1].objectness_score == 0.99

    # Repr
    assert (
        repr(doc_builder) == "KIEDocumentBuilder(resolve_lines=True, "
        "resolve_blocks=True, paragraph_break=0.035, export_as_straight_boxes=False, keep_reading_order=False)"
    )

    # Layout regions are attached to the page and exported
    args = (
        [np.zeros((100, 100, 3))],
        [{CLASS_NAME: np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])}],
        [{CLASS_NAME: np.array([0.9, 0.9])}],
        [{CLASS_NAME: [("hello", 0.99), ("world", 0.99)]}],
        [(100, 100)],
        [{CLASS_NAME: _orientations(2)}],
    )
    regions = {
        "boxes": np.array([[0.05, 0.02, 0.95, 0.08], [0.05, 0.2, 0.95, 0.5]], dtype=np.float32),
        "class_names": ["Title", "Text"],
        "scores": [0.95, 0.88],
    }
    page = builder.KIEDocumentBuilder()(*args, regions=[regions]).pages[0]
    assert all(isinstance(region, LayoutElement) for region in page.layout)
    assert [region.type for region in page.layout] == ["Title", "Text"]
    assert page.export()["layout"] == [region.export() for region in page.layout]
    # no regions -> empty layout
    assert builder.KIEDocumentBuilder()(*args).pages[0].layout == []


def test_documentbuilder_layout():
    doc_builder = builder.DocumentBuilder()
    args = _word_args(["hello", "world"], [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], shape=(100, 100))
    regions = {
        "boxes": np.array([[0.05, 0.02, 0.95, 0.08], [0.05, 0.2, 0.95, 0.5]], dtype=np.float32),
        "class_names": ["Title", "Text"],
        "scores": [0.95, 0.88],
    }
    page = doc_builder(*args, regions=[regions]).pages[0]
    # Layout regions are attached as LayoutElement and exported
    assert all(isinstance(region, LayoutElement) for region in page.layout)
    assert [region.type for region in page.layout] == ["Title", "Text"]
    assert page.layout[0].confidence == pytest.approx(0.95)
    assert np.allclose(np.array(page.layout[0].geometry), [[0.05, 0.02], [0.95, 0.08]], atol=1e-6)
    assert page.export()["layout"] == [region.export() for region in page.layout]
    # no regions -> empty layout
    page = doc_builder(*args).pages[0]
    assert page.layout == [] and page.export()["layout"] == []
    # Rotated layout polygons (4, 2) are converted to a 4-point geometry
    rotated = {
        "boxes": np.array([[[0.1, 0.1], [0.4, 0.12], [0.39, 0.3], [0.09, 0.28]]], dtype=np.float32),
        "class_names": ["Table"],
        "scores": [0.7],
    }
    region = doc_builder(*args, regions=[rotated]).pages[0].layout[0]
    assert region.type == "Table"
    assert isinstance(region.geometry, tuple) and len(region.geometry) == 4

    # 3 columns separated by gutters as narrow as the word gaps of the justified lines
    boxes, columns = [], []
    for col in range(3):
        x0 = 0.02 + 0.33 * col
        columns.append(((x0 - 0.01, 0.05), (x0 + 0.31, 0.95)))
        for row in range(2):
            for word in range(3):
                boxes.append([x0 + 0.105 * word, 0.4 + 0.05 * row, x0 + 0.095 + 0.105 * word, 0.43 + 0.05 * row])
    boxes = np.asarray(boxes)
    # Without layout, each row spans the 3 columns
    assert len(doc_builder._resolve_lines(boxes)) == 2
    # With a layout region per column, a line never spans two of them
    lines = doc_builder._resolve_lines(boxes, word_regions=doc_builder._word_regions(boxes, columns, ["Text"] * 3))
    assert sorted(lines) == [[6 * col + 3 * row + word for word in range(3)] for col in range(3) for row in range(2)]
    # Overlapping regions of the same label (duplicates of a layout model) are one region
    overlapping = [((0.0, 0.0), (0.6, 1.0)), ((0.3, 0.0), (1.0, 1.0))]
    word_regions = doc_builder._word_regions(boxes, overlapping, ["Text"] * 2)
    assert len(doc_builder._resolve_lines(boxes, word_regions=word_regions)) == 2
    # Words outside any region do not split a line
    assert doc_builder._word_regions(boxes, [((0.5, 0.0), (0.6, 0.1))], ["Text"])[0].tolist() == [-1] * 18
    # End to end
    page = doc_builder(
        *_word_args(["word"] * 18, boxes, shape=(100, 100)),
        regions=[{"boxes": np.asarray([[*p0, *p1] for p0, p1 in columns]), "class_names": ["Text"] * 3}],
    ).pages[0]
    assert [len(line.words) for block in page.blocks for line in block.lines] == [3] * 6
    # A line of 8 words in a Text region, with a Picture box over its last word (or an inline Formula box)
    boxes = np.asarray([[0.10 + 0.1 * i, 0.20, 0.18 + 0.1 * i, 0.23] for i in range(8)])
    text = ((0.08, 0.18), (0.92, 0.40))
    for nested, label in [(((0.78, 0.15), (0.95, 0.24)), "Picture"), (((0.38, 0.19), (0.59, 0.24)), "Formula")]:
        word_regions = doc_builder._word_regions(boxes, [text, nested], ["Text", label])
        assert doc_builder._resolve_lines(boxes, word_regions=word_regions) == [list(range(8))]
    # Two columns whose regions start slightly below the first line of the right column (its words are 46% covered)
    boxes = np.asarray([[0.05 + 0.1 * i, 0.30, 0.13 + 0.1 * i, 0.33] for i in range(4)])
    boxes = np.concatenate([boxes, boxes + [0.42, 0.0, 0.42, 0.0]])
    columns = [((0.03, 0.28), (0.45, 0.9)), ((0.46, 0.3162), (0.89, 0.9))]
    word_regions = doc_builder._word_regions(boxes, columns, ["Text", "Text"])
    assert sorted(doc_builder._resolve_lines(boxes, word_regions=word_regions)) == [[0, 1, 2, 3], [4, 5, 6, 7]]


@pytest.mark.parametrize(
    "input_boxes, sorted_idxs",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # diagonal
        [[[0, 0.5, 0.1, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [0, 1, 2]],  # same line, 2p
        [[[0, 0.5, 0.1, 0.6], [0.2, 0.49, 0.35, 0.59], [0.8, 0.52, 0.9, 0.63]], [0, 1, 2]],  # ~same line
        [[[0, 0.3, 0.4, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
        [
            [
                [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
                [[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]],
            ],
            [0, 1],
        ],  # rot
        # Boxes with zero height do not produce a NaN ordering (division by the median height)
        [[[0.5, 0.2, 0.6, 0.2], [0.1, 0.2, 0.2, 0.2]], [1, 0]],
    ],
)
def test_sort_boxes(input_boxes, sorted_idxs):
    doc_builder = builder.DocumentBuilder()
    assert doc_builder._sort_boxes(np.asarray(input_boxes))[0].tolist() == sorted_idxs


# Two columns whose baselines are offset by half a line (scanned page)
_OFFSET_COLUMNS = [
    [0.1422, 0.5908, 0.2472, 0.6045],  # 0-4: left column, "Nichttragende Wände bestehen aus Kalk-"
    [0.261, 0.5898, 0.3176, 0.6035],
    [0.3314, 0.5898, 0.3978, 0.6035],
    [0.413, 0.5928, 0.4406, 0.6035],
    [0.4558, 0.5889, 0.4972, 0.6035],
    [0.1394, 0.6055, 0.2099, 0.6182],  # 5-10: left column, "sandstein oder Vollwanddielen bzw. aus stabilen"
    [0.2126, 0.6064, 0.2486, 0.6172],
    [0.2513, 0.6055, 0.366, 0.6191],
    [0.3674, 0.6045, 0.4047, 0.6191],
    [0.4074, 0.6084, 0.4364, 0.6191],
    [0.4378, 0.6055, 0.4972, 0.6191],
    [0.5539, 0.5811, 0.5953, 0.5957],  # 11-13: right column, "Holz- oder Stahlkonstruktion."
    [0.5953, 0.5811, 0.634, 0.5947],
    [0.634, 0.5811, 0.7694, 0.5977],
    [0.5539, 0.5967, 0.6755, 0.6133],  # 14-18: right column, "Hauseingangstür als Holz-/Glaselement mit Si-/"
    [0.6782, 0.5977, 0.6989, 0.6104],
    [0.7045, 0.5977, 0.8426, 0.6113],
    [0.844, 0.5977, 0.8758, 0.6133],
    [0.8772, 0.5977, 0.9089, 0.6123],
]


def _slanted_lines(slope, ys):
    return [[0.05 + 0.11 * i, y + slope * i, 0.15 + 0.11 * i, y + 0.03 + slope * i] for y in ys for i in range(8)]


def _table_rows():
    # 3 rows of 4 cells (2 words each), the cells being separated by one word height
    boxes = []
    for row in range(3):
        x, y = 0.1, 0.3 + 0.03 * row
        for _ in range(4):
            for _ in range(2):
                boxes.append([x, y, x + 0.05, y + 0.015])
                x += 0.054
            x += 0.011
    return boxes


def _justified_lines(scale=1.0):
    boxes = []
    for y, gaps in [(0.10, [0.019, 0.019, 0.040, 0.019, 0.019]), (0.16, [0.019] * 5), (0.22, [0.019] * 5)]:
        x = 0.05
        for gap in [0.0, *gaps]:
            x += gap
            boxes.append([x, y * scale, x + 0.12, (y + 0.04) * scale])
            x += 0.12
    return boxes


@pytest.mark.parametrize(
    "input_boxes, shape, lines",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], None, [[2], [1], [0]]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], None, [[2], [1], [0]]],  # diagonal
        [[[0, 0.5, 0.14, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], None, [[0, 1], [2]]],  # same line, 2p
        [[[0, 0.5, 0.18, 0.6], [0.2, 0.48, 0.35, 0.58], [0.8, 0.52, 0.9, 0.63]], None, [[0, 1], [2]]],  # ~same line
        [[[0, 0.3, 0.48, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], None, [[0, 1], [2]]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], None, [[0], [1], [2]]],  # 2 lines
        [
            [
                [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
                [[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]],
            ],
            None,
            [[0], [1]],
        ],  # rot
        # Offset columns: a line does not lose a word to the next line of the other column
        [_OFFSET_COLUMNS, (2480, 1753), [[11, 12, 13], [0, 1, 2, 3, 4], [14, 15, 16, 17, 18], [5, 6, 7, 8, 9, 10]]],
        # A line rising by twice its height (perspective)
        [_slanted_lines(-0.007, [0.485]) + _slanted_lines(0, [0.7]), None, [list(range(8)), list(range(8, 16))]],
        # About 2 degrees of skew: the next line starts before the end of the current one
        [_slanted_lines(0.0045, [0.3, 0.339]), None, [list(range(8)), list(range(8, 16))]],
        [_slanted_lines(-0.0045, [0.3, 0.339]), None, [list(range(8)), list(range(8, 16))]],
        # 3 columns whose baselines are slightly offset
        [
            [[0.05, 0.19, 0.12, 0.21], [0.13, 0.19, 0.2, 0.21], [0.21, 0.19, 0.26, 0.21]]
            + [[0.36, 0.2005, 0.44, 0.2205], [0.45, 0.2005, 0.5, 0.2205], [0.51, 0.2005, 0.56, 0.2205]]
            + [[0.66, 0.194, 0.74, 0.214], [0.75, 0.197, 0.83, 0.217]],
            None,
            [[0, 1, 2], [6, 7], [3, 4, 5]],
        ],
        # A short word ending higher than its line is level with the label of the previous column (form)
        [
            [[0.1179, 0.2285, 0.1956, 0.2461], [0.1956, 0.2295, 0.2241, 0.2451]]  # "source OF"
            + [[0.3122, 0.2383, 0.3549, 0.2529], [0.3575, 0.2383, 0.4327, 0.252], [0.4378, 0.2363, 0.5518, 0.2529]]
            + [[0.557, 0.2373, 0.579, 0.249]],  # "HALL RODECAP, NOBLESVILLE, IN"
            (1000, 754),
            [[0, 1], [2, 3, 4, 5]],
        ],
        # Table cells separated by one word height: one line per row with the default `paragraph_break`
        [_table_rows(), (2339, 1654), [list(range(8 * row, 8 * row + 8)) for row in range(3)]],
        # Large type on a narrow crop: a word gap wider than `paragraph_break` but smaller than a word height
        [_justified_lines(), (643, 468), [list(range(6)), list(range(6, 12)), list(range(12, 18))]],
        # The page-relative paragraph break still applies to regular type
        [_justified_lines(0.2), (643, 468), [[0, 1, 2], [3, 4, 5], list(range(6, 12)), list(range(12, 18))]],
        # A box nested in a wider one (e.g. a punctuation mark detected twice) does not inflate the gap to the next word
        [[[0.10, 0.1, 0.30, 0.14], [0.25, 0.11, 0.26, 0.13], [0.32, 0.1, 0.45, 0.14]], None, [[0, 1, 2]]],
        # A figure label right below another one is not part of its row, even when level with a row further right
        [
            [[0.6327, 0.2393, 0.6656, 0.2529], [0.6289, 0.2471, 0.6693, 0.2607]]  # "Image" above "Encoder"
            + [[0.7098, 0.2432, 0.7199, 0.2568], [0.7401, 0.2432, 0.7629, 0.2568]],  # matrix cells
            (1584, 1224),
            [[0], [2, 3], [1]],
        ],
        # A word overlapping the previous one a little, a superscript or the parts of a fraction stay in their line
        [
            [[0.1, 0.5, 0.2, 0.53], [0.24, 0.5, 0.34, 0.53], [0.38, 0.49, 0.48, 0.52], [0.47, 0.51, 0.57, 0.54]],
            None,
            [[0], [1], [2, 3]],
        ],
        [
            [[0.1, 0.5, 0.2, 0.53], [0.24, 0.5, 0.34, 0.53], [0.38, 0.51, 0.48, 0.54], [0.465, 0.499, 0.485, 0.511]],
            None,
            [[0], [1], [2, 3]],
        ],
        [
            [[0.1, 0.5, 0.2, 0.53], [0.205, 0.5, 0.3, 0.53], [0.305, 0.49, 0.335, 0.516], [0.31, 0.514, 0.34, 0.54]]
            + [[0.345, 0.5, 0.44, 0.53]],
            None,
            [list(range(5))],
        ],
        # Nested boxes do not inflate the word gaps from which the break distance is estimated
        [
            [[0.1 + 0.045 * i + 0.02 * (i >= 5), 0.5, 0.14 + 0.045 * i + 0.02 * (i >= 5), 0.51] for i in range(10)]
            + [
                [0.11 + 0.045 * i + 0.02 * (i >= 5), 0.5, 0.12 + 0.045 * i + 0.02 * (i >= 5), 0.51]
                for i in (0, 1, 2, 5, 6, 7)
            ],
            None,
            [[0, 10, 1, 11, 2, 12, 3, 4], [5, 13, 6, 14, 7, 15, 8, 9]],
        ],
        # Boxes with zero height cannot be grouped into lines (no median height)
        [[[0.5, 0.2, 0.6, 0.2], [0.1, 0.2, 0.2, 0.2]], None, [[1], [0]]],
    ],
)
def test_resolve_lines(input_boxes, shape, lines):
    doc_builder = builder.DocumentBuilder()
    assert doc_builder._resolve_lines(np.asarray(input_boxes), shape) == lines


def test_table_geometry_helpers():
    polys = np.array(
        [
            [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3]],  # axis-aligned quad
            [[0.5, 0.5], [0.8, 0.6], [0.7, 0.9], [0.45, 0.8]],  # rotated quad
        ],
        dtype=np.float32,
    )
    points = np.array([[0.2, 0.2], [0.6, 0.7], [0.95, 0.95], [0.05, 0.05]], dtype=np.float32)
    inside = builder.DocumentBuilder._points_in_polygons(points, polys)
    assert inside.tolist() == [[True, False], [False, True], [False, False], [False, False]]
    # empty inputs yield empty masks instead of raising
    assert builder.DocumentBuilder._points_in_polygons(np.zeros((0, 2)), polys).shape == (0, 2)
    assert builder.DocumentBuilder._points_in_polygons(points, np.zeros((0, 4, 2))).shape == (4, 0)
    # flat straight box (xmin, ymin, xmax, ymax) -> (4, 2) polygon
    poly = builder.DocumentBuilder._as_cell_polygon([0.1, 0.2, 0.5, 0.6])
    assert poly.shape == (4, 2)
    assert np.allclose(poly, [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]])
    # (4, 2) polygons pass through unchanged
    quad = np.array([[0.1, 0.2], [0.5, 0.25], [0.45, 0.6], [0.05, 0.55]], dtype=np.float32)
    assert np.allclose(builder.DocumentBuilder._as_cell_polygon(quad), quad)


def _table_cell(x0, y0, x1, y1, rs, re, cs, ce, score=0.9):
    return {
        "geometry": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "score": score,
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
    }


# Word centers: a 2 x 2 table at the top, a 1 x 2 table below it and a caption outside both
_TABLE_WORDS = {
    "Name": (0.17, 0.15),
    "Age": (0.36, 0.15),
    "Alice": (0.17, 0.27),
    "30": (0.36, 0.27),
    "City": (0.17, 0.63),
    "Pop": (0.36, 0.63),
    "caption": (0.30, 0.92),
}


def _table_words(*names, confidences=None):
    boxes = [[cx - 0.02, cy - 0.01, cx + 0.02, cy + 0.01] for cx, cy in (_TABLE_WORDS[name] for name in names)]
    return _word_args(names, boxes, shape=(100, 100), confidences=confidences)


def test_documentbuilder_tables():
    doc_builder = builder.DocumentBuilder()
    # The OCR pipeline passes a list of grids (one per cropped table region), in page-relative coordinates.
    # The bottom table uses offset (1-based) logical coordinates to exercise local re-indexing.
    table_top = {
        "cells": [
            _table_cell(0.10, 0.10, 0.25, 0.20, 0, 0, 0, 0),
            _table_cell(0.28, 0.10, 0.45, 0.20, 0, 0, 1, 1),
            _table_cell(0.10, 0.22, 0.25, 0.32, 1, 1, 0, 0),
            _table_cell(0.28, 0.22, 0.45, 0.32, 1, 1, 1, 1),
        ],
        "num_rows": 2,
        "num_cols": 2,
    }
    table_bottom = {
        "cells": [_table_cell(0.10, 0.58, 0.25, 0.68, 1, 1, 1, 1), _table_cell(0.28, 0.58, 0.45, 0.68, 1, 1, 2, 2)],
        "num_rows": 99,  # deliberately wrong dims -> recomputed from local coordinates
        "num_cols": 99,
    }
    args = _table_words(*_TABLE_WORDS)
    page = doc_builder(*args, tables=[[table_top, table_bottom]]).pages[0]
    # One Table per provided grid
    assert len(page.tables) == 2 and all(isinstance(table, Table) for table in page.tables)
    assert page.tables[0].to_grid() == [["Name", "Age"], ["Alice", "30"]]
    # bottom table re-indexed from offset coordinates to a local 0-based 1 x 2 grid
    assert (page.tables[1].num_rows, page.tables[1].num_cols) == (1, 2)
    assert page.tables[1].to_grid() == [["City", "Pop"]]
    # Words assigned to a table are removed from the blocks; the caption remains
    assert _words(page) == ["caption"]
    # Tables are part of the page export
    assert len(page.export()["tables"]) == 2
    # A single grid (dict) is also accepted -> one table
    assert len(doc_builder(*_table_words("Name", "Age", "Alice", "30"), tables=[table_top]).pages[0].tables) == 1
    # No tables -> empty page.tables and every word is kept in the blocks
    page = doc_builder(*args).pages[0]
    assert page.tables == [] and page.export()["tables"] == []
    assert sorted(_words(page)) == sorted(_TABLE_WORDS)

    # Straight-mode table predictions store cells as flat (xmin, ymin, xmax, ymax) boxes
    # Regression test: this format used to crash the word-to-cell assignment with an IndexError
    straight = {
        "cells": [
            {**_table_cell(0, 0, 0, 0, 0, 0, col, col), "geometry": box}
            for col, box in enumerate([[0.10, 0.10, 0.25, 0.20], [0.28, 0.10, 0.45, 0.20]])
        ],
        "score": 0.9,
    }
    page = doc_builder(*_table_words("Name", "Age", "caption"), tables=[[straight]]).pages[0]
    assert len(page.tables) == 1 and page.tables[0].to_grid() == [["Name", "Age"]]
    assert _words(page) == ["caption"]

    # Each cell keeps the score of the table structure model, next to the recognition confidence of its words
    table = {
        "cells": [
            _table_cell(0.10, 0.10, 0.25, 0.20, 0, 0, 0, 0, score=0.8),
            _table_cell(0.28, 0.10, 0.45, 0.20, 0, 0, 1, 1, score=0.6),
            _table_cell(0.10, 0.22, 0.25, 0.32, 1, 1, 0, 0, score=0.4),  # no word inside this cell
        ],
        "num_rows": 2,
        "num_cols": 2,
    }
    page = doc_builder(*_table_words("Name", "Age", confidences=[0.95, 0.85]), tables=[table]).pages[0]
    cells = page.tables[0].cells
    assert [cell.value for cell in cells] == ["Name", "Age", ""]
    assert [cell.objectness_score for cell in cells] == [0.8, 0.6, 0.4]
    assert [cell.confidence for cell in cells] == pytest.approx([0.95, 0.85, 0.4])
    assert page.tables[0].confidence == pytest.approx(0.6)
    assert [cell["objectness_score"] for cell in page.export()["tables"][0]["cells"]] == [0.8, 0.6, 0.4]

    # Multi-word cells are read line by line, left to right
    words = ["world", "hello", "second", "line", "solo"]
    boxes = [
        [0.20, 0.13, 0.28, 0.16],  # "world"  - first line, right
        [0.11, 0.13, 0.19, 0.16],  # "hello"  - first line, left
        [0.11, 0.17, 0.20, 0.20],  # "second" - second line, left
        [0.21, 0.17, 0.27, 0.20],  # "line"   - second line, right
        [0.36, 0.13, 0.44, 0.16],  # "solo"   - other cell
    ]
    table = {
        "cells": [_table_cell(0.10, 0.10, 0.30, 0.22, 0, 0, 0, 0), _table_cell(0.32, 0.10, 0.50, 0.22, 0, 0, 1, 1)],
        "num_rows": 1,
        "num_cols": 2,
    }
    page = doc_builder(*_word_args(words, boxes), tables=[[table]]).pages[0]
    assert len(page.tables) == 1 and page.tables[0].to_grid() == [["hello world second line", "solo"]]
    # Every word ended up in the table, so nothing is left in the regular blocks
    assert page.blocks == []
    # A degenerate cell whose words all share the same center still produces a single row
    table = {"cells": [_table_cell(0.10, 0.10, 0.30, 0.22, 0, 0, 0, 0)], "num_rows": 1, "num_cols": 1}
    boxes = [[0.15, 0.15, 0.15, 0.15], [0.16, 0.15, 0.16, 0.15]]
    assert doc_builder(*_word_args(["a", "b"], boxes), tables=[[table]]).pages[0].tables[0].to_grid() == [["a b"]]

    # A word in the gutter goes to its nearest cell, a word far away from the table stays in the regular blocks
    words = ["left", "right", "outside", "faraway"]
    boxes = [
        [0.11, 0.12, 0.19, 0.16],  # inside cell 0
        [0.33, 0.12, 0.41, 0.16],  # inside cell 1
        [0.305, 0.12, 0.315, 0.16],  # in the gutter, closest to cell 1
        [0.11, 0.80, 0.25, 0.85],  # far outside the table
    ]
    table = {
        "cells": [_table_cell(0.10, 0.10, 0.30, 0.18, 0, 0, 0, 0), _table_cell(0.32, 0.10, 0.50, 0.18, 0, 0, 1, 1)],
        "num_rows": 1,
        "num_cols": 2,
    }
    page = doc_builder(*_word_args(words, boxes), tables=[[table]]).pages[0]
    grid = page.tables[0].to_grid()
    assert grid[0][0] == "left" and "outside" in grid[0][1]
    assert _words(page) == ["faraway"]

    # Rotated pages keep 4-point polygons for the table and its cells
    cell_boxes = [
        (0.10, 0.10, 0.28, 0.20),
        (0.30, 0.10, 0.48, 0.20),
        (0.10, 0.22, 0.28, 0.32),
        (0.30, 0.22, 0.48, 0.32),
    ]
    word_boxes = [
        (0.13, 0.13, 0.25, 0.17),
        (0.33, 0.13, 0.45, 0.17),
        (0.13, 0.25, 0.25, 0.29),
        (0.33, 0.25, 0.45, 0.29),
    ]
    table = {
        "cells": [
            {**_table_cell(0, 0, 0, 0, rs, re, cs, ce), "geometry": _rot_poly(*box, 20)}
            for box, (rs, re, cs, ce) in zip(cell_boxes, [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 1, 1)])
        ],
        "num_rows": 2,
        "num_cols": 2,
    }
    polys = [_rot_poly(*box, 20) for box in word_boxes]
    page = doc_builder(*_word_args(["Name", "Age", "Alice", "30"], polys), tables=[[table]]).pages[0]
    assert len(page.tables) == 1 and len(page.tables[0].geometry) == 4
    assert all(len(cell.geometry) == 4 for cell in page.tables[0].cells)
    assert page.tables[0].to_grid() == [["Name", "Age"], ["Alice", "30"]]
    # The export is still JSON-serializable
    json.dumps(page.export())


@pytest.mark.parametrize(
    "words, boxes, grids, tables",
    [
        # A table prediction with no cells (e.g. a false-positive "Table" region where the table model finds nothing)
        # is skipped: the words stay in the regular blocks since they were never consumed by a table
        [["hello"], [[0.1, 0.1, 0.2, 0.2]], [], [{"cells": [], "num_rows": 0, "num_cols": 0}]],
        [["body", "text"], [[0.10, 0.10, 0.25, 0.15], [0.30, 0.10, 0.45, 0.15]], [], [{"cells": []}]],
        # A detected table grid on a page without any recognized word
        [[], np.zeros((0, 4)), [[[""]]], [{"cells": [_table_cell(0.10, 0.10, 0.30, 0.20, 0, 0, 0, 0)]}]],
    ],
)
def test_documentbuilder_tables_without_content(words, boxes, grids, tables):
    page = builder.DocumentBuilder()(*_word_args(words, boxes, shape=(100, 100)), tables=[tables]).pages[0]
    assert [table.to_grid() for table in page.tables] == grids
    assert _words(page) == words
    if not words:
        assert page.blocks == [] and page.render() == ""


def test_documentbuilder_keep_reading_order():
    # Two columns of 3 lines each: a naive top-down sort interleaves the columns
    words = ["L0", "L1", "L2", "R0", "R1", "R2"]
    boxes = [[x, 0.1 + 0.2 * idx, x + 0.2, 0.2 + 0.2 * idx] for x in (0.1, 0.6) for idx in range(3)]
    args = _word_args(words, boxes, shape=(100, 100))
    page = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=True)(*args).pages[0]
    assert page.render(block_break=" ").split() == words
    # The flag reorders the stored blocks themselves
    assert _words(page) == words
    # Without the flag, the stored blocks keep their original (interleaved) order, but the exports are ordered
    page = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=False)(*args).pages[0]
    assert _words(page) != words
    assert page.render(block_break=" ").split() == words
    assert [word["value"] for b in page.export()["blocks"] for ln in b["lines"] for word in ln["words"]] == words

    # Layout regions are used to place page furniture: the top line is labeled as a page footer -> emitted last
    boxes = [[0.1, 0.05, 0.9, 0.1], [0.1, 0.3, 0.9, 0.4], [0.1, 0.5, 0.9, 0.6]]
    regions = {"boxes": np.asarray([[0.05, 0.02, 0.95, 0.15]]), "class_names": ["Page-footer"], "scores": [0.9]}
    doc_builder = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=True)
    page = doc_builder(*_word_args(["footer", "first", "second"], boxes, shape=(100, 100)), regions=[regions]).pages[0]
    assert page.render(block_break=" ").split() == ["first", "second", "footer"]

    # The stored order, render() and Markdown agree: no conflicting re-sort across the exports
    boxes = [[x, 0.15 + 0.03 * r, x + 0.37, 0.17 + 0.03 * r] for x in (0.08, 0.55) for r in range(3)]
    boxes.append([0.3, 0.05, 0.7, 0.08])  # a footer, geometrically at the top
    regions = {
        "boxes": np.asarray([[0.06, 0.13, 0.47, 0.26], [0.53, 0.13, 0.94, 0.26], [0.25, 0.03, 0.75, 0.10]]),
        "class_names": ["Text", "Text", "Page-footer"],
        "scores": [0.9, 0.9, 0.9],
    }
    expected = [*words, "foot"]
    page = doc_builder(*_word_args(expected, boxes, shape=(100, 100)), regions=[regions]).pages[0]
    assert _words(page) == expected
    # render() follows the stored order and is idempotent
    assert page.render(block_break=" ").split() == expected == page.render(block_break=" ").split()
    # markdown re-derives reading order from the lines and lands on the same order
    assert [token for token in page.export_as_markdown().replace("#", " ").split() if token in expected] == expected

    # The builder stores its linearization back in the page cache: the figures must be kept in it
    args = _word_args(
        ["intro", "inner", "caption", "outro"],
        [[0.1, 0.05, 0.9, 0.1], [0.3, 0.4, 0.5, 0.45], [0.2, 0.65, 0.8, 0.7], [0.1, 0.8, 0.9, 0.85]],
        shape=(100, 100),
    )
    args[0][0][30:60, 10:90] = 255
    regions = {
        "boxes": np.asarray([[0.08, 0.28, 0.92, 0.62], [0.15, 0.63, 0.85, 0.72]]),
        "class_names": ["Picture", "Caption"],
        "scores": [0.9, 0.9],
    }
    exports = {}
    for keep in (False, True):
        page = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=keep)(*args, regions=[regions]).pages[0]
        assert len(page.items_in_reading_order(include_figures=True)) > len(page.items_in_reading_order())
        exports[keep] = page.export_as_markdown(images="embedded")
    assert exports[True] == exports[False]
    assert exports[True].count("![caption](data:image/png;base64,") == 1
    assert "inner" not in exports[True]


@pytest.mark.parametrize("resolve_lines, resolve_blocks", [(True, True), (True, False), (False, True), (False, False)])
def test_documentbuilder_keep_reading_order_flags(resolve_lines, resolve_blocks):
    doc_builder = builder.DocumentBuilder(
        resolve_lines=resolve_lines, resolve_blocks=resolve_blocks, keep_reading_order=True
    )
    # Two columns, three rows each, two words per row; shuffled so the input order is not the reading order
    cells = []
    for column, tag in ((0.08, "L"), (0.55, "R")):
        for row in range(3):
            y = 0.12 + 0.04 * row
            cells.append(([column, y, column + 0.15, y + 0.02], f"{tag}{row}a"))
            cells.append(([column + 0.17, y, column + 0.34, y + 0.02], f"{tag}{row}b"))
    cells = [cells[idx] for idx in [7, 0, 11, 3, 5, 9, 1, 6, 10, 2, 8, 4]]
    args = _word_args([text for _, text in cells], [box for box, _ in cells], shape=(100, 100))
    columns = {
        "boxes": np.asarray([[0.06, 0.10, 0.45, 0.24], [0.53, 0.10, 0.92, 0.24]]),
        "class_names": ["Text", "Text"],
        "scores": [0.9, 0.9],
    }
    for regions in (None, columns):
        page = doc_builder(*args, regions=None if regions is None else [regions]).pages[0]
        assert page.render(block_break=" ").split() == "L0a L0b L1a L1b L2a L2b R0a R0b R1a R1b R2a R2b".split()
        if not resolve_lines and not resolve_blocks and regions is None:
            # lines are resolved internally only to order the content: each column collapses onto a single line
            assert all(len(block.lines) == 1 for block in page.blocks)

    # Layout regions place the page furniture, whatever its position on the page
    boxes = [
        [0.1, 0.05, 0.9, 0.09],  # footer text (near the top of the page)
        [0.1, 0.30, 0.9, 0.38],  # first body line
        [0.1, 0.45, 0.9, 0.53],  # second body line
        [0.1, 0.15, 0.9, 0.20],  # header text
    ]
    regions = {
        "boxes": np.asarray([[0.05, 0.03, 0.95, 0.11], [0.05, 0.13, 0.95, 0.22]]),
        "class_names": ["Page-footer", "Page-header"],
        "scores": [0.9, 0.9],
    }
    page = doc_builder(*_word_args(["foot", "first", "second", "head"], boxes, shape=(100, 100)), regions=[regions])
    assert page.pages[0].render(block_break=" ").split() == ["head", "first", "second", "foot"]


def test_documentbuilder_rotated_pages():
    # Two columns of words rotated by 25 degrees around the center of a non-square page are read column by column
    height, width = 1000, 800
    angle = np.deg2rad(25)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([width / 2, height / 2])

    def _rot_box(x0, y0, x1, y1):
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * [width, height]
        return ((pts - center) @ rot.T + center) / [width, height]

    words = ["L0", "L1", "L2", "R0", "R1", "R2"]
    polys = [_rot_box(x, 0.1 + 0.2 * idx, x + 0.2, 0.2 + 0.2 * idx) for x in (0.1, 0.6) for idx in range(3)]
    doc_builder = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=True)
    page = doc_builder(*_word_args(words, polys, shape=(height, width))).pages[0]
    assert page.render(block_break=" ").split() == words

    # Same without resolve_lines, for two words per row
    cells = []
    for column, tag in ((0.08, "L"), (0.55, "R")):
        for row in range(3):
            y = 0.15 + 0.04 * row
            cells.append((_rot_poly(column, y, column + 0.15, y + 0.02, 20), f"{tag}{row}a"))
            cells.append((_rot_poly(column + 0.17, y, column + 0.34, y + 0.02, 20), f"{tag}{row}b"))
    doc_builder = builder.DocumentBuilder(resolve_lines=False, resolve_blocks=False, keep_reading_order=True)
    page = doc_builder(*_word_args([text for _, text in cells], [box for box, _ in cells], shape=(100, 100))).pages[0]
    assert page.render(block_break=" ").split() == "L0a L0b L1a L1b L2a L2b R0a R0b R1a R1b R2a R2b".split()

    # Words rotated by 30 degrees mixed with an axis-aligned one
    polys = [_rot_poly(0.10, 0.10 + 0.12 * idx, 0.60, 0.16 + 0.12 * idx, 30) for idx in range(3)]
    polys.append([[0.10, 0.70], [0.60, 0.70], [0.60, 0.76], [0.10, 0.76]])  # not rotated at all
    page = builder.DocumentBuilder()(*_word_args(["r0", "r1", "r2", "flat"], polys)).pages[0]
    assert sorted(_words(page)) == ["flat", "r0", "r1", "r2"]
    json.dumps(page.export())
