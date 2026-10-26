import numpy as np
import pytest

from doctr.models.reading_order import (
    ReadingOrderPredictor,
    assign_layout_labels,
    deskew_reading_geometries,
    detect_text_direction,
    layout_label_role,
    normalize_layout_label,
    resolve_reading_segments,
    sort_reading_order,
)
from doctr.models.reading_order.base import _defer_floats, _region_groups


def _box(x0, y0, x1, y1):
    return ((x0, y0), (x1, y1))


def _column(x0, x1, y0, step, count, height=0.03):
    return [_box(x0, y0 + step * i, x1, y0 + step * i + height) for i in range(count)]


def _rotated_box(box, deg, width=800, height=1000):
    angle = np.deg2rad(deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([width / 2, height / 2])
    (x0, y0), (x1, y1) = box
    pts = np.array([[x0 * width, y0 * height], [x1 * width, y0 * height], [x1 * width, y1 * height]])
    pts = np.concatenate([pts, [[x0 * width, y1 * height]]])
    return ((pts - center) @ rot.T + center) / [width, height]


# Two columns of 4 lines
TWO_COLUMNS = _column(0.1, 0.45, 0.1, 0.2, 4, 0.15) + _column(0.55, 0.9, 0.1, 0.2, 4, 0.15)

# Page furniture, a figure with its caption, body text and a footnote
LABELED_PAGE = [
    _box(0.1, 0.92, 0.9, 0.97),  # 0: page footer
    _box(0.1, 0.02, 0.9, 0.06),  # 1: page header
    _box(0.1, 0.1, 0.9, 0.4),  # 2: body text
    _box(0.1, 0.45, 0.5, 0.7),  # 3: figure
    _box(0.1, 0.71, 0.5, 0.75),  # 4: caption below the figure
    _box(0.55, 0.45, 0.9, 0.88),  # 5: body on the right of the figure
    _box(0.1, 0.8, 0.5, 0.84),  # 6: footnote
]
PAGE_LABELS = ["Page-footer", "Page-header", "Text", "Picture", "Caption", "Text", "Footnote"]

# A page-wide figure with text inside, its caption, a figure in the left column with its caption, two columns of text
FIGURES_PAGE = [
    _box(0.10, 0.05, 0.90, 0.30),  # 0: figure (page wide)
    _box(0.15, 0.10, 0.30, 0.12),  # 1: text inside the figure, left side
    _box(0.70, 0.10, 0.85, 0.12),  # 2: text inside the figure, right side
    _box(0.10, 0.32, 0.90, 0.35),  # 3: its caption
    _box(0.10, 0.40, 0.45, 0.70),  # 4: figure in the left column
    _box(0.10, 0.72, 0.45, 0.80),  # 5: its caption
    *_column(0.55, 0.90, 0.40, 0.05, 8),  # 6-13: right column
    *_column(0.10, 0.45, 0.82, 0.04, 3),  # 14-16: left column
]
FIGURES_LABELS = ["Picture", "Picture", "Picture", "Caption", "Picture", "Caption"] + ["Text"] * 11

# Lines of a real receipt (CORD): the right-aligned amounts are level with their item, far from it
RECEIPT = [
    _box(0.096, 0.409, 0.478, 0.439),  # 0: Enting2 Dua Bandeng Ori
    _box(0.650, 0.407, 0.753, 0.431),  # 1: 31.000
    _box(0.094, 0.444, 0.178, 0.467),  # 2: 01069
    _box(0.243, 0.440, 0.411, 0.469),  # 3: 1 31.000
    _box(0.094, 0.471, 0.561, 0.500),  # 4: TAHU BAKSO Grg Ayam VC [biji
    _box(0.651, 0.470, 0.754, 0.492),  # 5: 68.000
    _box(0.091, 0.506, 0.174, 0.529),  # 6: 07165
    _box(0.243, 0.504, 0.411, 0.529),  # 7: 10 6.800
    _box(0.087, 0.564, 0.430, 0.593),  # 8: QTY 11 ITEMS 2
    _box(0.088, 0.599, 0.170, 0.622),  # 9: TOTAL
    _box(0.635, 0.597, 0.741, 0.619),  # 10: 99.000
    _box(0.086, 0.629, 0.172, 0.655),  # 11: KARTU
    _box(0.637, 0.629, 0.742, 0.652),  # 12: 99.000
]

# Receipt totals on a crumpled page (pixels): each amount sits lower than its label and also overlaps the label of the
# next row, but it overlaps its own label the most
CRUMPLED_TOTALS = [_box(60, 1200 + 60 * i, 600, 1240 + 60 * i) for i in range(3)] + [
    _box(114, 1376, 282, 1420),  # 3: label
    _box(420, 1388, 520, 1431),  # 4: its amount
    _box(112, 1408, 226, 1451),  # 5: label
    _box(420, 1422, 520, 1465),  # 6: its amount
    _box(110, 1445, 162, 1483),  # 7: label
    _box(410, 1456, 520, 1502),  # 8: its amount
    _box(110, 1481, 162, 1518),  # 9: last label
    *[_box(60, 1560 + 60 * i, 600, 1600 + 60 * i) for i in range(2)],  # 10-11: full-width lines
]

# A 3-line caption whose last line is closer to the table of the other column
SPLIT_CAPTION = [
    _box(0.526, 0.115, 0.903, 0.33),  # 0: figure
    _box(0.102, 0.248, 0.477, 0.443),  # 1: table
    _box(0.52, 0.352, 0.91, 0.366),  # 2: caption line 1
    _box(0.52, 0.37, 0.91, 0.384),  # 3: caption line 2
    _box(0.52, 0.419, 0.73, 0.434),  # 4: caption line 3
]
SPLIT_CAPTION_LABELS = ["Picture", "Table", "Caption", "Caption", "Caption"]


def _paragraph_beside_a_figure():
    """A single-column page (3-column author grid) with a paragraph ending one line below the figure beside it"""
    boxes, labels, groups = [[0.1, 0.05, 0.9, 0.09]], ["Title"], [0]
    for row in range(4):
        for x in (0.15, 0.41, 0.68):
            boxes.append([x, 0.14 + row * 0.03, x + 0.18, 0.154 + row * 0.03])
            labels.append(None)
            groups.append(-1)
    for y in np.linspace(0.30, 0.64, 12):
        boxes.append([0.08, y, 0.48, y + 0.014])
        labels.append("Text")
        groups.append(1)
    boxes += [[0.08, 0.662, 0.35, 0.676], [0.52, 0.31, 0.91, 0.667], [0.08, 0.70, 0.30, 0.712]]
    labels += ["Text", "Picture", "Section-header"]  # last line of the paragraph (below the figure), figure, header
    groups += [1, 2, 3]
    return [_box(*box) for box in boxes], labels, groups


BESIDE, BESIDE_LABELS, BESIDE_GROUPS = _paragraph_beside_a_figure()
LAST_LINE, FIGURE = len(BESIDE) - 3, len(BESIDE) - 2

# Two text columns on the right half and a wide figure on the left that does not vote
VOTERS_PAGE = [
    box for y in np.arange(6) * 0.05 + 0.1 for box in (_box(0.55, y, 0.72, y + 0.02), _box(0.78, y, 0.95, y + 0.02))
]
VOTERS_PAGE.append(_box(0.02, 0.1, 0.5, 0.4))


def _flat(segments):
    return [idx for segment in segments for idx in segment]


def _before(*groups):
    """Check that each group of indices is read entirely before the next one"""

    def check(order):
        return all(
            max(order.index(i) for i in group) < min(order.index(i) for i in nxt)
            for group, nxt in zip(groups, groups[1:])
        )

    return check


@pytest.mark.parametrize(
    "texts, language, expected",
    [
        (["Hello", "world"], None, "ltr"),
        (["Die schnelle braune Katze"], None, "ltr"),
        (["Привет мир"], None, "ltr"),
        (["こんにちは世界"], None, "ltr"),
        (["مرحبا", "بالعالم"], None, "rtl"),
        (["שלום עולם"], None, "rtl"),
        (["سلام دنیا"], None, "rtl"),  # Persian
        (["مرحبا Hello بالعالم العربي"], None, "rtl"),  # mixed, RTL-dominant
        (["Hello world مرحبا"], None, "ltr"),  # mixed, LTR-dominant
        (["123", "456"], None, "ltr"),  # no strong character, no hint
        (["123", "456"], "ar", "rtl"),  # no strong character, language hint
        (["123"], "he", "rtl"),
        (["123"], "fr", "ltr"),
        ([], None, "ltr"),
    ],
)
def test_detect_text_direction(texts, language, expected):
    assert detect_text_direction(texts, language=language) == expected


@pytest.mark.parametrize(
    "label, normalized, role",
    [
        ("Page-header", "page_header", "header"),
        (" Section header ", "section_header", "body"),
        ("Page-footer", "page_footer", "footer"),
        ("Footnote", "footnote", "footnote"),
        ("Caption", "caption", "caption"),
        ("Table", "table", "float"),
        ("Picture", "picture", "float"),
        ("Text", "text", "body"),
        ("Title", "title", "body"),
        (None, "", "body"),
    ],
)
def test_layout_labels(label, normalized, role):
    assert normalize_layout_label(label) == normalized
    assert layout_label_role(label) == role


@pytest.mark.parametrize(
    "geoms, kwargs, expected",
    [
        pytest.param([], {}, [], id="empty"),
        pytest.param([_box(0.1, 0.1, 0.2, 0.2)], {}, [0], id="single"),
        pytest.param([_box(0.1, 0.5, 0.9, 0.6), _box(0.1, 0.1, 0.9, 0.2), _box(0.1, 0.3, 0.9, 0.4)], {}, [1, 2, 0]),
        pytest.param(
            [_box(0.55, 0.2, 0.9, 0.8), _box(0.1, 0.05, 0.9, 0.15), _box(0.1, 0.2, 0.45, 0.8)],
            {},
            [1, 2, 0],
            id="title-over-columns",
        ),
        pytest.param(TWO_COLUMNS, {}, list(range(8)), id="columns"),
        pytest.param(TWO_COLUMNS, {"direction": "rtl"}, [4, 5, 6, 7, 0, 1, 2, 3], id="columns-rtl"),
        pytest.param(
            [_box(0.1, 0.02, 0.9, 0.06), _box(0.05, 0.1, 0.95, 0.6)]
            + _column(0.1, 0.4, 0.7, 0.05, 3)
            + _column(0.6, 0.9, 0.7, 0.05, 3),
            {"column_voters": [True, False] + [True] * 6},
            list(range(8)),
            id="non-voting-page-wide-figure",
        ),
        pytest.param(
            # A centered title crossing the gutter does not merge the columns: the column of a figure is read to its end
            [_box(0.3, 0.02, 0.7, 0.06), _box(0.1, 0.1, 0.45, 0.13), _box(0.1, 0.16, 0.13, 0.19)]
            + [_box(0.25, 0.22, 0.3, 0.25)]  # figure labels on 2 sides
            + _column(0.1, 0.45, 0.28, 0.05, 2)
            + _column(0.55, 0.9, 0.1, 0.07, 4),
            {},
            list(range(10)),
            id="title-crossing-the-gutter",
        ),
        pytest.param(
            [_box(0.8 - 0.15 * i, 0.1, 0.9 - 0.15 * i, 0.9) for i in range(4)],
            {"direction": "ttb-rtl"},
            [0, 1, 2, 3],
            id="vertical-rtl",
        ),
        pytest.param(
            [_box(0.8 - 0.15 * i, 0.1, 0.9 - 0.15 * i, 0.9) for i in range(4)],
            {"direction": "ttb-ltr"},
            [3, 2, 1, 0],
            id="vertical-ltr",
        ),
        pytest.param(
            [_box(0.8, 0.5, 0.9, 0.9), _box(0.8, 0.1, 0.9, 0.45), _box(0.6, 0.1, 0.7, 0.9)],
            {"direction": "ttb-rtl"},
            [1, 0, 2],
            id="vertical-stacked",
        ),
        pytest.param(LABELED_PAGE, {"labels": PAGE_LABELS}, [1, 2, 3, 4, 5, 6, 0], id="furniture-and-caption"),
        pytest.param(
            [*LABELED_PAGE[:4], _box(0.1, 0.41, 0.5, 0.44), *LABELED_PAGE[5:]],
            {"labels": PAGE_LABELS},
            [1, 2, 4, 3, 5, 6, 0],
            id="caption-above",
        ),
        pytest.param(
            # A caption too far from any (shrunk) float keeps its natural position in the body
            [*LABELED_PAGE[:3], _box(0.1, 0.45, 0.5, 0.5), _box(0.55, 0.02, 0.9, 0.06), *LABELED_PAGE[5:]],
            {"labels": PAGE_LABELS},
            _before([4], [3]),
            id="caption-out-of-reach",
        ),
        pytest.param(
            # The text inside a figure is read right after it, before its caption
            FIGURES_PAGE,
            {"labels": FIGURES_LABELS},
            [0, 1, 2, 3, 4, 5, 14, 15, 16, *range(6, 14)],
            id="float-members",
        ),
        pytest.param(
            FIGURES_PAGE,
            {"labels": [*FIGURES_LABELS[:2], "Text", *FIGURES_LABELS[3:]]},
            lambda order: order[:4] == [0, 1, 2, 3],
            id="float-members-labeled-as-text",
        ),
        pytest.param(FIGURES_PAGE, {}, lambda order: True, id="float-members-without-labels"),
        pytest.param(
            [_box(0.1, 0.1, 0.1, 0.1)] * 3 + [_box(0.5, 0.5, 0.5, 0.5)],
            {},
            lambda order: sorted(order) == list(range(4)),
            id="degenerate",
        ),
        pytest.param(
            # A line split into a narrow part and a stray fragment does not interrupt its column
            _column(0.10, 0.45, 0.10, 0.04, 1)
            + [_box(0.10, 0.14, 0.25, 0.17), _box(0.34, 0.14, 0.45, 0.17)]
            + _column(0.10, 0.45, 0.18, 0.04, 3)
            + _column(0.55, 0.90, 0.10, 0.04, 6),
            {},
            _before(range(6), range(6, 12)),
            id="fragmented-columns",
        ),
        pytest.param(
            [_box(0.35, 0.05, 0.65, 0.10), _box(0.10, 0.15, 0.45, 0.20)]  # element bridging both columns
            + [_box(0.10, 0.22, 0.16, 0.27), _box(0.17, 0.22, 0.24, 0.27), _box(0.25, 0.22, 0.45, 0.27)]  # 3 fragments
            + [_box(0.10, 0.29, 0.45, 0.34), _box(0.55, 0.15, 0.90, 0.20), _box(0.55, 0.22, 0.90, 0.27)],
            {},
            list(range(8)),
            id="fragmented-row",
        ),
        pytest.param(
            _column(0.05, 0.95, 0.02, 0.06, 3, 0.04)
            + [
                box
                for y in (0.20, 0.26, 0.32)
                for box in (_box(0.05, y, 0.30, y + 0.04), _box(0.65, y, 0.95, y + 0.04))
            ]
            + _column(0.05, 0.95, 0.38, 0.06, 2, 0.04),
            {},
            list(range(11)),
            id="key-value-rows",
        ),
        pytest.param(CRUMPLED_TOTALS, {}, list(range(12)), id="crumpled-receipt-totals"),
        pytest.param(
            # The cells of a single row are still read left to right when the middle one sits a little lower
            [_box(100, 100, 200, 120), _box(300, 103, 400, 123), _box(500, 100, 600, 120)],
            {},
            [0, 1, 2],
            id="shifted-cell",
        ),
        pytest.param(
            [_box(100, 100, 200, 120), _box(300, 103, 400, 123), _box(500, 100, 600, 120)]
            + [_box(100, 140, 200, 160), _box(300, 140, 400, 160)],
            {},
            list(range(5)),
            id="shifted-cell-next-row",
        ),
        pytest.param(RECEIPT, {}, list(range(13)), id="receipt-amounts"),
        pytest.param(
            [_box(1 - x1, y0, 1 - x0, y1) for (x0, y0), (x1, y1) in RECEIPT],
            {"direction": "rtl"},
            list(range(13)),
            id="receipt-amounts-rtl",
        ),
        pytest.param(
            # Text columns with the same gutter are still read column after column: their lines are long
            _column(0.05, 0.40, 0.10, 0.03, 4, 0.02) + _column(0.55, 0.90, 0.10, 0.03, 4, 0.02),
            {},
            list(range(8)),
            id="columns-with-a-wide-gutter",
        ),
        pytest.param(
            # So is a column of short lines on a page with many lines (e.g. a list beside a text column)
            _column(0.05, 0.40, 0.02, 0.012, 40, 0.01) + _column(0.55, 0.60, 0.02, 0.012, 40, 0.01),
            {},
            list(range(80)),
            id="short-lines-column",
        ),
        pytest.param(
            [_box(0.05, 0.02, 0.95, 0.06)]
            + _column(0.05, 0.45, 0.10, 0.06, 3, 0.04)
            + _column(0.55, 0.95, 0.10, 0.06, 3, 0.04),
            {},
            list(range(7)),
            id="header-over-columns",
        ),
        pytest.param(
            SPLIT_CAPTION,
            {"labels": SPLIT_CAPTION_LABELS},
            lambda order: order.index(4) == order.index(1) + 1,
            id="split-caption",
        ),
        pytest.param(
            # Grouped, the caption lines follow the float closest to their union
            SPLIT_CAPTION,
            {"labels": SPLIT_CAPTION_LABELS, "region_groups": [-1, -1, 7, 7, 7]},
            lambda order: order[order.index(0) + 1 : order.index(0) + 4] == [2, 3, 4],
            id="grouped-caption",
        ),
        pytest.param(
            # Without region groups, the figure splits the paragraph beside it...
            BESIDE,
            {"labels": BESIDE_LABELS},
            _before([FIGURE], [LAST_LINE]),
            id="float-splits-a-region",
        ),
        pytest.param(
            # ... but not with them
            BESIDE,
            {"labels": BESIDE_LABELS, "region_groups": BESIDE_GROUPS},
            lambda order: (
                _before([LAST_LINE], [FIGURE], [len(BESIDE) - 1])(order) and sorted(order) == list(range(len(BESIDE)))
            ),
            id="float-beside-a-region",
        ),
        pytest.param(
            # A text region enclosing a figure: the figure stays between the lines around it
            _column(0.1, 0.9, 0.1, 0.04, 3) + [_box(0.2, 0.3, 0.8, 0.6)] + _column(0.1, 0.9, 0.7, 0.04, 3),
            {"labels": ["Text"] * 3 + ["Picture"] + ["Text"] * 3, "region_groups": [0, 0, 0, 1, 0, 0, 0]},
            list(range(7)),
            id="float-stacked-in-a-loose-region",
        ),
        pytest.param(
            # A caption out of reach of any float is read in its own column
            _column(0.08, 0.46, 0.1, 0.05, 8)
            + [_box(0.54, 0.1, 0.92, 0.3), _box(0.54, 0.45, 0.92, 0.48)]
            + _column(0.54, 0.92, 0.55, 0.05, 3),
            {"labels": ["Text"] * 8 + ["Picture", "Caption"] + ["Text"] * 3},
            [*range(8), 8, 9, 10, 11, 12],
            id="caption-stays-in-its-column",
        ),
        pytest.param(
            VOTERS_PAGE,
            {"labels": ["Text"] * 12 + ["Picture"], "column_voters": [True] * 12 + [False]},
            [12, 0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
            id="voters-define-the-search-window",
        ),
    ],
)
def test_sort_reading_order(geoms, kwargs, expected):
    order = sort_reading_order(geoms, **kwargs)
    assert expected(order) if callable(expected) else order == expected
    assert sorted(order) == list(range(len(geoms)))


def test_sort_reading_order_inputs():
    # The result is independent of the input ordering
    rng = np.random.default_rng(42)
    for _ in range(5):
        perm = rng.permutation(8).tolist()
        order = sort_reading_order([TWO_COLUMNS[idx] for idx in perm])
        assert [perm[idx] for idx in order] == list(range(8))
    # (N, 4) arrays, (N, 4, 2) polygons (straight polygons are untouched) and absolute coordinates
    boxes = _column(0.1, 0.45, 0.1, 0.2, 2, 0.15) + _column(0.55, 0.9, 0.1, 0.2, 2, 0.15)
    as_array = np.asarray([(x0, y0, x1, y1) for ((x0, y0), (x1, y1)) in boxes])
    as_polys = np.asarray([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for ((x0, y0), (x1, y1)) in boxes])
    for geoms in (as_array, as_polys, as_array * 1000):
        assert sort_reading_order(geoms) == [0, 1, 2, 3]
    # Rotated pages are de-skewed
    geoms = [_box(0.1, 0.06, 0.9, 0.09)] + _column(0.1, 0.47, 0.12, 0.05, 5) + _column(0.53, 0.9, 0.12, 0.05, 5)
    for deg in (-35, -15, 0, 15, 35):
        rotated = [_rotated_box(box, deg) for box in geoms]
        assert sort_reading_order(rotated) == list(range(11))
        assert sort_reading_order(rotated, page_shape=(1000, 800)) == list(range(11))
    # Up to 44 degrees on non-square pages
    layout = [(80, 40, 670, 110), (80, 150, 360, 900), (390, 150, 670, 900)]  # title + 2 columns
    for height, width in [(1000, 750), (700, 2000)]:
        sx, sy = width / 750, height / 1000
        for angle in (-44, 30, 44):
            theta = np.deg2rad(angle)
            rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            center = np.array([width / 2, height / 2])
            polys = []
            for x0, y0, x1, y1 in layout:
                pts = np.array([[x0 * sx, y0 * sy], [x1 * sx, y0 * sy], [x1 * sx, y1 * sy], [x0 * sx, y1 * sy]])
                polys.append(((pts - center) @ rot.T + center) / np.array([width, height]))
            assert sort_reading_order(polys, page_shape=(height, width)) == [0, 1, 2], (height, width, angle)
    # Invalid inputs
    for kwargs in ({"labels": PAGE_LABELS[:-1]}, {"direction": "ttb"}, {"region_groups": [0, 1]}):
        with pytest.raises(ValueError):
            sort_reading_order(LABELED_PAGE, **kwargs)


@pytest.mark.parametrize(
    "geoms, kwargs, expected",
    [
        pytest.param([], {}, [], id="empty"),
        # 3 lines of a paragraph, a vertical gap, then 2 more lines
        pytest.param(_column(0.1, 0.9, 0.1, 0.05, 3) + _column(0.1, 0.9, 0.4, 0.05, 2), {}, [[0, 1, 2], [3, 4]]),
        pytest.param(
            _column(0.1, 0.9, 0.1, 0.05, 3) + _column(0.1, 0.9, 0.4, 0.05, 2),
            {"labels": ["Title", "Text", "Text", "Text", "Text"]},
            [[0], [1, 2], [3, 4]],
            id="label-change",
        ),
        pytest.param(
            _column(0.1, 0.9, 0.1, 0.05, 3) + _column(0.1, 0.9, 0.4, 0.05, 2),
            {"labels": ["Table", "Table", "Text", "Text", "Text"]},
            [[0], [1], [2], [3, 4]],
            id="floats-never-merge",
        ),
        pytest.param(TWO_COLUMNS, {}, [[0, 1, 2, 3], [4, 5, 6, 7]], id="columns"),
        pytest.param(
            # The fragments of a line split at a wide word gap belong to the paragraph of that line
            [_box(0.10, 0.10, 0.40, 0.13), _box(0.43, 0.10, 0.90, 0.13), _box(0.10, 0.14, 0.90, 0.17)],
            {},
            [[0, 1, 2]],
            id="split-line",
        ),
        pytest.param([_box(0.10, 0.10, 0.40, 0.13), _box(0.48, 0.10, 0.90, 0.13)], {}, [[0], [1]], id="gutter"),
        # The gap is measured in the unit of the element heights, relative coordinates of a wide page being stretched
        pytest.param([_box(0.10, 0.10, 0.40, 0.13), _box(0.44, 0.10, 0.90, 0.13)], {}, [[0, 1]], id="word-gap"),
        pytest.param(
            [_box(0.10, 0.10, 0.40, 0.13), _box(0.44, 0.10, 0.90, 0.13)],
            {"page_shape": (500, 1000)},
            [[0], [1]],
            id="word-gap-on-a-wide-page",
        ),
        pytest.param(_column(0.1, 0.9, 0.1, 0.04, 2), {"region_groups": [0, 0]}, [[0, 1]], id="same-region"),
        pytest.param(_column(0.1, 0.9, 0.1, 0.04, 2), {"region_groups": [0, 1]}, [[0], [1]], id="distinct-regions"),
        pytest.param(
            # A line split into 3 fragments (words missed by the detection), then the short last line of its paragraph,
            # then 2 more paragraphs (pixels)
            [
                _box(*box)
                for box in [
                    [46, 30, 731, 70],
                    [769, 36, 819, 70],
                    [857, 31, 1233, 68],
                    [41, 89, 125, 121],
                    [42, 245, 1199, 288],
                    [39, 288, 1208, 338],
                    [41, 459, 774, 502],
                    [825, 457, 1236, 500],
                    [41, 518, 1208, 558],
                    [39, 576, 1019, 622],
                ]
            ],
            {},
            [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9]],
            id="fragmented-paragraphs",
        ),
        pytest.param(
            [_box(0.05, 0.10, 0.30, 0.13), _box(0.70, 0.10, 0.95, 0.13), _box(0.05, 0.14, 0.95, 0.17)],
            {},
            [[0], [1, 2]],
            id="far-fragment",
        ),
        pytest.param(
            # Within a layout region of body text, the fragments of a row belong together whatever the gap
            [_box(0.05, 0.10, 0.30, 0.13), _box(0.70, 0.10, 0.95, 0.13), _box(0.05, 0.14, 0.95, 0.17)],
            {"labels": ["Text"] * 3, "region_groups": [0, 0, 0]},
            [[0, 1, 2]],
            id="far-fragment-in-a-region",
        ),
        pytest.param(
            # A degenerate page shape is ignored
            [_box(0.05, 0.10, 0.30, 0.13), _box(0.70, 0.10, 0.95, 0.13), _box(0.05, 0.14, 0.95, 0.17)],
            {"page_shape": (0, 0)},
            [[0], [1, 2]],
            id="degenerate-page-shape",
        ),
        pytest.param(
            SPLIT_CAPTION,
            {"labels": SPLIT_CAPTION_LABELS, "region_groups": [-1, -1, 7, 7, 7]},
            lambda segments: _flat(segments)[_flat(segments).index(0) + 1 : _flat(segments).index(0) + 4] == [2, 3, 4],
            id="grouped-caption",
        ),
        pytest.param(
            BESIDE,
            {"labels": BESIDE_LABELS, "region_groups": BESIDE_GROUPS},
            lambda segments: [seg for seg in segments if LAST_LINE in seg][0][-2:] == [LAST_LINE - 1, LAST_LINE],
            id="paragraph-beside-a-figure",
        ),
    ],
)
def test_resolve_reading_segments(geoms, kwargs, expected):
    segments = resolve_reading_segments(geoms, **kwargs)
    assert expected(segments) if callable(expected) else segments == expected
    assert sorted(_flat(segments)) == list(range(len(geoms)))


@pytest.mark.parametrize(
    "geoms, regions, labels, expected",
    [
        pytest.param(
            [_box(0.1, 0.1, 0.4, 0.2), _box(0.6, 0.1, 0.9, 0.2), _box(0.1, 0.5, 0.4, 0.6)],
            [_box(0.05, 0.05, 0.45, 0.25), _box(0.55, 0.05, 0.95, 0.25)],
            ["Title", "Text"],
            ["Title", "Text", None],
            id="basic",
        ),
        pytest.param(
            [_box(0.1, 0.1, 0.4, 0.2)],
            np.asarray([[(0.05, 0.05), (0.45, 0.05), (0.45, 0.25), (0.05, 0.25)]]),
            ["Table"],
            ["Table"],
            id="polygon-regions",
        ),
        pytest.param([], [_box(0.05, 0.05, 0.45, 0.25)], ["Title"], [], id="empty"),
        pytest.param(
            [_box(0.1, 0.1, 0.4, 0.2)],
            [_box(0.0, 0.0, 1.0, 1.0), _box(0.08, 0.08, 0.5, 0.3)],
            ["Picture", "Caption"],
            ["Caption"],
        ),
        pytest.param(
            # The smallest region wins, except for a float nested in page furniture
            [
                _box(0.30, 0.52, 0.60, 0.55),  # caption inside the picture
                _box(0.20, 0.30, 0.40, 0.33),  # picture text
                _box(0.03, 0.015, 0.12, 0.035),  # logo text, in the page header
                _box(0.30, 0.015, 0.60, 0.035),  # header text
                _box(0.10, 0.80, 0.90, 0.83),  # body text
            ],
            [
                _box(0.10, 0.20, 0.90, 0.60),
                _box(0.25, 0.51, 0.65, 0.56),
                _box(0.0, 0.0, 1.0, 0.05),
                _box(0.02, 0.01, 0.15, 0.04),
            ],
            ["Picture", "Caption", "Page-header", "Picture"],
            ["Caption", "Picture", "Page-header", "Page-header", None],
            id="nested",
        ),
        pytest.param(
            [_box(0.30, 0.52, 0.60, 0.55), _box(0.20, 0.30, 0.40, 0.33)]
            + [_box(0.03, 0.015, 0.12, 0.035), _box(0.30, 0.015, 0.60, 0.035), _box(0.10, 0.80, 0.90, 0.83)],
            [
                _box(0.02, 0.01, 0.15, 0.04),
                _box(0.0, 0.0, 1.0, 0.05),
                _box(0.25, 0.51, 0.65, 0.56),
                _box(0.10, 0.20, 0.90, 0.60),
            ],
            ["Picture", "Page-header", "Caption", "Picture"],
            ["Caption", "Picture", "Page-header", "Page-header", None],
            id="nested-reversed",
        ),
        pytest.param(
            # Only floats yield to the furniture
            [_box(0.3, 0.01, 0.7, 0.04)],
            [_box(0.0, 0.0, 1.0, 0.05), _box(0.25, 0.005, 0.75, 0.045)],
            ["Page-header", "Title"],
            ["Title"],
            id="furniture",
        ),
        pytest.param(
            # A region covering less than `min_coverage` never wins
            [_box(0.0, 0.0, 0.4, 0.1)],
            [_box(0.3, 0.0, 0.5, 0.1)],
            ["Text"],
            [None],
            id="min-coverage",
        ),
        pytest.param(
            # A smaller region covering the element partially does not win over the one containing it
            [_box(0.1, 0.10, 0.9, 0.12)],
            [_box(0.05, 0.05, 0.95, 0.5), _box(0.1, 0.09, 0.52, 0.13)],
            ["Text", "Section-header"],
            ["Text"],
            id="partial",
        ),
        pytest.param(
            # A line slightly overflowing a caption nested in a picture still gets the caption label
            [_box(0.30, 0.52, 0.60, 0.55)],
            [_box(0.10, 0.20, 0.90, 0.60), _box(0.31, 0.51, 0.65, 0.56)],
            ["Picture", "Caption"],
            ["Caption"],
            id="overflowing-caption",
        ),
    ],
)
def test_assign_layout_labels(geoms, regions, labels, expected):
    assert assign_layout_labels(geoms, regions, labels) == expected


def test_layout_region_helpers():
    with pytest.raises(ValueError):
        assign_layout_labels([_box(0.1, 0.1, 0.4, 0.2)], [_box(0.05, 0.05, 0.45, 0.25)] * 2, ["Title"])
    # Overlapping regions of the same label are grouped
    regions = np.asarray([
        [0.0, 0.0, 0.5, 0.5],  # 0
        [0.3, 0.3, 0.6, 0.6],  # 1: overlaps 0 by 44% of its area
        [0.7, 0.0, 0.9, 0.2],  # 2: disjoint
        [0.0, 0.0, 0.5, 0.5],  # 3: same box as 0, other label
    ])
    groups = _region_groups(regions, ["Text", "Text", "Text", "Picture"])
    assert groups[0] == groups[1] and len({groups[0], groups[2], groups[3]}) == 3
    # A small overlap does not group
    assert len(set(_region_groups(regions[:2], ["Text"] * 2, min_overlap=0.5))) == 2


@pytest.mark.parametrize(
    "order, labels, members, groups, boxes, expected",
    [
        # A float interrupting region 0, then two floats interrupting region 3
        (
            list(range(9)),
            ["text", "text", "picture", "text", "text", "table", "picture", "text", "text"],
            {},
            [0, 0, 1, 0, 3, 4, 5, 3, 6],
            None,
            [0, 1, 3, 2, 4, 7, 5, 6, 8],
        ),
        # The elements inside a float move with it
        (
            [0, 1, 2, 3, 4],
            ["text", "picture", "picture", "text", "text"],
            {1: [2]},
            [0, 1, 1, 0, 2],
            None,
            [0, 3, 1, 2, 4],
        ),
        # Floats interleaved with the lines of a region all wait for its end
        (list(range(7)), ["text", "table"] * 3 + ["text"], {}, [0, 1, 0, 2, 0, 3, 0], None, [0, 2, 4, 6, 1, 3, 5]),
        # No region, a float between two regions or at the edges: nothing moves
        ([0, 1, 2], ["text", "picture", "text"], {}, [-1, 1, -1], None, [0, 1, 2]),
        ([0, 1, 2], ["text", "picture", "text"], {}, [0, 1, 2], None, [0, 1, 2]),
        ([0, 1, 2], ["picture", "text", "text"], {}, [1, 0, 0], None, [0, 1, 2]),
        ([0, 1, 2], ["text", "text", "picture"], {}, [0, 0, 1], None, [0, 1, 2]),
        # Only a float standing beside its region moves: not a figure stacked in a loose region...
        (
            [0, 1, 2],
            ["text", "picture", "text"],
            {},
            [0, 1, 0],
            np.array([[0.1, 0.1, 0.9, 0.2], [0.2, 0.3, 0.8, 0.6], [0.1, 0.7, 0.9, 0.8]]),
            [0, 1, 2],
        ),
        # ... but a figure beside the paragraph...
        (
            [0, 1, 2],
            ["text", "picture", "text"],
            {},
            [0, 1, 0],
            np.array([[0.05, 0.3, 0.45, 0.32], [0.55, 0.3, 0.95, 0.6], [0.05, 0.62, 0.45, 0.64]]),
            [0, 2, 1],
        ),
        # ... and not text crossing the figure
        (
            [0, 1, 2],
            ["text", "picture", "text"],
            {},
            [0, 1, 0],
            np.array([[0.1, 0.3, 0.6, 0.32], [0.5, 0.3, 0.95, 0.6], [0.1, 0.7, 0.9, 0.8]]),
            [0, 1, 2],
        ),
        # Without boxes, every interrupting float moves
        ([0, 1, 2], ["text", "picture", "text"], {}, [0, 1, 0], None, [0, 2, 1]),
    ],
)
def test_defer_floats(order, labels, members, groups, boxes, expected):
    assert _defer_floats(order, labels, members, groups, boxes) == expected


def test_deskew_reading_geometries():
    geoms = [_box(0.1, 0.12, 0.47, 0.15), _box(0.53, 0.12, 0.9, 0.15)]
    rotated = [_rotated_box(box, 25) for box in geoms]
    # straight 2-point boxes are returned unchanged
    out, regions = deskew_reading_geometries(geoms, [_box(0.0, 0.0, 1.0, 0.5)])
    assert out == list(geoms) and len(regions) == 1
    # rotated polygons are de-skewed (with the page angle estimated from themselves or from `angle_geoms`): the two
    # boxes end up on the same visual row
    for angle_geoms in (None, np.stack(rotated)):
        out, _ = deskew_reading_geometries(rotated, page_shape=(1000, 800), angle_geoms=angle_geoms)
        y_centers = [np.asarray(poly)[:, 1].mean() for poly in out]
        assert abs(y_centers[0] - y_centers[1]) < 0.005
    # a straight region is expanded to its corners and rotated with the elements
    out, regions = deskew_reading_geometries(rotated, [_box(0.0, 0.1, 1.0, 0.2)], page_shape=(1000, 800))
    assert np.asarray(regions[0]).shape == (4, 2)
    # the operation is idempotent
    again, _ = deskew_reading_geometries(out, page_shape=(1000, 800))
    assert all(np.allclose(a, b) for a, b in zip(out, again))


def test_reading_order_predictor():
    predictor = ReadingOrderPredictor()
    assert predictor.direction == "auto"
    assert "auto" in repr(predictor)
    geoms = [_box(0.55, 0.2, 0.9, 0.8), _box(0.1, 0.05, 0.9, 0.15), _box(0.1, 0.2, 0.45, 0.8)]
    # Automatic direction detection from the text content
    assert predictor(geoms, texts=["right column", "the title", "left column"]) == [1, 2, 0]
    assert predictor(geoms, texts=["العمود الأيسر", "العنوان", "العمود الأيمن"]) == [1, 0, 2]
    # Language hint fallback when no text is available
    assert predictor(geoms, language="ar") == [1, 0, 2]
    assert predictor.resolve_direction(["hello"]) == "ltr"
    assert ReadingOrderPredictor(direction="rtl").resolve_direction(["hello"]) == "rtl"
    # Rotated page
    left = [_rotated_box(_box(0.1, 0.1 + 0.2 * idx, 0.3, 0.2 + 0.2 * idx), 25) for idx in range(3)]
    right = [_rotated_box(_box(0.6, 0.1 + 0.2 * idx, 0.8, 0.2 + 0.2 * idx), 25) for idx in range(3)]
    assert predictor(left + right, page_shape=(1000, 800)) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        ReadingOrderPredictor(direction="bottom-up")
