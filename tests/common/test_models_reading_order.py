import numpy as np
import pytest

from doctr.models.reading_order import (
    ReadingOrderPredictor,
    assign_layout_labels,
    detect_text_direction,
    layout_label_role,
    normalize_layout_label,
    resolve_reading_segments,
    sort_reading_order,
)


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


def test_normalize_layout_label():
    assert normalize_layout_label("Page-header") == "page_header"
    assert normalize_layout_label(" Section header ") == "section_header"
    assert normalize_layout_label(None) == ""


@pytest.mark.parametrize(
    "label, role",
    [
        ("Page-header", "header"),
        ("Page-footer", "footer"),
        ("Footnote", "footnote"),
        ("Caption", "caption"),
        ("Table", "float"),
        ("Picture", "float"),
        ("Text", "body"),
        ("Title", "body"),
        (None, "body"),
    ],
)
def test_layout_label_role(label, role):
    assert layout_label_role(label) == role


def _two_columns(num_lines: int = 4):
    left = [((0.1, 0.1 + 0.2 * i), (0.45, 0.25 + 0.2 * i)) for i in range(num_lines)]
    right = [((0.55, 0.1 + 0.2 * i), (0.9, 0.25 + 0.2 * i)) for i in range(num_lines)]
    return left + right


def test_sort_reading_order_basic():
    # Degenerate inputs
    assert sort_reading_order([]) == []
    assert sort_reading_order([((0.1, 0.1), (0.2, 0.2))]) == [0]
    # Single column, top to bottom
    geoms = [((0.1, 0.5), (0.9, 0.6)), ((0.1, 0.1), (0.9, 0.2)), ((0.1, 0.3), (0.9, 0.4))]
    assert sort_reading_order(geoms) == [1, 2, 0]
    # A title spanning two columns
    geoms = [((0.55, 0.2), (0.9, 0.8)), ((0.1, 0.05), (0.9, 0.15)), ((0.1, 0.2), (0.45, 0.8))]
    assert sort_reading_order(geoms) == [1, 2, 0]


def test_sort_reading_order_columns():
    boxes = _two_columns()
    # The left column must be read entirely before the right one
    assert sort_reading_order(boxes) == list(range(8))
    # Right-to-left: the right column comes first
    assert sort_reading_order(boxes, direction="rtl") == [4, 5, 6, 7, 0, 1, 2, 3]
    # The result is independent of the input ordering
    rng = np.random.default_rng(42)
    for _ in range(5):
        perm = rng.permutation(8).tolist()
        order = sort_reading_order([boxes[idx] for idx in perm])
        assert [perm[idx] for idx in order] == list(range(8))


def test_sort_reading_order_input_formats():
    boxes = _two_columns(2)
    expected = sort_reading_order(boxes)
    # (N, 4) array
    as_array = np.asarray([(x0, y0, x1, y1) for ((x0, y0), (x1, y1)) in boxes])
    assert sort_reading_order(as_array) == expected
    # (N, 4, 2) rotated polygons
    as_polys = np.asarray([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for ((x0, y0), (x1, y1)) in boxes])
    assert sort_reading_order(as_polys) == expected
    # absolute coordinates
    assert sort_reading_order(as_array * 1000) == expected


def test_sort_reading_order_vertical():
    # 4 vertical columns, from right to left (traditional Japanese/Chinese)
    cols = [((0.8 - 0.15 * i, 0.1), (0.9 - 0.15 * i, 0.9)) for i in range(4)]
    assert sort_reading_order(cols, direction="ttb-rtl") == [0, 1, 2, 3]
    assert sort_reading_order(cols, direction="ttb-ltr") == [3, 2, 1, 0]
    # Two stacked elements within a column are read top to bottom
    cols = [((0.8, 0.5), (0.9, 0.9)), ((0.8, 0.1), (0.9, 0.45)), ((0.6, 0.1), (0.7, 0.9))]
    assert sort_reading_order(cols, direction="ttb-rtl") == [1, 0, 2]


def test_sort_reading_order_labels():
    geoms = [
        ((0.1, 0.92), (0.9, 0.97)),  # 0: page footer
        ((0.1, 0.02), (0.9, 0.06)),  # 1: page header
        ((0.1, 0.1), (0.9, 0.4)),  # 2: body text
        ((0.1, 0.45), (0.5, 0.7)),  # 3: figure
        ((0.1, 0.71), (0.5, 0.75)),  # 4: caption below the figure
        ((0.55, 0.45), (0.9, 0.88)),  # 5: body on the right of the figure
        ((0.1, 0.8), (0.5, 0.84)),  # 6: footnote
    ]
    labels = ["Page-footer", "Page-header", "Text", "Picture", "Caption", "Text", "Footnote"]
    assert sort_reading_order(geoms, labels=labels) == [1, 2, 3, 4, 5, 6, 0]
    # A caption above its figure is read before it
    labels_above = list(labels)
    geoms_above = list(geoms)
    geoms_above[4] = ((0.1, 0.41), (0.5, 0.44))
    assert sort_reading_order(geoms_above, labels=labels_above) == [1, 2, 4, 3, 5, 6, 0]
    # A caption too far from any float keeps its natural position in the body
    geoms_far = list(geoms)
    geoms_far[3] = ((0.1, 0.45), (0.5, 0.5))  # shrink the figure
    geoms_far[4] = ((0.55, 0.02), (0.9, 0.06))  # move the caption to the top right
    order = sort_reading_order(geoms_far, labels=labels)
    assert order.index(4) < order.index(3)
    with pytest.raises(ValueError):
        sort_reading_order(geoms, labels=labels[:-1])
    with pytest.raises(ValueError):
        sort_reading_order(geoms, direction="ttb")


def test_sort_reading_order_degenerate_geometries():
    # Identical & zero-area boxes should not crash nor loop
    geoms = [((0.1, 0.1), (0.1, 0.1))] * 3 + [((0.5, 0.5), (0.5, 0.5))]
    order = sort_reading_order(geoms)
    assert sorted(order) == list(range(4))


def test_resolve_reading_segments():
    # 3 lines of a paragraph, a vertical gap, then 2 more lines
    geoms = [((0.1, 0.1 + 0.05 * i), (0.9, 0.13 + 0.05 * i)) for i in range(3)]
    geoms += [((0.1, 0.4 + 0.05 * i), (0.9, 0.43 + 0.05 * i)) for i in range(2)]
    assert resolve_reading_segments(geoms) == [[0, 1, 2], [3, 4]]
    # A label change breaks the segment
    labels = ["Title", "Text", "Text", "Text", "Text"]
    assert resolve_reading_segments(geoms, labels=labels) == [[0], [1, 2], [3, 4]]
    # Floats are never merged with their neighbors
    labels = ["Table", "Table", "Text", "Text", "Text"]
    assert resolve_reading_segments(geoms, labels=labels) == [[0], [1], [2], [3, 4]]
    # Two columns end up in distinct segments
    boxes = _two_columns()
    assert resolve_reading_segments(boxes) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert resolve_reading_segments([]) == []


def test_assign_layout_labels():
    geoms = [((0.1, 0.1), (0.4, 0.2)), ((0.6, 0.1), (0.9, 0.2)), ((0.1, 0.5), (0.4, 0.6))]
    regions = [((0.05, 0.05), (0.45, 0.25)), ((0.55, 0.05), (0.95, 0.25))]
    labels = assign_layout_labels(geoms, regions, ["Title", "Text"])
    assert labels == ["Title", "Text", None]
    # Rotated region geometries are supported
    poly_regions = np.asarray([[(0.05, 0.05), (0.45, 0.05), (0.45, 0.25), (0.05, 0.25)]])
    assert assign_layout_labels(geoms[:1], poly_regions, ["Table"]) == ["Table"]
    assert assign_layout_labels([], regions, ["Title", "Text"]) == []
    with pytest.raises(ValueError):
        assign_layout_labels(geoms, regions, ["Title"])


def test_reading_order_predictor():
    predictor = ReadingOrderPredictor()
    assert predictor.direction == "auto"
    assert "auto" in repr(predictor)
    geoms = [((0.55, 0.2), (0.9, 0.8)), ((0.1, 0.05), (0.9, 0.15)), ((0.1, 0.2), (0.45, 0.8))]
    # Automatic direction detection from the text content
    assert predictor(geoms, texts=["right column", "the title", "left column"]) == [1, 2, 0]
    assert predictor(geoms, texts=["العمود الأيسر", "العنوان", "العمود الأيمن"]) == [1, 0, 2]
    # Language hint fallback when no text is available
    assert predictor(geoms, language="ar") == [1, 0, 2]
    assert predictor.resolve_direction(["hello"]) == "ltr"
    assert ReadingOrderPredictor(direction="rtl").resolve_direction(["hello"]) == "rtl"
    with pytest.raises(ValueError):
        ReadingOrderPredictor(direction="bottom-up")


# regression test for a bug where a stray fragment of a split line was read after the next line in the same column
def test_sort_reading_order_fragmented_columns():
    left = [
        ((0.10, 0.10), (0.45, 0.13)),  # 0 wide
        ((0.10, 0.14), (0.25, 0.17)),  # 1 narrow (left part of a split line)
        ((0.34, 0.14), (0.45, 0.17)),  # 2 stray fragment (right part), same visual row as 1
        ((0.10, 0.18), (0.45, 0.21)),  # 3
        ((0.10, 0.22), (0.45, 0.25)),  # 4
        ((0.10, 0.26), (0.45, 0.29)),  # 5
    ]
    right = [((0.55, 0.10 + 0.04 * i), (0.90, 0.13 + 0.04 * i)) for i in range(6)]  # 6..11
    order = sort_reading_order(left + right)
    # every left element (0..5) is read before every right element (6..11)
    assert max(order.index(i) for i in range(6)) < min(order.index(i) for i in range(6, 12))
