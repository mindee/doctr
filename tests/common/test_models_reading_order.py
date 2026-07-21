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


def _rotated_box(box, deg, width=800, height=1000):
    angle = np.deg2rad(deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([width / 2, height / 2])
    (x0, y0), (x1, y1) = box
    pts = np.array([
        [x0 * width, y0 * height],
        [x1 * width, y0 * height],
        [x1 * width, y1 * height],
        [x0 * width, y1 * height],
    ])
    return ((pts - center) @ rot.T + center) / [width, height]


def test_sort_reading_order_rotated_pages():
    title = [((0.1, 0.06), (0.9, 0.09))]
    left = [((0.1, 0.12 + 0.05 * idx), (0.47, 0.15 + 0.05 * idx)) for idx in range(5)]
    right = [((0.53, 0.12 + 0.05 * idx), (0.9, 0.15 + 0.05 * idx)) for idx in range(5)]
    geoms = title + left + right
    expected = list(range(11))
    for deg in (-35, -15, 15, 35):
        rotated = [_rotated_box(box, deg) for box in geoms]
        assert sort_reading_order(rotated) == expected
        assert sort_reading_order(rotated, page_shape=(1000, 800)) == expected
    # straight polygons are untouched (angle below the threshold, behavior identical to 2-point boxes)
    straight = np.asarray([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for ((x0, y0), (x1, y1)) in geoms])
    assert sort_reading_order(straight) == expected


def test_deskew_reading_geometries():
    geoms = [((0.1, 0.12), (0.47, 0.15)), ((0.53, 0.12), (0.9, 0.15))]
    rotated = [_rotated_box(box, 25) for box in geoms]
    # straight 2-point boxes are returned unchanged
    out, regions = deskew_reading_geometries(geoms, [((0.0, 0.0), (1.0, 0.5))])
    assert out == list(geoms) and len(regions) == 1
    # rotated polygons are de-skewed: the two boxes end up on the same visual row
    out, _ = deskew_reading_geometries(rotated, page_shape=(1000, 800))
    y_centers = [np.asarray(poly)[:, 1].mean() for poly in out]
    assert abs(y_centers[0] - y_centers[1]) < 0.005
    # a straight region is expanded to its corners and rotated with the elements
    out, regions = deskew_reading_geometries(rotated, [((0.0, 0.1), (1.0, 0.2))], page_shape=(1000, 800))
    assert np.asarray(regions[0]).shape == (4, 2)
    # the operation is idempotent
    again, _ = deskew_reading_geometries(out, page_shape=(1000, 800))
    assert all(np.allclose(a, b) for a, b in zip(out, again))
    # angle_geoms as the estimation source
    out, _ = deskew_reading_geometries(rotated, page_shape=(1000, 800), angle_geoms=np.stack(rotated))
    y_centers = [np.asarray(poly)[:, 1].mean() for poly in out]
    assert abs(y_centers[0] - y_centers[1]) < 0.005


def test_reading_order_predictor_rotated():
    left = [_rotated_box(((0.1, 0.1 + 0.2 * idx), (0.3, 0.2 + 0.2 * idx)), 25) for idx in range(3)]
    right = [_rotated_box(((0.6, 0.1 + 0.2 * idx), (0.8, 0.2 + 0.2 * idx)), 25) for idx in range(3)]
    order = ReadingOrderPredictor()(left + right, page_shape=(1000, 800))
    assert order == [0, 1, 2, 3, 4, 5]


def test_deskew_strong_rotation_non_square_page():
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


# Auto generated regression tests for known failures of the reading order algorithm


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


def test_fragmented_row_with_merged_column_components():
    geoms = [
        ((0.35, 0.05), (0.65, 0.10)),  # 0 gutter-straddling element (bridges both columns)
        ((0.10, 0.15), (0.45, 0.20)),  # 1 left col, row 1
        ((0.10, 0.22), (0.16, 0.27)),  # 2 left col, row 2, fragment A
        ((0.17, 0.22), (0.24, 0.27)),  # 3 left col, row 2, fragment B
        ((0.25, 0.22), (0.45, 0.27)),  # 4 left col, row 2, fragment C
        ((0.10, 0.29), (0.45, 0.34)),  # 5 left col, row 3
        ((0.55, 0.15), (0.90, 0.20)),  # 6 right col, row 1
        ((0.55, 0.22), (0.90, 0.27)),  # 7 right col, row 2
    ]
    assert sort_reading_order(geoms) == [0, 1, 2, 3, 4, 5, 6, 7]
