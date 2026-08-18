import math

import numpy as np
import pytest
from test_io_elements import _mock_kie_pages, _mock_pages

from doctr.utils import reconstitution


def _assert_valid_render(render: np.ndarray, dimensions: tuple[int, int]) -> None:
    assert isinstance(render, np.ndarray)
    assert render.dtype == np.uint8
    assert render.shape == (*dimensions, 3)
    # Something must actually have been drawn on the page
    assert (render < 255).any()


def _table_page(h_px: int = 1000, w_px: int = 1600) -> dict:
    words, x = [], 200.0
    for token in ("The", "table", "below", "lists", "the", "items"):
        box_w = 12 * len(token)
        words.append({
            "value": token,
            "confidence": 1.0,
            "geometry": ((x / w_px, 300 / h_px), ((x + box_w) / w_px, 322 / h_px)),
        })
        x += box_w + 9
    cells = []
    for row, values in enumerate((("Product", "Price"), ("Widget", "9.99"))):
        for col, value in enumerate(values):
            x0, y0 = 200 + col * 400, 500 + row * 60
            cells.append({
                "value": value,
                "confidence": 1.0,
                "geometry": ((x0 / w_px, y0 / h_px), ((x0 + 380) / w_px, (y0 + 50) / h_px)),
                "row_start": row,
                "row_end": row,
                "col_start": col,
                "col_end": col,
            })
    return {
        "dimensions": (h_px, w_px),
        "blocks": [{"geometry": ((0, 0), (1, 1)), "lines": [{"geometry": ((0, 0), (1, 1)), "words": words}]}],
        "tables": [
            {
                "geometry": ((200 / w_px, 500 / h_px), (980 / w_px, 610 / h_px)),
                "num_rows": 2,
                "num_cols": 2,
                "confidence": 0.9,
                "cells": cells,
            }
        ],
    }


def test_synthesize_page():
    pages = _mock_pages()
    # Test without probability rendering
    render_no_proba = reconstitution.synthesize_page(pages[0].export(), draw_proba=False)
    _assert_valid_render(render_no_proba, pages[0].dimensions)
    # Text is drawn in black on white: the render must stay grayscale
    assert (render_no_proba[..., 0] == render_no_proba[..., 2]).all()

    # Test with probability rendering
    render_with_proba = reconstitution.synthesize_page(pages[0].export(), draw_proba=True)
    _assert_valid_render(render_with_proba, pages[0].dimensions)
    # Confidence boxes are colored (red-to-blue gradient), so R and B must differ somewhere
    assert (render_with_proba[..., 0] != render_with_proba[..., 2]).any()

    # Test with only one line
    pages_one_line = pages[0].export()
    pages_one_line["blocks"][0]["lines"] = [pages_one_line["blocks"][0]["lines"][0]]
    render_one_line = reconstitution.synthesize_page(pages_one_line, draw_proba=True)
    _assert_valid_render(render_one_line, pages[0].dimensions)

    # Test with polygons
    pages_poly = pages[0].export()
    pages_poly["blocks"][0]["lines"][0]["geometry"] = [(0, 0), (0, 1), (1, 1), (1, 0)]
    render_poly = reconstitution.synthesize_page(pages_poly, draw_proba=True)
    _assert_valid_render(render_poly, pages[0].dimensions)


def test_synthesize_page_colors():
    page = _mock_pages()[0].export()

    # Custom text color
    render = reconstitution.synthesize_page(page, text_color=(255, 0, 0))
    assert ((render[..., 0] > 200) & (render[..., 1] < 100) & (render[..., 2] < 100)).any()

    # Custom background color
    render = reconstitution.synthesize_page(page, background_color=(0, 0, 0), text_color=(255, 255, 255))
    # Corners are part of the background
    assert (render[0, 0] == 0).all()
    assert (render > 128).any()


def test_synthesize_page_font_size_bounds():
    page = _mock_pages()[0].export()
    render = reconstitution.synthesize_page(page, min_font_size=10, max_font_size=12)
    _assert_valid_render(render, (300, 200))


def test_synthesize_page_unicode():
    # Non-Latin text must render without raising (wide-coverage default font)
    page = _mock_pages()[0].export()
    page["blocks"][0]["lines"][0]["words"][0]["value"] = "Привет"
    page["blocks"][0]["lines"][0]["words"][1]["value"] = "Ελληνικά"
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (300, 200))


def test_synthesize_kie_page():
    pages = _mock_kie_pages()
    # Test without probability rendering
    render_no_proba = reconstitution.synthesize_kie_page(pages[0].export(), draw_proba=False)
    _assert_valid_render(render_no_proba, pages[0].dimensions)

    # Test with probability rendering
    render_with_proba = reconstitution.synthesize_kie_page(pages[0].export(), draw_proba=True)
    _assert_valid_render(render_with_proba, pages[0].dimensions)

    # Font size bounds are now part of the public signature (previously documented but missing)
    render_sized = reconstitution.synthesize_kie_page(pages[0].export(), min_font_size=10, max_font_size=20)
    _assert_valid_render(render_sized, pages[0].dimensions)


def test_synthesize_kie_page_rotated_prediction(caplog):
    page = _mock_kie_pages()[0].export()
    class_name = next(iter(page["predictions"]))
    # Replace the first prediction geometry with a ~17 degree rotated polygon
    page["predictions"][class_name][0]["geometry"] = [(0.2, 0.20), (0.6, 0.28), (0.58, 0.38), (0.18, 0.30)]

    reconstitution._warn_rotation_once.cache_clear()
    render = reconstitution.synthesize_kie_page(page, draw_proba=True)
    _assert_valid_render(render, (300, 200))

    # The rotation warning must be emitted once, and only once, per process
    reconstitution._warn_rotation_once.cache_clear()
    caplog.clear()
    with caplog.at_level("WARNING"):
        reconstitution.synthesize_kie_page(page)
        reconstitution.synthesize_kie_page(page)
    warnings = [record for record in caplog.records if "rotation" in record.message.lower()]
    assert len(warnings) == 1


def test_synthesize_page_words_do_not_overlap():
    page = {
        "dimensions": (300, 400),
        "blocks": [
            {
                "geometry": ((0.05, 0.4), (0.5, 0.48)),
                "lines": [
                    {
                        "geometry": ((0.05, 0.4), (0.5, 0.48)),
                        "words": [
                            {"value": "Wideword", "confidence": 0.9, "geometry": ((0.05, 0.4), (0.3, 0.48))},
                            {"value": "Next", "confidence": 0.9, "geometry": ((0.32, 0.4), (0.5, 0.48))},
                        ],
                    }
                ],
            }
        ],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (300, 400))

    # The vertical strip between the first word's box and the second word's box must be blank
    gap = render[:, int(round(400 * 0.3)) + 1 : int(round(400 * 0.32)) - 1]
    assert (gap == 255).all()


def test_synthesize_page_rotated_line():
    h_px, w_px = 400, 600
    angle = math.radians(-18)
    dx, dy = math.cos(angle), math.sin(angle)
    px, py = -math.sin(angle), math.cos(angle)
    height = 30

    def rot_word(value, start_x, start_y, width):
        x0, y0 = start_x, start_y
        x1, y1 = x0 + width * dx, y0 + width * dy
        x2, y2 = x1 + height * px, y1 + height * py
        x3, y3 = x0 + height * px, y0 + height * py
        poly = [(x / w_px, y / h_px) for x, y in ((x0, y0), (x1, y1), (x2, y2), (x3, y3))]
        return {"value": value, "confidence": 0.9, "geometry": poly}, (x1, y1)

    w1, end1 = rot_word("Rotated", 60, 220, 150)
    w2, _ = rot_word("baseline", end1[0] + 20 * dx, end1[1] + 20 * dy, 160)
    page = {
        "dimensions": (h_px, w_px),
        "blocks": [
            {
                "geometry": ((0, 0), (1, 1)),
                "lines": [
                    {
                        "geometry": [w1["geometry"][0], w2["geometry"][1], w2["geometry"][2], w1["geometry"][3]],
                        "words": [w1, w2],
                    },
                ],
            }
        ],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (h_px, w_px))

    # With an upward tilt, the second word's ink must sit clearly above the first word's start row;
    # a horizontal per-bbox render would not place ink that high at those x-positions
    right_half = render[: 220 - 2 * height, w_px // 2 :]
    assert (right_half < 128).any()


def test_synthesize_page_tilted_straight_boxes():
    h_px, w_px = 700, 1400
    tilt, height = math.radians(20), 40
    words, x = [], 100.0
    for token in ("Photographed", "a", "pages", "of", "tilted", "i", "documents", "to", "read"):
        text_w = 26 * len(token) + 10
        y = 150.0 + (x - 100) * math.tan(tilt)
        box_w = text_w * math.cos(tilt)
        box_h = text_w * math.sin(tilt) + height * math.cos(tilt)
        words.append({
            "value": token,
            "confidence": 0.9,
            "geometry": ((x / w_px, y / h_px), ((x + box_w) / w_px, (y + box_h) / h_px)),
        })
        x += box_w + 18
    xs = [c for word in words for c in (word["geometry"][0][0], word["geometry"][1][0])]
    ys = [c for word in words for c in (word["geometry"][0][1], word["geometry"][1][1])]
    geometry = ((min(xs), min(ys)), (max(xs), max(ys)))
    page = {
        "dimensions": (h_px, w_px),
        "blocks": [{"geometry": geometry, "lines": [{"geometry": geometry, "words": words}]}],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (h_px, w_px))

    # The ink must sit on one straight line: measured across its own direction it can only spread
    # as far as the text is tall, not as far as the boxes were inflated
    ink = np.argwhere(render.min(axis=2) < 128).astype(float)
    centred = ink - ink.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    assert np.abs(centred @ axes[1]).max() < 40


def test_synthesize_page_words_stay_apart_when_boxes_touch():
    page = {
        "dimensions": (300, 900),
        "blocks": [
            {
                "geometry": ((0.05, 0.35), (0.80, 0.62)),
                "lines": [
                    {
                        "geometry": ((0.05, 0.35), (0.80, 0.62)),
                        "words": [
                            {"value": "dominant", "confidence": 0.9, "geometry": ((0.05, 0.35), (0.32, 0.62))},
                            {"value": "language", "confidence": 0.9, "geometry": ((0.30, 0.35), (0.57, 0.62))},
                            {"value": "used", "confidence": 0.9, "geometry": ((0.55, 0.35), (0.80, 0.62))},
                        ],
                    }
                ],
            }
        ],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (300, 900))

    # Blank columns split the line into the gaps between letters and the two wider ones between
    # the three words - if the words run together the widest gaps are just letter gaps
    columns = np.nonzero((render.min(axis=2) < 128).any(axis=0))[0]
    blanks = sorted((b - a - 1 for a, b in zip(columns, columns[1:]) if b - a > 1), reverse=True)
    assert blanks[1] > 1.5 * float(np.median(blanks))


def test_synthesize_page_line_angle_survives_noisy_boxes():
    h_px, w_px = 600, 1600
    words, x = [], 80.0
    for idx, token in enumerate(("required", "a", "large", "cluster", "of", "boxes", "or", "access", "to", "read")):
        box_w = 22 * len(token) + 12
        lean = 5 if idx % 4 else -5  # most boxes lean one way, some the other
        corners = ((x, 300 + lean), (x + box_w, 300 - lean), (x + box_w, 334 - lean), (x, 334 + lean))
        words.append({
            "value": token,
            "confidence": 0.9,
            "geometry": [(px / w_px, py / h_px) for px, py in corners],
        })
        x += box_w + 16
    xs = [p[0] for word in words for p in word["geometry"]]
    ys = [p[1] for word in words for p in word["geometry"]]
    geometry = ((min(xs), min(ys)), (max(xs), max(ys)))
    page = {
        "dimensions": (h_px, w_px),
        "blocks": [{"geometry": geometry, "lines": [{"geometry": geometry, "words": words}]}],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (h_px, w_px))

    # The words all sit on a flat baseline, so the render must be flat too: a line drawn at the
    # median of the leaning box angles climbs or falls by tens of pixels across the page
    ink = np.argwhere(render.min(axis=2) < 128)
    left, right = ink[ink[:, 1] < w_px // 3], ink[ink[:, 1] > 2 * w_px // 3]
    assert abs(right[:, 0].mean() - left[:, 0].mean()) < 12


def test_synthesize_page_line_holding_two_rows():
    h_px, w_px = 400, 800
    words = []
    for tokens, top in ((("first", "row", "of", "words"), 120), (("second", "row", "here"), 180)):
        x = 60.0
        for token in tokens:
            box_w = 16 * len(token)
            words.append({
                "value": token,
                "confidence": 0.9,
                "geometry": ((x / w_px, top / h_px), ((x + box_w) / w_px, (top + 30) / h_px)),
            })
            x += box_w + 14
    xs = [c for word in words for c in (word["geometry"][0][0], word["geometry"][1][0])]
    ys = [c for word in words for c in (word["geometry"][0][1], word["geometry"][1][1])]
    geometry = ((min(xs), min(ys)), (max(xs), max(ys)))
    page = {
        "dimensions": (h_px, w_px),
        "blocks": [{"geometry": geometry, "lines": [{"geometry": geometry, "words": words}]}],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (h_px, w_px))

    # Both rows must carry ink and the band between them stay clear; flattening the line onto one
    # baseline empties one row and fills the gap
    assert all((render[top : top + 30].min(axis=2) < 128).sum() > 50 for top in (120, 180))
    assert (render[154:176] == 255).all()


def test_synthesize_page_table_cells():
    page = _table_page()
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (1000, 1600))

    def ink_rows(y0, y1, x0, x1):
        rows = np.nonzero((render[y0:y1, x0:x1].min(axis=2) < 128).any(axis=1))[0]
        return rows.max() - rows.min() + 1 if len(rows) else 0

    # every cell holds its text
    assert all(ink_rows(500, 610, x, x + 380) > 0 for x in (200, 600))
    # a table sized off its rows instead of off the page would be far taller than the body text
    assert ink_rows(500, 548, 200, 980) <= 1.5 * ink_rows(295, 330, 200, 980)


def test_synthesize_page_placeholders():
    # The grid of a table and the outline of an empty layout region are both placeholders: neither
    # is text the page holds, so neither may be drawn unless the caller asks for them
    table = _table_page()
    plain_table = reconstitution.synthesize_page(table)
    marked_table = reconstitution.synthesize_page(table, draw_placeholders=True)
    _assert_valid_render(marked_table, (1000, 1600))

    # The top edge of the table carries no text of its own, so it is only ever the grid
    assert (plain_table[500:502, 200:981] == 255).all()
    assert (marked_table[500:502, 200:981] < 255).any()
    # and the grid is drawn in a lighter tone than the text, so it stays under it
    grid = marked_table[500:502, 200:981].min(axis=2)
    assert ((grid > 150) & (grid < 220)).any()
    # the text itself is untouched: inside a cell, away from its borders, both renders agree
    assert np.array_equal(plain_table[520:540, 210:570], marked_table[520:540, 210:570])

    page = {
        "dimensions": (300, 400),
        "blocks": [
            {
                "geometry": ((0.1, 0.1), (0.4, 0.2)),
                "lines": [
                    {
                        "geometry": ((0.1, 0.1), (0.4, 0.2)),
                        "words": [{"value": "hello", "confidence": 0.9, "geometry": ((0.1, 0.1), (0.4, 0.2))}],
                    }
                ],
            }
        ],
    }
    plain = reconstitution.synthesize_page(page)
    _assert_valid_render(plain, (300, 400))

    # a region wrapped around text that is already on the page must not add anything
    covered = reconstitution.synthesize_page({
        **page,
        "layout": [{"geometry": ((0.05, 0.05), (0.45, 0.25)), "type": "Text", "confidence": 0.9}],
    })
    assert np.array_equal(plain, covered)

    # an empty one is only marked when the caller asks for it
    empty = {**page, "layout": [{"geometry": ((0.5, 0.5), (0.95, 0.9)), "type": "Picture", "confidence": 0.8}]}
    assert np.array_equal(plain, reconstitution.synthesize_page(empty))
    assert (reconstitution.synthesize_page(empty, draw_placeholders=True)[150:270, 200:380] < 255).any()

    # a region wrapped around a value-only entry counts as filled, too
    value_only = {
        "dimensions": (300, 400),
        "blocks": [{"geometry": ((0.1, 0.1), (0.4, 0.2)), "lines": [{"geometry": ((0.1, 0.1), (0.4, 0.2))}]}],
    }
    value_only["blocks"][0]["lines"][0]["value"] = "hello"
    marked = reconstitution.synthesize_page(
        {**value_only, "layout": [{"geometry": ((0.05, 0.05), (0.45, 0.25)), "type": "Text"}]},
        draw_placeholders=True,
    )
    assert np.array_equal(reconstitution.synthesize_page(value_only), marked)


def test_synthesize_page_section_header_keeps_its_size():
    h_px, w_px = 1200, 1600

    def line(text, y, size):
        words, x = [], 200.0
        for token in text.split():
            box_w = int(size * 0.58 * len(token))
            words.append({
                "value": token,
                "confidence": 1.0,
                "geometry": ((x / w_px, y / h_px), ((x + box_w) / w_px, (y + size) / h_px)),
            })
            x += box_w + int(size * 0.4)
        xs = [c for word in words for c in (word["geometry"][0][0], word["geometry"][1][0])]
        ys = [c for word in words for c in (word["geometry"][0][1], word["geometry"][1][1])]
        return {"geometry": ((min(xs), min(ys)), (max(xs), max(ys))), "words": words}

    page = {
        "dimensions": (h_px, w_px),
        "blocks": [
            {
                "geometry": ((0, 0), (1, 1)),
                "lines": [
                    line("Results", 300, 24),  # a header only a little larger than the body
                    line("Results", 380, 22),
                    line("and the mean of each run is here", 420, 22),
                    line("with the deviation in brackets", 460, 22),
                ],
            }
        ],
        "layout": [
            {"geometry": ((0.11, 0.24), (0.40, 0.28)), "type": "Section-header", "confidence": 0.95},
            {"geometry": ((0.11, 0.30), (0.80, 0.42)), "type": "Text", "confidence": 0.96},
        ],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (h_px, w_px))

    def ink_width(y0, y1):
        columns = np.nonzero((render[y0:y1, 190:400].min(axis=2) < 128).any(axis=0))[0]
        return columns.max() - columns.min() + 1 if len(columns) else 0

    # The same word sits in both, so its width follows the font size: harmonizing across the whole
    # page instead of inside each region would render the header at the size of the body text
    assert ink_width(290, 345) > 1.05 * ink_width(370, 412)


def test_synthesize_page_headings_are_bold():
    h_px, w_px = 600, 1200

    def line(text, y, size=26):
        words, x = [], 200.0
        for token in text.split():
            box_w = int(size * 0.58 * len(token))
            words.append({
                "value": token,
                "confidence": 1.0,
                "geometry": ((x / w_px, y / h_px), ((x + box_w) / w_px, (y + size) / h_px)),
            })
            x += box_w + int(size * 0.4)
        xs = [c for word in words for c in (word["geometry"][0][0], word["geometry"][1][0])]
        ys = [c for word in words for c in (word["geometry"][0][1], word["geometry"][1][1])]
        return {"geometry": ((min(xs), min(ys)), (max(xs), max(ys))), "words": words}

    page = {
        "dimensions": (h_px, w_px),
        "blocks": [{"geometry": ((0, 0), (1, 1)), "lines": [line("Heading", 200), line("Heading", 350)]}],
        "layout": [
            {"geometry": ((0.12, 0.31), (0.50, 0.40)), "type": "Title", "confidence": 0.95},
            {"geometry": ((0.12, 0.56), (0.50, 0.66)), "type": "Text", "confidence": 0.95},
        ],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (h_px, w_px))

    def ink(y0, y1):
        marked = render[y0:y1, 190:500].min(axis=2) < 128
        columns = np.nonzero(marked.any(axis=0))[0]
        return marked.sum(), (columns.max() - columns.min() + 1 if len(columns) else 0)

    title, body = ink(190, 250), ink(340, 400)
    # the same word in the same size of box: the title carries more ink without growing wider,
    # which is weight rather than size
    assert title[0] > 1.3 * body[0]
    assert title[1] < 1.15 * body[1]


def test_points_rejects_unusable_geometries():
    # not a sequence of pairs at all
    assert reconstitution._points(None) is None
    assert reconstitution._points("nope") is None
    assert reconstitution._points([(0.0, 0.0), (1.0, "x")]) is None
    # neither a box nor a polygon
    assert reconstitution._points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]) is None
    # coordinates that cannot be drawn
    assert reconstitution._points([(0.0, 0.0), (float("nan"), 1.0)]) is None
    assert reconstitution._points([(0, 0), (1, 1)]) == [(0.0, 0.0), (1.0, 1.0)]


def test_draw_word_falls_back_to_ascii():
    drawn = []

    class _AsciiOnlyDraw:
        def text(self, xy, text, **kwargs):
            if any(ord(character) > 127 for character in text):
                raise UnicodeEncodeError("ascii", text, 0, 1, "not encodable")
            drawn.append(text)

    font = reconstitution._cached_font(None, 12)
    reconstitution._draw_word(_AsciiOnlyDraw(), (0, 0), "Ünïcôde", font, (0, 0, 0))
    assert drawn == ["Unicode"]


def test_tilt_needs_words_that_agree():
    word = reconstitution._Word
    # too few words to read a trend from
    assert reconstitution._tilt([word("a", index, 0, 10, 10, 0.0) for index in range(3)]) is None
    # words sitting on top of each other: every pair is too close to measure a direction
    assert reconstitution._tilt([word("a", 100, 100 + 3 * index, 20, 20, 0.0) for index in range(4)]) is None
    # words scattered around: the pairs do not agree on any one direction
    scattered = [word("a", 100 * index, y, 20, 20, 0.0) for index, y in enumerate((0, 400, 90, 900))]
    assert reconstitution._tilt(scattered) is None


def test_unrotate_box_keeps_what_it_cannot_invert():
    word = reconstitution._Word("a", 100, 100, 40, 20, 0.0)
    # past ~39 degrees the inversion stops being trustworthy
    assert reconstitution._unrotate_box(word, 45.0) == word
    # a box whose recovered extent would be empty is left alone as well
    thin = reconstitution._Word("a", 100, 100, 10, 20, 0.0)
    assert reconstitution._unrotate_box(thin, 35.0) == thin
    # a box that can be inverted comes back with the angle it was measured at
    assert reconstitution._unrotate_box(word, -10.0).angle == -10.0


def test_space_width_without_a_measurable_space():
    class _NoAdvance:
        def getlength(self, text):
            raise ValueError("no advance for this font")

        def getbbox(self, text):
            return (0, 0, 8, 10)

    assert reconstitution._space_width(_NoAdvance()) == 4.0
    assert reconstitution._space_width(reconstitution._cached_font(None, 12)) >= 1.0


def test_synthesize_page_skips_unusable_words():
    page = {
        "dimensions": (200, 400),
        "blocks": [
            {
                "geometry": ((0.1, 0.4), (0.9, 0.6)),
                "lines": [
                    {
                        "geometry": ((0.1, 0.4), (0.9, 0.6)),
                        "words": [
                            "not a word at all",
                            {"value": "   ", "confidence": 0.9, "geometry": ((0.1, 0.4), (0.3, 0.6))},
                            {"value": "gone", "confidence": 0.9, "geometry": None},
                            {"value": "kept", "confidence": 0.9, "geometry": ((0.4, 0.4), (0.9, 0.6))},
                        ],
                    }
                ],
            }
        ],
    }
    render = reconstitution.synthesize_page(page)
    _assert_valid_render(render, (200, 400))
    # only the one usable word was drawn, and it stayed inside its own box
    assert (render[:, : int(0.4 * 400)] == 255).all()


def test_synthesize_page_crowded_line_drops_a_size():
    # Boxes stacked on the same spot: the words cannot all fit at the size the boxes suggest, so
    # the line has to be set smaller instead of being squashed to nothing
    words = [{"value": "stacked", "confidence": 0.9, "geometry": ((0.1, 0.4), (0.4, 0.6))} for _ in range(6)]
    page = {
        "dimensions": (300, 400),
        "blocks": [
            {"geometry": ((0.1, 0.4), (0.4, 0.6)), "lines": [{"geometry": ((0.1, 0.4), (0.4, 0.6)), "words": words}]}
        ],
    }
    render = reconstitution.synthesize_page(page, min_font_size=8, max_font_size=50)
    _assert_valid_render(render, (300, 400))


def test_synthesize_page_skips_unusable_entries():
    page = {
        "dimensions": (200, 400),
        "blocks": [
            {
                "geometry": ((0, 0), (1, 1)),
                "lines": [
                    # no geometry to place anything with
                    {
                        "geometry": None,
                        "words": [{"value": "lost", "confidence": 0.9, "geometry": ((0.1, 0.1), (0.3, 0.3))}],
                    },
                    # a geometry, but nothing at all to write in it
                    {"geometry": ((0.1, 0.6), (0.5, 0.8)), "value": "  ", "confidence": 0.4, "words": []},
                ],
            }
        ],
    }
    # nothing is drawn, but the empty entry is still outlined when confidences are asked for
    assert (reconstitution.synthesize_page(page) == 255).all()
    assert (reconstitution.synthesize_page(page, draw_proba=True) < 255).any()


def test_synthesize_page_reports_entries_it_cannot_draw(caplog, monkeypatch):
    page = {
        "dimensions": (200, 400),
        "blocks": [
            {
                "geometry": ((0.1, 0.4), (0.9, 0.6)),
                "lines": [
                    {
                        "geometry": ((0.1, 0.4), (0.9, 0.6)),
                        "words": [{"value": "hello", "confidence": 0.9, "geometry": ((0.1, 0.4), (0.9, 0.6))}],
                    }
                ],
            }
        ],
    }

    def _raise(*args, **kwargs):
        raise ValueError("boom")

    # an entry that cannot be sized is dropped, with a warning, instead of failing the whole page
    monkeypatch.setattr(reconstitution, "_entry_font_size", _raise)
    with caplog.at_level("WARNING"):
        assert (reconstitution.synthesize_page(page) == 255).all()
    assert any("could not size entry" in record.message.lower() for record in caplog.records)

    # and so is an entry that cannot be drawn
    monkeypatch.undo()
    caplog.clear()
    monkeypatch.setattr(reconstitution, "_synthesize_line", _raise)
    with caplog.at_level("WARNING"):
        assert (reconstitution.synthesize_page(page) == 255).all()
    assert any("could not render entry" in record.message.lower() for record in caplog.records)


def test_synthesize_page_ignores_malformed_tables_and_regions():
    page = {
        "dimensions": (300, 400),
        "blocks": [],
        "tables": [
            "not a table",
            {"geometry": None, "cells": ["not a cell", {"geometry": None, "value": "x"}]},
            {
                "geometry": ((0.1, 0.1), (0.9, 0.5)),
                "cells": [{"geometry": ((0.1, 0.1), (0.9, 0.5)), "value": "x", "row_start": 0}],
            },
        ],
        "layout": ["not a region", {"geometry": None, "type": "Picture"}],
    }
    render = reconstitution.synthesize_page(page, draw_placeholders=True)
    _assert_valid_render(render, (300, 400))
    # only the one table that could be placed was drawn
    assert (render[:20] == 255).all()


def test_synthesize_page_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        reconstitution.synthesize_page({"blocks": []})
    with pytest.raises(ValueError, match="dimensions"):
        reconstitution.synthesize_page({"dimensions": ("a", "b"), "blocks": []})
    with pytest.raises(ValueError, match="positive"):
        reconstitution.synthesize_page({"dimensions": (0, 100), "blocks": []})
    with pytest.raises(ValueError, match="MAX_IMAGE_PIXELS"):
        reconstitution.synthesize_page({"dimensions": (100_000, 100_000), "blocks": []})
