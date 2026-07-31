import math

import numpy as np
from test_io_elements import _mock_kie_pages, _mock_pages

from doctr.utils import reconstitution


def _assert_valid_render(render: np.ndarray, dimensions: tuple[int, int]) -> None:
    assert isinstance(render, np.ndarray)
    assert render.dtype == np.uint8
    assert render.shape == (*dimensions, 3)
    # Something must actually have been drawn on the page
    assert (render < 255).any()


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
