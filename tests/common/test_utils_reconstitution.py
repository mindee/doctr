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
    # Two adjacent words whose boxes are much narrower than the naive line-level font
    # would require: the render must keep the gap between the boxes blank (regression test)
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
    # A line whose words carry rotated 4-point polygons must render without error,
    # follow the rotation (ink appears along the tilted baseline, not just the top band),
    # and keep adjacent words from overlapping
    import math

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
