import numpy as np
from test_io_elements import _mock_kie_pages, _mock_pages

from doctr.utils import reconstitution


def test_synthesize_page():
    pages = _mock_pages()
    # Test without probability rendering
    render_no_proba = reconstitution.synthesize_page(pages[0].export(), draw_proba=False)
    assert isinstance(render_no_proba, np.ndarray)
    assert render_no_proba.shape == (*pages[0].dimensions, 3)

    # Test with probability rendering
    render_with_proba = reconstitution.synthesize_page(pages[0].export(), draw_proba=True)
    assert isinstance(render_with_proba, np.ndarray)
    assert render_with_proba.shape == (*pages[0].dimensions, 3)


def test_synthesize_kie_page():
    pages = _mock_kie_pages()
    # Test without probability rendering
    render_no_proba = reconstitution.synthesize_kie_page(pages[0].export(), draw_proba=False)
    assert isinstance(render_no_proba, np.ndarray)
    assert render_no_proba.shape == (*pages[0].dimensions, 3)

    # Test with probability rendering
    render_with_proba = reconstitution.synthesize_kie_page(pages[0].export(), draw_proba=True)
    assert isinstance(render_with_proba, np.ndarray)
    assert render_with_proba.shape == (*pages[0].dimensions, 3)


def test_synthesize_with_edge_cases():
    page = {
        "dimensions": (1000, 1000),
        "blocks": [
            {
                "lines": [
                    {
                        "words": [
                            {"value": "Test", "geometry": [(0, 0), (1, 0), (1, 1), (0, 1)], "confidence": 1.0},
                            {
                                "value": "Overflow",
                                "geometry": [(0.9, 0.9), (1.1, 0.9), (1.1, 1.1), (0.9, 1.1)],
                                "confidence": 0.5,
                            },
                        ]
                    }
                ]
            }
        ],
    }
    render = reconstitution.synthesize_page(page, draw_proba=True)
    assert isinstance(render, np.ndarray)
    assert render.shape == (1000, 1000, 3)
