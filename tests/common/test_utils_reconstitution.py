import numpy as np
from test_io_elements import _mock_pages

from doctr.utils import reconstitution


def test_synthesize_page():
    pages = _mock_pages()
    reconstitution.synthesize_page(pages[0].export(), draw_proba=False)
    render = reconstitution.synthesize_page(pages[0].export(), draw_proba=True)
    assert isinstance(render, np.ndarray)
    assert render.shape == (*pages[0].dimensions, 3)
