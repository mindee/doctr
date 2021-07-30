import pytest
import numpy as np

from doctr.utils import visualization
from test_io_elements import _mock_pages


def test_visualize_page():
    pages = _mock_pages()
    image = np.ones((300, 200, 3))
    visualization.visualize_page(pages[0].export(), image, words_only=False)
    visualization.visualize_page(pages[0].export(), image, words_only=True, interactive=False)
    # geometry checks
    with pytest.raises(ValueError):
        visualization.create_obj_patch([1, 2], (100, 100))

    with pytest.raises(ValueError):
        visualization.create_obj_patch((1, 2), (100, 100))

    with pytest.raises(ValueError):
        visualization.create_obj_patch((1, 2, 3, 4, 5), (100, 100))


def test_draw_page():
    pages = _mock_pages()
    visualization.synthetize_page(pages[0].export(), draw_proba=True)
    visualization.synthetize_page(pages[0].export(), draw_proba=False)
