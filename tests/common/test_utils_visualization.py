import numpy as np
import pytest
from test_io_elements import _mock_pages

from doctr.utils import visualization


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


def test_draw_boxes():
    image = np.ones((256, 256, 3), dtype=np.float32)
    boxes = [
        [0.1, 0.1, 0.2, 0.2],
        [0.15, 0.15, 0.19, 0.2],  # to suppress
        [0.5, 0.5, 0.6, 0.55],
        [0.55, 0.5, 0.7, 0.55],  # to suppress
    ]
    visualization.draw_boxes(boxes=np.array(boxes), image=image, block=False)
