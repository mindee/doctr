import numpy as np

from doctr.utils import visualization
from test_documents_elements import _mock_pages


def test_visualize_page():
    pages = _mock_pages()
    image = np.ones((300, 200, 3))
    visualization.visualize_page(pages[0].export(), image, words_only=False)
    visualization.visualize_page(pages[0].export(), image, words_only=True, interactive=False)


def test_draw_page():
    pages = _mock_pages()
    visualization.synthetize_page(pages[0].export(), draw_proba=True)
    visualization.synthetize_page(pages[0].export(), draw_proba=False)


def test_draw_boxes():
    image = np.ones((256, 256, 3))
    boxes = [
        [0.1, 0.1, 0.2, 0.2, 0.95],
        [0.15, 0.15, 0.19, 0.2, 0.90],  # to suppress
        [0.5, 0.5, 0.6, 0.55, 0.90],
        [0.55, 0.5, 0.7, 0.55, 0.85],  # to suppress
    ]
    visualization.draw_boxes(boxes=np.array(boxes), image=pages[0], block=True)
