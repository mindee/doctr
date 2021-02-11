import numpy as np
import matplotlib.pyplot as plt

from doctr.utils import visualization
from test_documents_elements import _mock_pages


def test_visualize_page():
    pages = _mock_pages()
    image = np.ones((300, 200, 3))
    visualization.visualize_page(pages[0].export(), image, words_only=False)
    plt.show(block=False)
