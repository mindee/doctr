import numpy as np
import matplotlib.pyplot as plt

from doctr.utils import visualization
from test_documents_elements import _mock_pages


def test_visualize_page():
    pages = _mock_pages()
    images = [np.ones((300, 200, 3)), np.ones((500, 1000, 3))]
    for image, page in zip(images, pages):
        visualization.visualize_page(page, image, words_only=False)
        plt.show(block=False)
