import numpy as np

from doctr import utils
from test_documents import _mock_pages


def test_bbox_to_polygon():
    assert utils.bbox_to_polygon(((0, 0), (1, 1))) == ((0, 0), (1, 0), (0, 1), (1, 1))


def test_polygon_to_bbox():
    assert utils.polygon_to_bbox(((0, 0), (1, 0), (0, 1), (1, 1))) == ((0, 0), (1, 1))


def test_resolve_enclosing_bbox():
    assert utils.resolve_enclosing_bbox([((0, 0.5), (1, 0)), ((0.5, 0), (1, 0.25))]) == ((0, 0), (1, 0.5))


def test_visualize_page():
    pages = _mock_pages()
    images = [np.ones((300, 200, 3)), np.ones((500, 1000, 3))]
    for image, page in zip(images, pages):
        visualization.visualize_page(page, image)
