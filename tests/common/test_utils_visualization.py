import json

import numpy as np
import pytest
from test_io_elements import _mock_kie_pages, _mock_layout, _mock_pages

from doctr.utils import visualization


def test_visualize_page():
    pages = _mock_pages()
    image = np.ones((300, 200, 3))
    visualization.visualize_page(pages[0].export(), image, words_only=False)
    visualization.visualize_page(pages[0].export(), image, words_only=True, interactive=False)

    # with detected layout regions
    page_export = pages[0].export()
    page_export["layout"] = [region.export() for region in _mock_layout()]
    visualization.visualize_page(page_export, image, words_only=False, display_layout=True)
    visualization.visualize_page(page_export, image, words_only=False, display_layout=True, interactive=False)
    visualization.visualize_page(page_export, image, words_only=False, display_layout=False, interactive=False)

    with pytest.raises(ValueError):
        visualization.create_obj_patch([1, 2], (100, 100))
    with pytest.raises(ValueError):
        visualization.create_obj_patch((1, 2), (100, 100))
    with pytest.raises(ValueError):
        visualization.create_obj_patch((1, 2, 3, 4, 5), (100, 100))


def test_visualize_json_exports():
    # A page export read back from a JSON file stores the geometries as lists instead of tuples
    image = np.ones((300, 200, 3))
    page_export = _mock_pages()[0].export()
    page_export["layout"] = [region.export() for region in _mock_layout()]
    page_export["blocks"][0]["lines"][0]["words"][0]["geometry"] = ((0.1, 0.1), (0.4, 0.1), (0.4, 0.2), (0.1, 0.2))
    json_export = json.loads(json.dumps(page_export))
    for interactive in (True, False):
        visualization.visualize_page(json_export, image, words_only=False, interactive=interactive)
        fig = visualization.visualize_page(page_export, image, words_only=False, interactive=interactive)
        json_fig = visualization.visualize_page(json_export, image, words_only=False, interactive=interactive)
        assert len(json_fig.axes[0].patches) == len(fig.axes[0].patches)
        assert [text.get_text() for text in json_fig.axes[0].texts] == [text.get_text() for text in fig.axes[0].texts]

    kie_export = json.loads(json.dumps(_mock_kie_pages()[0].export()))
    fig = visualization.visualize_kie_page(kie_export, image, interactive=False, add_labels=True)
    assert [text.get_text() for text in fig.axes[0].texts] == ["hello", "world"]

    with pytest.raises(ValueError):
        visualization.create_obj_patch([[0.1, 0.1], [0.2]], (100, 100))


def test_visualize_kie_page():
    pages = _mock_kie_pages()
    image = np.ones((300, 200, 3))
    visualization.visualize_kie_page(pages[0].export(), image, words_only=False)
    visualization.visualize_kie_page(pages[0].export(), image, words_only=True, interactive=False)

    # with detected layout regions
    page_export = pages[0].export()
    page_export["layout"] = [region.export() for region in _mock_layout()]
    visualization.visualize_kie_page(page_export, image, words_only=False, display_layout=True)
    visualization.visualize_kie_page(page_export, image, words_only=False, display_layout=False, interactive=False)

    # static plots label each prediction when add_labels is set
    fig = visualization.visualize_kie_page(pages[0].export(), image, interactive=False, add_labels=True)
    assert [text.get_text() for text in fig.axes[0].texts] == ["hello", "world"]
    fig = visualization.visualize_kie_page(pages[0].export(), image, interactive=False, add_labels=False)
    assert len(fig.axes[0].texts) == 0


def test_draw_boxes():
    image = np.ones((256, 256, 3), dtype=np.float32)
    boxes = [
        [0.1, 0.1, 0.2, 0.2],
        [0.15, 0.15, 0.19, 0.2],  # to suppress
        [0.5, 0.5, 0.6, 0.55],
        [0.55, 0.5, 0.7, 0.55],  # to suppress
    ]
    visualization.draw_boxes(boxes=np.array(boxes), image=image, block=False)
