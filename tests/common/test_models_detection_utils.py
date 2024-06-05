import numpy as np
import pytest

from doctr.models.detection._utils import _remove_padding


@pytest.mark.parametrize("pages", [[np.zeros((1000, 1000))], [np.zeros((1000, 2000))], [np.zeros((2000, 1000))]])
@pytest.mark.parametrize("preserve_aspect_ratio", [True, False])
@pytest.mark.parametrize("symmetric_pad", [True, False])
@pytest.mark.parametrize("assume_straight_pages", [True, False])
def test_remove_padding(pages, preserve_aspect_ratio, symmetric_pad, assume_straight_pages):
    h, w = pages[0].shape
    # straight pages test cases
    if assume_straight_pages:
        loc_preds = [{"words": np.array([[0.7, 0.1, 0.7, 0.2]])}]
        if h == w or not preserve_aspect_ratio:
            expected = loc_preds
        else:
            if symmetric_pad:
                if h > w:
                    expected = [{"words": np.array([[0.9, 0.1, 0.9, 0.2]])}]
                else:
                    expected = [{"words": np.array([[0.7, 0.0, 0.7, 0.0]])}]
            else:
                if h > w:
                    expected = [{"words": np.array([[1.0, 0.1, 1.0, 0.2]])}]
                else:
                    expected = [{"words": np.array([[0.7, 0.2, 0.7, 0.4]])}]
    # non-straight pages test cases
    else:
        loc_preds = [{"words": np.array([[[0.9, 0.1], [0.9, 0.2], [0.8, 0.2], [0.8, 0.2]]])}]
        if h == w or not preserve_aspect_ratio:
            expected = loc_preds
        else:
            if symmetric_pad:
                if h > w:
                    expected = [{"words": np.array([[[1.0, 0.1], [1.0, 0.2], [1.0, 0.2], [1.0, 0.2]]])}]
                else:
                    expected = [{"words": np.array([[[0.9, 0.0], [0.9, 0.0], [0.8, 0.0], [0.8, 0.0]]])}]
            else:
                if h > w:
                    expected = [{"words": np.array([[[1.0, 0.1], [1.0, 0.2], [1.0, 0.2], [1.0, 0.2]]])}]
                else:
                    expected = [{"words": np.array([[[0.9, 0.2], [0.9, 0.4], [0.8, 0.4], [0.8, 0.4]]])}]

    result = _remove_padding(pages, loc_preds, preserve_aspect_ratio, symmetric_pad, assume_straight_pages)
    for res, exp in zip(result, expected):
        assert np.allclose(res["words"], exp["words"])
