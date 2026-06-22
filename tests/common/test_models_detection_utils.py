import numpy as np
import pytest

from doctr.models.detection._utils import _remove_padding


def _attach_score(loc_pred: np.ndarray, score: float = 0.8) -> np.ndarray:
    """Attach a box score to a prediction.

    Straight pages: (N, 4) -> (N, 5) by appending a score column.
    Non-straight pages: (N, 4, 2) -> (N, 5, 2) by appending a trailing (0, score) row,
    matching the format produced by the detection post-processors.
    """
    if loc_pred.ndim == 2:  # straight pages
        scores = np.full((loc_pred.shape[0], 1), score, dtype=loc_pred.dtype)
        return np.concatenate([loc_pred, scores], axis=1)
    # non-straight pages
    score_rows = np.tile(np.array([[[0.0, score]]], dtype=loc_pred.dtype), (loc_pred.shape[0], 1, 1))
    return np.concatenate([loc_pred, score_rows], axis=1)


@pytest.mark.parametrize("pages", [[np.zeros((1000, 1000))], [np.zeros((1000, 2000))], [np.zeros((2000, 1000))]])
@pytest.mark.parametrize("preserve_aspect_ratio", [True, False])
@pytest.mark.parametrize("symmetric_pad", [True, False])
@pytest.mark.parametrize("assume_straight_pages", [True, False])
# with_score=True  -> (N, 5) / (N, 5, 2): the box score must be preserved untouched by padding removal
# with_score=False -> (N, 4) / (N, 4, 2): every coordinate is a real point and must be rectified
@pytest.mark.parametrize("with_score", [True, False])
def test_remove_padding(pages, preserve_aspect_ratio, symmetric_pad, assume_straight_pages, with_score):
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

    # Optionally attach the box score; it must survive padding removal unchanged
    if with_score:
        loc_preds = [{k: _attach_score(v) for k, v in d.items()} for d in loc_preds]
        expected = [{k: _attach_score(v) for k, v in d.items()} for d in expected]

    result = _remove_padding(pages, loc_preds, preserve_aspect_ratio, symmetric_pad, assume_straight_pages)
    for res, exp in zip(result, expected):
        assert np.allclose(res["words"], exp["words"])
