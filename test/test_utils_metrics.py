import pytest
import numpy as np
from doctr.utils import metrics


@pytest.mark.parametrize(
    "gt, pred, ignore_case, ignore_accents, result",
    [
        [['grass', '56', 'True', 'STOP'], ['grass', '56', 'true', 'stop'], True, False, 1.0],
        [['grass', '56', 'True', 'STOP'], ['grass', '56', 'true', 'stop'], False, False, .5],
        [['éléphant'], ['elephant'], False, True, 1.0],
    ],
)
def test_exact_match(gt, pred, ignore_case, ignore_accents, result):
    metric = metrics.ExactMatch(ignore_case, ignore_accents)
    with pytest.raises(AssertionError):
        metric.summary()
    if ignore_accents:
        with pytest.raises(NotImplementedError):
            metric.update(gt, pred)
    else:
        metric.update(gt, pred)
        assert metric.summary() == result


@pytest.mark.parametrize(
    "mat, row_indices, col_indices",
    [
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2], [0, 1, 2]],  # Perfect diagonal
        [[[1, 0, 0], [0, .4, 0], [0, 0, 1]], [0, 2], [0, 2]],  # below threshold
        [[[1, 0, 0], [0, 1, 0], [0, 0, 0.5], [0, 0, 1]], [0, 1, 3], [0, 1, 2]],  # rectangular matrix
    ],
)
def test_assign_pairs(mat, row_indices, col_indices):
    gt_idxs, pred_idxs = metrics.assign_pairs(np.asarray(mat))
    assert all(a == b for a, b in zip(row_indices, gt_idxs)), print(gt_idxs)
    assert all(a == b for a, b in zip(col_indices, pred_idxs)), print(pred_idxs)


@pytest.mark.parametrize(
    "box1, box2, iou, abs_tol",
    [
        [[0, 0, .5, .5], [0, 0, .5, .5], 1, 0],  # Perfect match
        [[0, 0, .5, .5], [.5, .5, 1, 1], 0, 0],  # No match
        [[0, 0, 1, 1], [.5, .5, 1, 1], 0.25, 0],  # Partial match
        [[.2, .2, .6, .6], [.4, .4, .8, .8], 4 / 28, 1e-7],  # Partial match
        [[0, 0, .1, .1], [.9, .9, 1, 1], 0, 0],  # Boxes far from each other
    ],
)
def test_box_iou(box1, box2, iou, abs_tol):
    assert abs(metrics.box_iou(np.asarray([box1]), np.asarray([box2])) - iou) <= abs_tol


@pytest.mark.parametrize(
    "gts, preds, iou_thresh, recall, precision, mean_iou",
    [
        [[[0, 0, .5, .5]], [[0, 0, .5, .5]], 0.5, 1, 1, 1],  # Perfect match
        [[[0, 0, 1, 1]], [[0, 0, .5, .5], [.6, .6, .7, .7]], 0.2, 1, 0.5, 0.125],  # Bad match
        [[[0, 0, 1, 1]], [[0, 0, .5, .5], [.6, .6, .7, .7]], 0.5, 0, 0, 0.125],  # Bad match
    ],
)
def test_localization_confusion(gts, preds, iou_thresh, recall, precision, mean_iou):

    metric = metrics.LocalizationConfusion(iou_thresh)
    metric.update(np.asarray(gts), np.asarray(preds))
    assert metric.summary() == (recall, precision, mean_iou)


@pytest.mark.parametrize(
    "gts_vert, preds_vert, gts_texts, preds_texts, iou_thresh, max_dist, recall, precision, mean_iou, mean_distance",
    [
        [[[0, 0, .5, .5]], [[0, 0, .5, .5]], ["elephant"], ["elephant"], 0.5, 0, 1, 1, 1, 0],  # Perfect match
        [[[0, 0, .5, .5]], [[0, 0, .5, .5]], ["elefant"], ["elephant"], 0.5, 1, 0, 0, 1, 2],  # Bad match
        [[[0, 0, 1, 1]], [[0, 0, .5, .5], [.6, .6, .7, .7]], ["home"], ["hom"], 0.2, 1, 1, 0.5, 0.125, 1],  # Good match
    ],
)
def test_ocr_metric(
    gts_vert, preds_vert, gts_texts, preds_texts, iou_thresh, max_dist, recall, precision, mean_iou, mean_distance
):
    metric = metrics.OCRMetric(iou_thresh, max_dist)
    metric.update(np.asarray(gts_vert), np.asarray(preds_vert), gts_texts, preds_texts)
    assert metric.summary() == (recall, precision, mean_iou, mean_distance)
