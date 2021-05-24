import pytest
import numpy as np
from doctr.utils import metrics


@pytest.mark.parametrize(
    "gt, pred, raw, caseless, unidecode, unicase",
    [
        [['grass', '56', 'True', 'EUR'], ['grass', '56', 'true', '€'], .5, .75, .75, 1],
        [['éléphant', 'ça'], ['elephant', 'ca'], 0, 0, 1, 1],
    ],
)
def test_text_match(gt, pred, raw, caseless, unidecode, unicase):
    metric = metrics.TextMatch()
    with pytest.raises(AssertionError):
        metric.summary()

    with pytest.raises(AssertionError):
        metric.update(['a', 'b'], ['c'])

    metric.update(gt, pred)
    assert metric.summary() == dict(raw=raw, caseless=caseless, unidecode=unidecode, unicase=unicase)

    metric.reset()
    assert metric.raw == metric.caseless == metric.unidecode == metric.unicase == metric.total == 0


@pytest.mark.parametrize(
    "box1, box2, iou, abs_tol",
    [
        [[[.1, .1, .1, .1, 0]], [[.1, .1, .1, .1, 0]], 1, 0],  # Perfect match
        [[[.1, .1, .1, .1, 0]], [[.5, .5, .2, .2, 0]], 0, 0],  # No match
        [[[.5, .5, .2, .2, 0]], [[.4, .4, .2, .2, 0]], 1 / 7, 1e-7],  # Partial match
        [[[.1, .1, .1, .1, 0]], [[.9, .9, .1, .1, 0]], 0, 0],  # Boxes far from each other
        [np.zeros((0, 5)), [[0, 0, .5, .5, 0]], 0, 0],  # Zero-sized inputs
        [[[0, 0, .5, .5, 0]], np.zeros((0, 5)), 0, 0],  # Zero-sized inputs
    ],
)
def test_box_iou(box1, box2, iou, abs_tol):
    iou_mat = metrics.box_iou(np.asarray(box1), np.asarray(box2))
    assert iou_mat.shape == (len(box1), len(box2))
    if iou_mat.size > 0:
        assert abs(iou_mat - iou) <= abs_tol


@pytest.mark.parametrize(
    "gts, preds, iou_thresh, recall, precision, mean_iou",
    [
        [[[[.1, .1, .1, .1, 0]]], [[[.1, .1, .1, .1, 0]]], 0.5, 1, 1, 1],  # Perfect match
        [[[[.1, .1, .1, .1, 0]]], [[[.1, .1, .05, .05, 0], [.6, .6, .2, .2, 0]]], 0.2, 1, 0.5, 0.125],  # Bad match
        [[[[.1, .1, .1, .1, 0]], [[.3, .3, .1, .1, 0]]], [[[.1, .1, .1, .1, 0]], None], 0.5, 0.5, 1, 1],  # Empty
    ],
)
def test_localization_confusion(gts, preds, iou_thresh, recall, precision, mean_iou):

    metric = metrics.LocalizationConfusion(iou_thresh)
    for _gts, _preds in zip(gts, preds):
        metric.update(np.asarray(_gts), np.zeros((0, 5)) if _preds is None else np.asarray(_preds))
    assert metric.summary()[:2] == (recall, precision)
    assert abs(metric.summary()[2] - mean_iou) <= 1e-7
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.matches == metric.tot_iou == 0


@pytest.mark.parametrize(
    "gt_boxes, gt_words, pred_boxes, pred_words, iou_thresh, recall, precision, mean_iou",
    [
        [  # Perfect match
            [[[.5, .5, .5, .5, 0]]], [["elephant"]],
            [[[.5, .5, .5, .5, 0]]], [["elephant"]],
            0.5,
            {"raw": 1, "caseless": 1, "unidecode": 1, "unicase": 1},
            {"raw": 1, "caseless": 1, "unidecode": 1, "unicase": 1},
            1,
        ],
        [  # Bad match
            [[[.5, .5, .5, .5, 0]]], [["elefant"]],
            [[[.5, .5, .5, .5, 0]]], [["elephant"]],
            0.5,
            {"raw": 0, "caseless": 0, "unidecode": 0, "unicase": 0},
            {"raw": 0, "caseless": 0, "unidecode": 0, "unicase": 0},
            1,
        ],
        [  # Good match
            [[[.5, .5, 1, 1, 0]]], [["EUR"]],
            [[[.5, .5, .5, .5, 0], [.6, .6, .5, .5, 0]]], [["€", "e"]],
            0.2,
            {"raw": 0, "caseless": 0, "unidecode": 1, "unicase": 1},
            {"raw": 0, "caseless": 0, "unidecode": .5, "unicase": .5},
            0.125,
        ],
        [  # No preds on 2nd sample
            [[[.6, .6, .5, .5, 0]], [[.4, .4, .5, .5, 0]]], [["Elephant"], ["elephant"]],
            [[[.6, .6, .5, .5, 0]], None], [["elephant"], []],
            0.5,
            {"raw": 0, "caseless": .5, "unidecode": 0, "unicase": .5},
            {"raw": 0, "caseless": 1, "unidecode": 0, "unicase": 1},
            1,
        ],
    ],
)
def test_ocr_metric(
    gt_boxes, gt_words, pred_boxes, pred_words, iou_thresh, recall, precision, mean_iou
):
    metric = metrics.OCRMetric(iou_thresh)
    for _gboxes, _gwords, _pboxes, _pwords in zip(gt_boxes, gt_words, pred_boxes, pred_words):
        metric.update(
            np.asarray(_gboxes),
            np.zeros((0, 5)) if _pboxes is None else np.asarray(_pboxes),
            _gwords,
            _pwords
        )
    _recall, _precision, _mean_iou = metric.summary()
    assert _recall == recall
    assert _precision == precision
    assert _mean_iou == mean_iou
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.tot_iou == 0
    assert metric.raw_matches == metric.caseless_matches == metric.unidecode_matches == metric.unicase_matches == 0
