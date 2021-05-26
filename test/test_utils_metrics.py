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
        [[[0, 0, .5, .5]], [[0, 0, .5, .5]], 1, 0],  # Perfect match
        [[[0, 0, .5, .5]], [[.5, .5, 1, 1]], 0, 0],  # No match
        [[[0, 0, 1, 1]], [[.5, .5, 1, 1]], 0.25, 0],  # Partial match
        [[[.2, .2, .6, .6]], [[.4, .4, .8, .8]], 4 / 28, 1e-7],  # Partial match
        [[[0, 0, .1, .1]], [[.9, .9, 1, 1]], 0, 0],  # Boxes far from each other
        [np.zeros((0, 4)), [[0, 0, .5, .5]], 0, 0],  # Zero-sized inputs
        [[[0, 0, .5, .5]], np.zeros((0, 4)), 0, 0],  # Zero-sized inputs
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
        [[[[0, 0, .5, .5]]], [[[0, 0, .5, .5]]], 0.5, 1, 1, 1],  # Perfect match
        [[[[0, 0, 1, 1]]], [[[0, 0, .5, .5], [.6, .6, .7, .7]]], 0.2, 1, 0.5, 0.125],  # Bad match
        [[[[0, 0, 1, 1]]], [[[0, 0, .5, .5], [.6, .6, .7, .7]]], 0.5, 0, 0, 0.125],  # Bad match
        [[[[0, 0, .5, .5]], [[0, 0, .5, .5]]], [[[0, 0, .5, .5]], None], 0.5, 0.5, 1, 1],  # No preds on 2nd sample
    ],
)
def test_localization_confusion(gts, preds, iou_thresh, recall, precision, mean_iou):

    metric = metrics.LocalizationConfusion(iou_thresh)
    for _gts, _preds in zip(gts, preds):
        metric.update(np.asarray(_gts), np.zeros((0, 4)) if _preds is None else np.asarray(_preds))
    assert metric.summary() == (recall, precision, mean_iou)
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.matches == metric.tot_iou == 0


@pytest.mark.parametrize(
    "gt_boxes, gt_words, pred_boxes, pred_words, iou_thresh, recall, precision, mean_iou",
    [
        [  # Perfect match
            [[[0, 0, .5, .5]]], [["elephant"]],
            [[[0, 0, .5, .5]]], [["elephant"]],
            0.5,
            {"raw": 1, "caseless": 1, "unidecode": 1, "unicase": 1},
            {"raw": 1, "caseless": 1, "unidecode": 1, "unicase": 1},
            1,
        ],
        [  # Bad match
            [[[0, 0, .5, .5]]], [["elefant"]],
            [[[0, 0, .5, .5]]], [["elephant"]],
            0.5,
            {"raw": 0, "caseless": 0, "unidecode": 0, "unicase": 0},
            {"raw": 0, "caseless": 0, "unidecode": 0, "unicase": 0},
            1,
        ],
        [  # Good match
            [[[0, 0, 1, 1]]], [["EUR"]],
            [[[0, 0, .5, .5], [.6, .6, .7, .7]]], [["€", "e"]],
            0.2,
            {"raw": 0, "caseless": 0, "unidecode": 1, "unicase": 1},
            {"raw": 0, "caseless": 0, "unidecode": .5, "unicase": .5},
            0.125,
        ],
        [  # No preds on 2nd sample
            [[[0, 0, .5, .5]], [[0, 0, .5, .5]]], [["Elephant"], ["elephant"]],
            [[[0, 0, .5, .5]], None], [["elephant"], []],
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
            np.zeros((0, 4)) if _pboxes is None else np.asarray(_pboxes),
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
