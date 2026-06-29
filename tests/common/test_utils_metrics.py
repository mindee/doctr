import numpy as np
import pytest

from doctr.utils import metrics


@pytest.mark.parametrize(
    "gt, pred, raw, caseless, anyascii, unicase",
    [
        [["grass", "56", "True", "EUR"], ["grass", "56", "true", "€"], 0.5, 0.75, 0.75, 1],
        [["éléphant", "ça"], ["elephant", "ca"], 0, 0, 1, 1],
    ],
)
def test_text_match(gt, pred, raw, caseless, anyascii, unicase):
    metric = metrics.TextMatch()
    with pytest.raises(AssertionError):
        metric.summary()

    with pytest.raises(AssertionError):
        metric.update(["a", "b"], ["c"])

    metric.update(gt, pred)
    assert metric.summary() == dict(raw=raw, caseless=caseless, anyascii=anyascii, unicase=unicase)

    metric.reset()
    assert metric.raw == metric.caseless == metric.anyascii == metric.unicase == metric.total == 0


@pytest.mark.parametrize(
    "box1, box2, iou, abs_tol",
    [
        [[[0, 0, 0.5, 0.5]], [[0, 0, 0.5, 0.5]], 1, 0],  # Perfect match
        [[[0, 0, 0.5, 0.5]], [[0.5, 0.5, 1, 1]], 0, 0],  # No match
        [[[0, 0, 1, 1]], [[0.5, 0.5, 1, 1]], 0.25, 0],  # Partial match
        [[[0.2, 0.2, 0.6, 0.6]], [[0.4, 0.4, 0.8, 0.8]], 4 / 28, 1e-7],  # Partial match
        [[[0, 0, 0.1, 0.1]], [[0.9, 0.9, 1, 1]], 0, 0],  # Boxes far from each other
        [np.zeros((0, 4)), [[0, 0, 0.5, 0.5]], 0, 0],  # Zero-sized inputs
        [[[0, 0, 0.5, 0.5]], np.zeros((0, 4)), 0, 0],  # Zero-sized inputs
    ],
)
def test_box_iou(box1, box2, iou, abs_tol):
    iou_mat = metrics.box_iou(np.asarray(box1), np.asarray(box2))
    assert iou_mat.shape == (len(box1), len(box2))
    if iou_mat.size > 0:
        assert abs(iou_mat - iou) <= abs_tol


@pytest.mark.parametrize(
    "rbox1, rbox2, iou, abs_tol",
    [
        [[[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]], [[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]], 1, 0],  # Perfect match
        [[[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]], [[[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]]], 0, 1e-4],  # No match
        [
            [[[0, 0], [1.0, 0], [1.0, 1.0], [0, 1.0]]],
            [[[0.5, 0.5], [1, 0.5], [1.0, 1.0], [0.5, 1]]],
            0.25,
            5e-3,
        ],  # Partial match
        [
            [[[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]],
            [[[0.4, 0.4], [0.8, 0.4], [0.8, 0.8], [0.4, 0.8]]],
            4 / 28,
            7e-3,
        ],  # Partial match
        [
            [[[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]]],
            [[[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]]],
            0,
            0,
        ],  # Boxes far from each other
        [np.zeros((0, 4, 2)), [[[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]]], 0, 0],  # Zero-sized inputs
        [[[[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]]], np.zeros((0, 4, 2)), 0, 0],  # Zero-sized inputs
    ],
)
def test_polygon_iou(rbox1, rbox2, iou, abs_tol):
    iou_mat = metrics.polygon_iou(np.asarray(rbox1), np.asarray(rbox2))
    assert iou_mat.shape == (len(rbox1), len(rbox2))
    if iou_mat.size > 0:
        assert abs(iou_mat - iou) <= abs_tol

    # Ensure broadcasting doesn't change the result
    iou_matbis = metrics.polygon_iou(np.asarray(rbox1), np.asarray(rbox2))
    assert np.all((iou_mat - iou_matbis) <= 1e-7)

    # Incorrect boxes
    with pytest.raises(AssertionError):
        metrics.polygon_iou(np.zeros((2, 5), dtype=float), np.ones((3, 4), dtype=float))


@pytest.mark.parametrize(
    "gts, preds, iou_thresh, recall, precision, mean_iou",
    [
        [[[[0, 0, 0.5, 0.5]]], [[[0, 0, 0.5, 0.5]]], 0.5, 1, 1, 1],  # Perfect match
        [[[[0, 0, 1, 1]]], [[[0, 0, 0.5, 0.5], [0.6, 0.6, 0.7, 0.7]]], 0.2, 1, 0.5, 0.13],  # Bad match
        [[[[0, 0, 1, 1]]], [[[0, 0, 0.5, 0.5], [0.6, 0.6, 0.7, 0.7]]], 0.5, 0, 0, 0.13],  # Bad match
        [
            [[[0, 0, 0.5, 0.5]], [[0, 0, 0.5, 0.5]]],
            [[[0, 0, 0.5, 0.5]], None],
            0.5,
            0.5,
            1,
            1,
        ],  # No preds on 2nd sample
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
    "gts, preds, iou_thresh, recall, precision, mean_iou",
    [
        [
            [[[[0.05, 0.05], [0.15, 0.05], [0.15, 0.15], [0.05, 0.15]]]],
            [[[[0.05, 0.05], [0.15, 0.05], [0.15, 0.15], [0.05, 0.15]]]],
            0.5,
            1,
            1,
            1,
        ],  # Perfect match
        [
            [[[[0.1, 0.05], [0.2, 0.05], [0.2, 0.15], [0.1, 0.15]]]],
            [[[[0.1, 0.05], [0.3, 0.05], [0.3, 0.15], [0.1, 0.15]], [[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8]]]],
            0.2,
            1,
            0.5,
            0.25,
        ],  # Bad match
        [
            [
                [[[0.05, 0.05], [0.15, 0.05], [0.15, 0.15], [0.05, 0.15]]],
                [[[0.25, 0.25], [0.35, 0.25], [35, 0.35], [0.25, 0.35]]],
            ],
            [[[[0.05, 0.05], [0.15, 0.05], [0.15, 0.15], [0.05, 0.15]]], None],
            0.5,
            0.5,
            1,
            1,
        ],  # Empty
    ],
)
def test_r_localization_confusion(gts, preds, iou_thresh, recall, precision, mean_iou):
    metric = metrics.LocalizationConfusion(iou_thresh, use_polygons=True)
    for _gts, _preds in zip(gts, preds):
        metric.update(np.asarray(_gts), np.zeros((0, 5)) if _preds is None else np.asarray(_preds))
    assert metric.summary()[:2] == (recall, precision)
    assert abs(metric.summary()[2] - mean_iou) <= 5e-3
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.matches == metric.tot_iou == 0


@pytest.mark.parametrize(
    "gt_boxes, gt_words, pred_boxes, pred_words, iou_thresh, recall, precision, mean_iou",
    [
        [  # Perfect match
            [[[0, 0, 0.5, 0.5]]],
            [["elephant"]],
            [[[0, 0, 0.5, 0.5]]],
            [["elephant"]],
            0.5,
            {"raw": 1, "caseless": 1, "anyascii": 1, "unicase": 1},
            {"raw": 1, "caseless": 1, "anyascii": 1, "unicase": 1},
            1,
        ],
        [  # Bad match
            [[[0, 0, 0.5, 0.5]]],
            [["elefant"]],
            [[[0, 0, 0.5, 0.5]]],
            [["elephant"]],
            0.5,
            {"raw": 0, "caseless": 0, "anyascii": 0, "unicase": 0},
            {"raw": 0, "caseless": 0, "anyascii": 0, "unicase": 0},
            1,
        ],
        [  # Good match
            [[[0, 0, 1, 1]]],
            [["EUR"]],
            [[[0, 0, 0.5, 0.5], [0.6, 0.6, 0.7, 0.7]]],
            [["€", "e"]],
            0.2,
            {"raw": 0, "caseless": 0, "anyascii": 1, "unicase": 1},
            {"raw": 0, "caseless": 0, "anyascii": 0.5, "unicase": 0.5},
            0.13,
        ],
        [  # No preds on 2nd sample
            [[[0, 0, 0.5, 0.5]], [[0, 0, 0.5, 0.5]]],
            [["Elephant"], ["elephant"]],
            [[[0, 0, 0.5, 0.5]], None],
            [["elephant"], []],
            0.5,
            {"raw": 0, "caseless": 0.5, "anyascii": 0, "unicase": 0.5},
            {"raw": 0, "caseless": 1, "anyascii": 0, "unicase": 1},
            1,
        ],
    ],
)
def test_ocr_metric(gt_boxes, gt_words, pred_boxes, pred_words, iou_thresh, recall, precision, mean_iou):
    metric = metrics.OCRMetric(iou_thresh)
    for _gboxes, _gwords, _pboxes, _pwords in zip(gt_boxes, gt_words, pred_boxes, pred_words):
        metric.update(
            np.asarray(_gboxes), np.zeros((0, 4)) if _pboxes is None else np.asarray(_pboxes), _gwords, _pwords
        )
    _recall, _precision, _mean_iou = metric.summary()
    assert _recall == recall
    assert _precision == precision
    assert _mean_iou == mean_iou
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.tot_iou == 0
    assert metric.raw_matches == metric.caseless_matches == metric.anyascii_matches == metric.unicase_matches == 0
    # Shape check
    with pytest.raises(AssertionError):
        metric.update(
            np.asarray(_gboxes),
            np.zeros((0, 4)),
            _gwords,
            ["I", "have", "a", "bad", "feeling", "about", "this"],
        )


@pytest.mark.parametrize(
    "gt_boxes, gt_classes, pred_boxes, pred_classes, iou_thresh, recall, precision, mean_iou",
    [
        [  # Perfect match
            [[[0, 0, 0.5, 0.5]]],
            [[0]],
            [[[0, 0, 0.5, 0.5]]],
            [[0]],
            0.5,
            1,
            1,
            1,
        ],
        [  # Bad match
            [[[0, 0, 0.5, 0.5]]],
            [[0]],
            [[[0, 0, 0.5, 0.5]]],
            [[1]],
            0.5,
            0,
            0,
            1,
        ],
        [  # No preds on 2nd sample
            [[[0, 0, 0.5, 0.5]], [[0, 0, 0.5, 0.5]]],
            [[0], [1]],
            [[[0, 0, 0.5, 0.5]], None],
            [[0], []],
            0.5,
            0.5,
            1,
            1,
        ],
    ],
)
def test_detection_metric(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_thresh, recall, precision, mean_iou):
    metric = metrics.DetectionMetric(iou_thresh)
    for _gboxes, _gclasses, _pboxes, _pclasses in zip(gt_boxes, gt_classes, pred_boxes, pred_classes):
        metric.update(
            np.asarray(_gboxes),
            np.zeros((0, 4)) if _pboxes is None else np.asarray(_pboxes),
            np.array(_gclasses, dtype=np.int64),
            np.array(_pclasses, dtype=np.int64),
        )
    _recall, _precision, _mean_iou = metric.summary()
    assert _recall == recall
    assert _precision == precision
    assert _mean_iou == mean_iou
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.tot_iou == 0
    assert metric.num_matches == 0
    # Shape check
    with pytest.raises(AssertionError):
        metric.update(
            np.asarray(_gboxes), np.zeros((0, 4)), np.array(_gclasses, dtype=np.int64), np.array([1, 2], dtype=np.int64)
        )


def test_nms():
    boxes = [
        [0.1, 0.1, 0.2, 0.2, 0.95],
        [0.15, 0.15, 0.19, 0.2, 0.90],  # to suppress
        [0.5, 0.5, 0.6, 0.55, 0.90],
        [0.55, 0.5, 0.7, 0.55, 0.85],  # to suppress
    ]
    to_keep = metrics.nms(np.asarray(boxes), thresh=0.2)
    assert to_keep == [0, 2]


@pytest.mark.parametrize(
    "gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, expected_keys",
    [
        [  # Perfect single detection
            [[[0, 0, 1, 1]]],
            [[0]],
            [[[0, 0, 1, 1]]],
            [[0]],
            [[0.9]],
            ["mAP@[.5:.95]", "AP@[.5]", "AP@[.75]", "AP_per_IoU"],
        ],
        [  # Multiple predictions, one correct
            [[[0, 0, 1, 1]]],
            [[0]],
            [[[0, 0, 1, 1], [0.5, 0.5, 0.7, 0.7]]],
            [[0, 0]],
            [[0.9, 0.2]],
            ["mAP@[.5:.95]", "AP@[.5]", "AP@[.75]", "AP_per_IoU"],
        ],
        [  # No predictions
            [[[0, 0, 1, 1]]],
            [[0]],
            [[]],
            [[]],
            [[]],
            ["mAP@[.5:.95]", "AP@[.5]", "AP@[.75]", "AP_per_IoU"],
        ],
        [  # No ground truths
            [[]],
            [[]],
            [[[0, 0, 1, 1]]],
            [[0]],
            [[0.9]],
            ["mAP@[.5:.95]", "AP@[.5]", "AP@[.75]", "AP_per_IoU"],
        ],
    ],
)
def test_object_detection_metric_basic(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, expected_keys):
    metric = metrics.ObjectDetectionMetric()

    for _gboxes, _glabels, _pboxes, _plabels, _pscores in zip(
        gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores
    ):
        metric.update(
            np.asarray(_gboxes, dtype=float),
            np.asarray(_pboxes, dtype=float),
            np.asarray(_glabels, dtype=np.int64),
            np.asarray(_plabels, dtype=np.int64),
            np.asarray(_pscores, dtype=float),
        )

    summary = metric.summary()

    # Key presence
    assert all(k in summary for k in expected_keys)

    # Value ranges
    assert 0.0 <= summary["mAP@[.5:.95]"] <= 1.0
    if summary["AP@[.5]"] is not None:
        assert 0.0 <= summary["AP@[.5]"] <= 1.0

    metric.reset()
    assert metric._gts == []
    assert metric._preds == []


def test_object_detection_metric_cases():
    # Perfect match should yield mAP of 1
    metric = metrics.ObjectDetectionMetric()
    metric.update(
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0.9], dtype=float),
    )
    summary = metric.summary()
    assert summary["mAP@[.5:.95]"] == pytest.approx(1.0, abs=1e-6)
    assert summary["AP@[.5]"] == pytest.approx(1.0, abs=1e-6)

    # Class mismatch should yield mAP of 0
    metric = metrics.ObjectDetectionMetric()
    metric.update(
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),  # wrong class
        np.asarray([0.9], dtype=float),
    )
    summary = metric.summary()
    assert summary["mAP@[.5:.95]"] == pytest.approx(0.0, abs=1e-6)
    assert summary["AP@[.5]"] == pytest.approx(0.0, abs=1e-6)

    # Empty predictions should yield mAP of 0
    metric = metrics.ObjectDetectionMetric()
    metric.update(
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.zeros((0, 4), dtype=float),
        np.asarray([0], dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=float),
    )
    summary = metric.summary()
    assert summary["mAP@[.5:.95]"] == pytest.approx(0.0, abs=1e-6)
    assert summary["AP@[.5]"] == pytest.approx(0.0, abs=1e-6)

    # Multiple classes and samples
    metric = metrics.ObjectDetectionMetric()
    metric.update(
        np.asarray([[0, 0, 1, 1], [0.2, 0.2, 0.4, 0.4]], dtype=float),
        np.asarray([[0, 0, 1, 1], [0.2, 0.2, 0.4, 0.4]], dtype=float),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0.9, 0.8], dtype=float),
    )
    summary = metric.summary()
    assert summary["mAP@[.5:.95]"] == pytest.approx(1.0, abs=1e-6)
    assert summary["AP@[.5]"] == pytest.approx(1.0, abs=1e-6)

    # Shape mismatch should raise an error
    metric = metrics.ObjectDetectionMetric()
    with pytest.raises(AssertionError):
        metric.update(
            np.asarray([[0, 0, 1, 1]], dtype=float),
            np.asarray([[0, 0, 1, 1]], dtype=float),
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),  # mismatch
            np.asarray([0.9], dtype=float),
        )

    # False positives should reduce mAP
    metric = metrics.ObjectDetectionMetric()
    metric.update(
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.asarray([[0.5, 0.5, 0.7, 0.7], [0, 0, 1, 1]], dtype=float),
        np.asarray([0], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([0.9, 0.8], dtype=float),
    )
    summary = metric.summary()
    assert summary["mAP@[.5:.95]"] < 1.0
    assert summary["AP@[.5]"] < 1.0

    # Test with polygons
    metric = metrics.ObjectDetectionMetric(use_polygons=True)
    metric.update(
        np.asarray([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=float),
        np.asarray([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=float),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0.9], dtype=float),
    )
    summary = metric.summary()
    assert summary["mAP@[.5:.95]"] == pytest.approx(1.0, abs=1e-6)
    assert summary["AP@[.5]"] == pytest.approx(1.0, abs=1e-6)

    # False positives should reduce mAP even with perfect localization and class match
    metric = metrics.ObjectDetectionMetric()
    metric.update(
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0.2], dtype=float),
    )
    metric.update(
        np.asarray([[0, 0, 1, 1]], dtype=float),
        np.asarray([[0.5, 0.5, 0.7, 0.7]], dtype=float),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0.9], dtype=float),
    )
    summary = metric.summary()
    # Global ranking should place FP before TP therefore AP must be < 1
    assert summary["mAP@[.5:.95]"] < 1.0


def _square(x, y):
    return [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]]


def _table_cells(use_polygons):
    polygons = np.asarray([_square(0, 0), _square(2, 0), _square(0, 2)], dtype=np.float32)
    if use_polygons:
        return polygons
    return np.concatenate((polygons.min(axis=1), polygons.max(axis=1)), axis=1)


@pytest.mark.parametrize("use_polygons", [False, True])
def test_table_cell_metric(use_polygons):
    gt_cells = _table_cells(use_polygons)
    gt_logic = np.asarray([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int64)

    # Perfect match -> everything is 1
    metric = metrics.TableCellMetric(iou_thresh=0.5, use_polygons=use_polygons)
    metric.update(gt_cells, gt_logic, gt_cells.copy(), gt_logic.copy())
    res = metric.summary()
    assert res["recall"] == 1.0 and res["precision"] == 1.0 and res["f1"] == 1.0
    assert res["structure_acc"] == 1.0

    # One wrong logical coordinate -> geometry perfect, structure accuracy 2/3
    bad_logic = gt_logic.copy()
    bad_logic[1] = [5, 5, 5, 5]
    metric = metrics.TableCellMetric(iou_thresh=0.5, use_polygons=use_polygons)
    metric.update(gt_cells, gt_logic, gt_cells.copy(), bad_logic)
    res = metric.summary()
    assert res["recall"] == 1.0 and res["structure_acc"] == pytest.approx(2 / 3)

    # A missing prediction -> recall 2/3, precision 1
    metric = metrics.TableCellMetric(iou_thresh=0.5, use_polygons=use_polygons)
    metric.update(gt_cells, gt_logic, gt_cells[:2], gt_logic[:2])
    res = metric.summary()
    assert res["recall"] == pytest.approx(2 / 3) and res["precision"] == 1.0

    # Empty edge cases
    empty_cells = np.zeros((0, 4, 2) if use_polygons else (0, 4), dtype=np.float32)
    metric = metrics.TableCellMetric(use_polygons=use_polygons)
    metric.update(gt_cells, gt_logic, empty_cells, np.zeros((0, 4), dtype=np.int64))
    res = metric.summary()
    assert res["recall"] == 0.0 and res["precision"] is None and res["structure_acc"] is None

    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.matches == metric.struct_matches == 0
