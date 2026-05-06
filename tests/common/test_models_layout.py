import numpy as np
import pytest

from doctr.models.layout.lw_detr.base import LWDETRPostProcessor


def test_lwdetr_postprocessor():
    postprocessor = LWDETRPostProcessor(
        num_classes=5,
        score_thresh=0.2,
        iou_thresh=0.5,
        topk=50,
        assume_straight_pages=True,
    )

    r_postprocessor = LWDETRPostProcessor(
        num_classes=5,
        score_thresh=0.2,
        iou_thresh=0.5,
        topk=50,
        assume_straight_pages=False,
    )

    # Input validation
    with pytest.raises(Exception):
        postprocessor(np.random.rand(2, 20, 5).astype(np.float32), np.random.rand(2, 20, 5).astype(np.float32))

    # Forward pass
    batch_size, num_queries = 2, 20
    logits = np.random.randn(batch_size, num_queries, 6).astype(np.float32)
    boxes = np.random.rand(batch_size, num_queries, 6).astype(np.float32)

    out = postprocessor(logits, boxes)
    r_out = r_postprocessor(logits, boxes)

    # Batch composition
    assert isinstance(out, list)
    assert len(out) == batch_size

    assert isinstance(r_out, list)
    assert len(r_out) == batch_size

    assert all(isinstance(sample, tuple) and len(sample) == 3 for sample in out)

    assert all(isinstance(sample, tuple) and len(sample) == 3 for sample in r_out)

    labels, bboxes, scores = out[0]
    r_labels, r_bboxes, r_scores = r_out[0]

    assert isinstance(labels, list)
    assert isinstance(scores, list)
    assert isinstance(bboxes, np.ndarray)

    # straight pages → (K, 4)
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4

    # rotated pages → (K, 4, 2)
    assert isinstance(r_bboxes, np.ndarray)
    assert r_bboxes.ndim == 3
    assert r_bboxes.shape[2] == 2

    # Score / label validity
    assert all(isinstance(s, float) for s in scores)
    assert all(s >= 0 for s in scores)
    assert len(labels) == len(scores)
