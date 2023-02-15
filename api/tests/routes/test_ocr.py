import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from doctr.utils.metrics import box_iou


@pytest.mark.asyncio
async def test_perform_ocr(test_app_asyncio, mock_detection_image):
    response = await test_app_asyncio.post("/ocr", files={"file": mock_detection_image})
    assert response.status_code == 200
    json_response = response.json()

    gt_boxes = np.array([[1240, 430, 1355, 470], [1360, 430, 1495, 470]], dtype=np.float32)
    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] / 1654
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] / 2339
    gt_labels = ["Hello", "world!"]

    # Check that IoU with GT if reasonable
    assert isinstance(json_response, list) and len(json_response) == gt_boxes.shape[0]
    pred_boxes = np.array([elt["box"] for elt in json_response])
    pred_labels = np.array([elt["value"] for elt in json_response])
    iou_mat = box_iou(gt_boxes, pred_boxes)
    gt_idxs, pred_idxs = linear_sum_assignment(-iou_mat)
    is_kept = iou_mat[gt_idxs, pred_idxs] >= 0.8
    gt_idxs, pred_idxs = gt_idxs[is_kept], pred_idxs[is_kept]
    assert gt_idxs.shape[0] == gt_boxes.shape[0]
    assert all(gt_labels[gt_idx] == pred_labels[pred_idx] for gt_idx, pred_idx in zip(gt_idxs, pred_idxs))
