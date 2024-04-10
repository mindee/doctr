import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from doctr.utils.metrics import box_iou


@pytest.mark.asyncio
async def test_text_detection(test_app_asyncio, mock_detection_image, mock_txt_file):
    response = await test_app_asyncio.post("/detection", files={"files": [mock_detection_image] * 2})
    assert response.status_code == 200
    json_response = response.json()

    gt_boxes = np.array([[1240, 430, 1355, 470], [1360, 430, 1495, 470]], dtype=np.float32)
    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] / 1654
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] / 2339

    # Check that IoU with GT if reasonable
    assert isinstance(json_response, list) and len(json_response) == 2
    first_pred = json_response[0]  # it's enough to test for the first file because the same image is used twice
    assert isinstance(first_pred, dict) and len(first_pred["geometries"]) == gt_boxes.shape[0]
    pred_boxes = np.array(first_pred["geometries"])
    iou_mat = box_iou(gt_boxes, pred_boxes)
    gt_idxs, pred_idxs = linear_sum_assignment(-iou_mat)
    is_kept = iou_mat[gt_idxs, pred_idxs] >= 0.8
    assert gt_idxs[is_kept].shape[0] == gt_boxes.shape[0]

    response = await test_app_asyncio.post("/detection", files={"files": [mock_txt_file]})
    assert response.status_code == 400
