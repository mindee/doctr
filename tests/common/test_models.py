from io import BytesIO

import cv2
import numpy as np
import pytest
import requests

from doctr.io import reader
from doctr.models._utils import (
    aggregate_confidence,
    estimate_orientation,
    get_language,
    invert_data_structure,
    mask_boxes,
)
from doctr.models.predictor.base import _OCRPredictor
from doctr.utils import geometry


@pytest.fixture(scope="function")
def mock_image(tmpdir_factory):
    url = "https://doctr-static.mindee.com/models?id=v0.2.1/bitmap30.png&src=0"
    file = BytesIO(requests.get(url).content)
    tmp_path = str(tmpdir_factory.mktemp("data").join("mock_bitmap.jpg"))
    with open(tmp_path, "wb") as f:
        f.write(file.getbuffer())
    image = reader.read_img_as_numpy(tmp_path)
    return image


@pytest.fixture(scope="function")
def mock_bitmap(mock_image):
    bitmap = np.squeeze(cv2.cvtColor(mock_image, cv2.COLOR_BGR2GRAY) / 255.0)
    bitmap = np.expand_dims(bitmap, axis=-1)
    return bitmap


def test_estimate_orientation(mock_image, mock_bitmap, mock_tilted_payslip):
    assert estimate_orientation(mock_image * 0) == 0

    # test binarized image
    angle = estimate_orientation(mock_bitmap)
    assert abs(angle) - 30 < 1.0

    angle = estimate_orientation(mock_bitmap * 255)
    assert abs(angle) - 30.0 < 1.0

    angle = estimate_orientation(mock_image)
    assert abs(angle) - 30.0 < 1.0

    rotated = geometry.rotate_image(mock_image, angle)
    angle_rotated = estimate_orientation(rotated)
    assert abs(angle_rotated) == 0

    mock_tilted_payslip = reader.read_img_as_numpy(mock_tilted_payslip)
    assert estimate_orientation(mock_tilted_payslip) == -30
    # A float image (values in [0, 1]) gives the same angle as the same image in uint8
    assert estimate_orientation(mock_tilted_payslip.astype(np.float32) / 255) == -30

    rotated = geometry.rotate_image(mock_tilted_payslip, -30, expand=True)
    angle_rotated = estimate_orientation(rotated)
    assert abs(angle_rotated) < 1.0

    with pytest.raises(AssertionError):
        estimate_orientation(np.ones((10, 10, 10)))

    # test with general_page_orientation
    assert estimate_orientation(mock_bitmap, (90, 0.9)) in range(140, 160)

    rotated = geometry.rotate_image(mock_tilted_payslip, -30)
    assert estimate_orientation(rotated, (0, 0.9)) in range(-10, 10)

    assert estimate_orientation(mock_image, (0, 0.9)) - 30 < 1.0

    # Aspect Ratio Independence (Portrait vs Landscape)
    # Pad the tilted image to be very tall (Portrait)
    portrait_img = cv2.copyMakeBorder(mock_tilted_payslip, 500, 500, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # Pad the tilted image to be very wide (Landscape)
    landscape_img = cv2.copyMakeBorder(mock_tilted_payslip, 0, 0, 500, 500, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    assert abs(estimate_orientation(portrait_img) - (-30)) <= 1.0
    assert abs(estimate_orientation(landscape_img) - (-30)) <= 1.0

    # Perpendicular Noise Test
    vertical_noise = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.line(vertical_noise, (500, 100), (500, 900), (255, 255, 255), 10)
    assert estimate_orientation(vertical_noise) == 0


def test_get_lang():
    sentence = "This is a test sentence."
    expected_lang = "en"
    threshold_prob = 0.99

    lang = get_language(sentence)

    assert lang[0] == expected_lang
    assert lang[1] > threshold_prob

    lang = get_language("a")
    assert lang[0] == "unknown"
    assert lang[1] == 0.0


def test_mask_boxes():
    img = np.full((100, 200, 3), 100, dtype=np.uint8)
    boxes = np.array([[0.1, 0.2, 0.4, 0.6]], dtype=np.float32)
    out = mask_boxes(img, boxes, fill_value=0)
    assert out is not img
    assert np.all(img == 100)
    assert out.dtype == img.dtype and out.shape == img.shape

    # abs box -> x in [20, 80], y in [20, 60] -> interior filled, far outside unchanged
    assert np.all(out[25:55, 25:75] == 0)
    assert out[5, 5, 0] == 100
    assert out[90, 190, 0] == 100
    # rotated polygons (N, 4, 2), relative coords
    polys = np.array([[[0.5, 0.1], [0.9, 0.1], [0.9, 0.5], [0.5, 0.5]]], dtype=np.float32)
    out = mask_boxes(img, polys, fill_value=0)

    # abs box -> x in [100, 180], y in [10, 50]
    assert np.all(out[15:45, 105:175] == 0)
    assert out[5, 5, 0] == 100
    # default fill value is 255
    out = mask_boxes(np.zeros((50, 50, 3), dtype=np.uint8), np.array([[0.2, 0.2, 0.8, 0.8]], np.float32))
    assert np.all(out[15:35, 15:35] == 255)
    # multiple boxes in a single call
    boxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0]], dtype=np.float32)
    out = mask_boxes(img, boxes, fill_value=0)
    assert out[5, 5, 0] == 0 and out[95, 195, 0] == 0
    assert out[50, 100, 0] == 100  # center untouched
    # out-of-range coords are clipped to the image (no error, whole image covered)
    out = mask_boxes(img, np.array([[-0.5, -0.5, 1.5, 1.5]], dtype=np.float32), fill_value=0)
    assert np.all(out == 0)
    # empty boxes
    assert mask_boxes(img, np.zeros((0, 4), dtype=np.float32)) is img
    assert mask_boxes(img, np.zeros((0, 4, 2), dtype=np.float32)) is img
    # grayscale
    gray = np.full((100, 200), 100, dtype=np.uint8)
    out = mask_boxes(gray, np.array([[0.1, 0.2, 0.4, 0.6]], dtype=np.float32), fill_value=0)
    assert np.all(out[25:55, 25:75] == 0)
    assert out[5, 5] == 100


def test_convert_list_dict():
    dic = {"k1": [[0], [0], [0]], "k2": [[1], [1], [1]]}
    tar_dict = [{"k1": [0], "k2": [1]}, {"k1": [0], "k2": [1]}, {"k1": [0], "k2": [1]}]

    converted_dic = invert_data_structure(dic)
    converted_list = invert_data_structure(tar_dict)

    assert converted_dic == tar_dict
    assert converted_list == dic
    # empty inputs
    assert invert_data_structure({}) == []
    assert invert_data_structure([]) == {}
    assert invert_data_structure({"k1": [], "k2": []}) == []
    with pytest.raises(AssertionError):
        invert_data_structure({"k1": [0], "k2": []})


@pytest.mark.parametrize(
    "method, expected",
    [
        ("mean", 0.77),
        ("min", 0.12),
        ("max", 0.99),
        ("median", 0.985),
        ("geometric_mean", 0.5827),
        ("harmonic_mean", 0.3517),
        (lambda scores: float(np.percentile(scores, 25)), 0.765),
    ],
)
def test_aggregate_confidence(method, expected):
    # One uncertain character among confident ones
    scores = np.array([0.99, 0.98, 0.12, 0.99], dtype=np.float32)
    assert aggregate_confidence(scores, method) == pytest.approx(expected, abs=1e-4)
    assert aggregate_confidence(scores.tolist(), method) == pytest.approx(expected, abs=1e-4)
    assert aggregate_confidence(scores.reshape(2, 2), method) == pytest.approx(expected, abs=1e-4)
    assert aggregate_confidence([], method) == 0.0


def test_aggregate_confidence_edge_cases():
    # A zero score cancels the geometric and harmonic means
    assert aggregate_confidence([0.0, 0.9], "geometric_mean") == 0.0
    assert aggregate_confidence([0.0, 0.9], "harmonic_mean") == 0.0
    # The result is clipped to [0, 1]
    assert aggregate_confidence([0.5], lambda scores: -1.0) == 0.0
    assert aggregate_confidence([0.5], lambda scores: 2.0) == 1.0
    # A NaN aggregation (e.g. the mean of an empty selection) counts as no confidence, a 1-element array is accepted
    assert aggregate_confidence([0.4, 0.5], lambda scores: np.mean(scores[scores > 0.6])) == 0.0
    assert aggregate_confidence([0.4, 0.5], lambda scores: np.quantile(scores, [0.5])) == pytest.approx(0.45)
    # Invalid methods, including values which are not hashable
    for invalid in ["average", ["mean"], {"mean"}, None, 1]:
        with pytest.raises(ValueError, match="Unknown confidence aggregation"):
            aggregate_confidence([0.5], invalid)


def test_align_hook_output():
    boxes = [
        np.array([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6], [0.8, 0.1, 0.9, 0.2]], dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
    ]
    scores = [np.array([0.9, 0.3, 0.6], dtype=np.float32), np.zeros((0,), dtype=np.float32)]

    # Unchanged boxes keep their scores
    out_boxes, out_scores = _OCRPredictor._align_hook_output(boxes, [b.copy() for b in boxes], scores)
    assert all(np.array_equal(a, b) for a, b in zip(out_boxes, boxes))
    assert out_scores[0] == pytest.approx([0.9, 0.3, 0.6])

    # Moved, reordered and filtered boxes keep the score of their closest box; lists are converted to arrays,
    # and a box added on a page without predictions gets a score of 1
    hooked = [np.array([[0.81, 0.1, 0.91, 0.2], [0.11, 0.1, 0.21, 0.2]]), [[0.0, 0.0, 0.1, 0.1]]]
    out_boxes, out_scores = _OCRPredictor._align_hook_output(boxes, hooked, scores)
    assert out_boxes[0].dtype == np.float32 and out_boxes[0].shape == (2, 4)
    assert out_boxes[1].shape == (1, 4)
    assert out_scores[0] == pytest.approx([0.6, 0.9])
    assert out_scores[1] == pytest.approx([1.0])

    # All the boxes can be removed
    out_boxes, out_scores = _OCRPredictor._align_hook_output(boxes, [[], []], scores)
    assert [b.shape for b in out_boxes] == [(0, 4), (0, 4)]
    assert [len(s) for s in out_scores] == [0, 0]

    # Rotated boxes
    polys = [
        np.array([
            [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
            [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6]],
        ])
    ]
    out_boxes, out_scores = _OCRPredictor._align_hook_output(polys, [polys[0][1:]], [np.array([0.4, 0.7])])
    assert out_boxes[0].shape == (1, 4, 2)
    assert out_scores[0] == pytest.approx([0.7])

    # Invalid outputs
    with pytest.raises(ValueError, match="one array of boxes per page"):
        _OCRPredictor._align_hook_output(boxes, boxes[:1], scores)
    with pytest.raises(ValueError, match="one array of boxes per page"):
        _OCRPredictor._align_hook_output(boxes, boxes[0], scores)
    with pytest.raises(ValueError, match="format of the boxes"):
        _OCRPredictor._align_hook_output(boxes, [boxes[0][:, :2], boxes[1]], scores)
