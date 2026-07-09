from io import BytesIO

import cv2
import numpy as np
import pytest
import requests

from doctr.io import reader
from doctr.models._utils import estimate_orientation, get_language, invert_data_structure, mask_boxes
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
