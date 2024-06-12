from io import BytesIO

import cv2
import numpy as np
import pytest
import requests

from doctr.io import reader
from doctr.models._utils import estimate_orientation, get_language, invert_data_structure
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


def test_convert_list_dict():
    dic = {"k1": [[0], [0], [0]], "k2": [[1], [1], [1]]}
    tar_dict = [{"k1": [0], "k2": [1]}, {"k1": [0], "k2": [1]}, {"k1": [0], "k2": [1]}]

    converted_dic = invert_data_structure(dic)
    converted_list = invert_data_structure(tar_dict)

    assert converted_dic == tar_dict
    assert converted_list == dic
