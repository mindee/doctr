from io import BytesIO

import cv2
import numpy as np
import pytest
import requests

from doctr.io import reader
from doctr.models._utils import estimate_orientation, get_bitmap_angle, get_language
from doctr.utils import geometry


@pytest.fixture(scope="function")
def mock_image(tmpdir_factory):
    url = 'https://github.com/mindee/doctr/releases/download/v0.2.1/bitmap30.png'
    file = BytesIO(requests.get(url).content)
    tmp_path = str(tmpdir_factory.mktemp("data").join("mock_bitmap.jpg"))
    with open(tmp_path, 'wb') as f:
        f.write(file.getbuffer())
    image = reader.read_img_as_numpy(tmp_path)
    return image


@pytest.fixture(scope="function")
def mock_bitmap(mock_image):
    bitmap = np.squeeze(cv2.cvtColor(mock_image, cv2.COLOR_BGR2GRAY) / 255.)
    return bitmap


def test_get_bitmap_angle(mock_bitmap):
    angle = get_bitmap_angle(mock_bitmap)
    assert abs(angle - 30.) < 1.


def test_estimate_orientation(mock_image, mock_tilted_payslip):
    assert estimate_orientation(mock_image * 0) == 0

    angle = estimate_orientation(mock_image)
    assert abs(angle - 30.) < 1.

    rotated = geometry.rotate_image(mock_image, -angle)
    angle_rotated = estimate_orientation(rotated)
    assert abs(angle_rotated) < 1.

    mock_tilted_payslip = reader.read_img_as_numpy(mock_tilted_payslip)
    assert (estimate_orientation(mock_tilted_payslip) - 30.) < 1.

    rotated = geometry.rotate_image(mock_tilted_payslip, -30, expand=True)
    angle_rotated = estimate_orientation(rotated)
    assert abs(angle_rotated) < 1.


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
