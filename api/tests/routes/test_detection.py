import numpy as np
import pytest


def common_test(json_response, expected_response):
    assert isinstance(json_response, list) and len(json_response) == 2
    first_pred = json_response[0]  # it's enough to test for the first file because the same image is used twice

    assert isinstance(first_pred["name"], str)
    np.testing.assert_allclose(first_pred["geometries"], expected_response["geometries"], rtol=1e-2)


@pytest.mark.asyncio
async def test_text_detection_box(test_app_asyncio, mock_detection_image, mock_detection_response):
    headers = {
        "accept": "application/json",
    }
    params = {"det_arch": "db_resnet50"}
    files = [
        ("files", ("test.jpg", mock_detection_image, "image/jpeg")),
        ("files", ("test2.jpg", mock_detection_image, "image/jpeg")),
    ]
    response = await test_app_asyncio.post("/detection", params=params, files=files, headers=headers)
    assert response.status_code == 200
    json_response = response.json()

    expected_box_response = mock_detection_response["box"]
    common_test(json_response, expected_box_response)


@pytest.mark.asyncio
async def test_text_detection_poly(test_app_asyncio, mock_detection_image, mock_detection_response):
    headers = {
        "accept": "application/json",
    }
    params = {"det_arch": "db_resnet50", "assume_straight_pages": False}
    files = [
        ("files", ("test.jpg", mock_detection_image, "image/jpeg")),
        ("files", ("test2.jpg", mock_detection_image, "image/jpeg")),
    ]
    response = await test_app_asyncio.post("/detection", params=params, files=files, headers=headers)
    assert response.status_code == 200
    json_response = response.json()

    expected_poly_response = mock_detection_response["poly"]
    common_test(json_response, expected_poly_response)


@pytest.mark.asyncio
async def test_text_detection_invalid_file(test_app_asyncio, mock_txt_file):
    headers = {
        "accept": "application/json",
    }
    files = [
        ("files", ("test.txt", mock_txt_file)),
    ]
    response = await test_app_asyncio.post("/detection", files=files, headers=headers)
    assert response.status_code == 400
