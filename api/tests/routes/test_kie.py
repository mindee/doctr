import numpy as np
import pytest


def common_test(json_response, expected_response):
    first_pred = json_response[0]  # it's enough to test for the first file because the same image is used twice
    assert isinstance(first_pred["name"], str)
    assert (
        isinstance(first_pred["dimensions"], (tuple, list))
        and len(first_pred["dimensions"]) == 2
        and all(isinstance(dim, int) for dim in first_pred["dimensions"])
    )
    assert isinstance(first_pred["predictions"], list)
    assert isinstance(expected_response["predictions"], list)

    for pred, expected_pred in zip(first_pred["predictions"], expected_response["predictions"]):
        assert pred["class_name"] == expected_pred["class_name"]
        assert isinstance(pred["items"], list)
        assert isinstance(expected_pred["items"], list)

        for pred_item, expected_pred_item in zip(pred["items"], expected_pred["items"]):
            assert isinstance(pred_item["value"], str) and pred_item["value"] == expected_pred_item["value"]
            assert isinstance(pred_item["confidence"], (int, float))
            np.testing.assert_allclose(pred_item["geometry"], expected_pred_item["geometry"], rtol=1e-2)
            assert isinstance(pred_item["objectness_score"], (int, float))
            assert isinstance(pred_item["crop_orientation"], dict)
            assert isinstance(pred_item["crop_orientation"]["value"], int) and isinstance(
                pred_item["crop_orientation"]["confidence"], (float, int, type(None))
            )


@pytest.mark.asyncio
async def test_kie_box(test_app_asyncio, mock_detection_image, mock_kie_response):
    headers = {
        "accept": "application/json",
    }
    params = {"det_arch": "db_resnet50", "reco_arch": "crnn_vgg16_bn"}
    files = [
        ("files", ("test.jpg", mock_detection_image, "image/jpeg")),
        ("files", ("test2.jpg", mock_detection_image, "image/jpeg")),
    ]
    response = await test_app_asyncio.post("/kie", params=params, files=files, headers=headers)
    assert response.status_code == 200
    json_response = response.json()

    expected_box_response = mock_kie_response["box"]
    assert isinstance(json_response, list) and len(json_response) == 2
    common_test(json_response, expected_box_response)


@pytest.mark.asyncio
async def test_kie_poly(test_app_asyncio, mock_detection_image, mock_kie_response):
    headers = {
        "accept": "application/json",
    }
    params = {"det_arch": "db_resnet50", "reco_arch": "crnn_vgg16_bn", "assume_straight_pages": False}
    files = [
        ("files", ("test.jpg", mock_detection_image, "image/jpeg")),
        ("files", ("test2.jpg", mock_detection_image, "image/jpeg")),
    ]
    response = await test_app_asyncio.post("/kie", params=params, files=files, headers=headers)
    assert response.status_code == 200
    json_response = response.json()

    expected_poly_response = mock_kie_response["poly"]
    assert isinstance(json_response, list) and len(json_response) == 2
    common_test(json_response, expected_poly_response)


@pytest.mark.asyncio
async def test_kie_invalid_file(test_app_asyncio, mock_txt_file):
    headers = {
        "accept": "application/json",
    }
    files = [
        ("files", ("test.txt", mock_txt_file)),
    ]
    response = await test_app_asyncio.post("/kie", files=files, headers=headers)
    assert response.status_code == 400
