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
    for item, expected_item in zip(first_pred["items"], expected_response["items"]):
        for block, expected_block in zip(item["blocks"], expected_item["blocks"]):
            np.testing.assert_allclose(block["geometry"], expected_block["geometry"], rtol=1e-2)
            assert isinstance(block["objectness_score"], (int, float))
            for line, expected_line in zip(block["lines"], expected_block["lines"]):
                np.testing.assert_allclose(line["geometry"], expected_line["geometry"], rtol=1e-2)
                assert isinstance(line["objectness_score"], (int, float))
                for word, expected_word in zip(line["words"], expected_line["words"]):
                    np.testing.assert_allclose(word["geometry"], expected_word["geometry"], rtol=1e-2)
                    assert isinstance(word["objectness_score"], (int, float))
                    assert isinstance(word["value"], str) and word["value"] == expected_word["value"]
                    assert isinstance(word["confidence"], (int, float))
                    assert isinstance(word["crop_orientation"], dict)
                    assert isinstance(word["crop_orientation"]["value"], int) and isinstance(
                        word["crop_orientation"]["confidence"], (float, int, type(None))
                    )


@pytest.mark.asyncio
async def test_ocr_box(test_app_asyncio, mock_detection_image, mock_ocr_response):
    headers = {
        "accept": "application/json",
    }
    params = {"det_arch": "db_resnet50", "reco_arch": "crnn_vgg16_bn"}
    files = [
        ("files", ("test.jpg", mock_detection_image, "image/jpeg")),
        ("files", ("test2.jpg", mock_detection_image, "image/jpeg")),
    ]
    response = await test_app_asyncio.post("/ocr", params=params, files=files, headers=headers)
    assert response.status_code == 200
    json_response = response.json()

    expected_box_response = mock_ocr_response["box"]
    assert isinstance(json_response, list) and len(json_response) == 2
    common_test(json_response, expected_box_response)


@pytest.mark.asyncio
async def test_ocr_poly(test_app_asyncio, mock_detection_image, mock_ocr_response):
    headers = {
        "accept": "application/json",
    }
    params = {"det_arch": "db_resnet50", "reco_arch": "crnn_vgg16_bn", "assume_straight_pages": False}
    files = [
        ("files", ("test.jpg", mock_detection_image, "image/jpeg")),
        ("files", ("test2.jpg", mock_detection_image, "image/jpeg")),
    ]
    response = await test_app_asyncio.post("/ocr", params=params, files=files, headers=headers)
    assert response.status_code == 200
    json_response = response.json()

    expected_poly_response = mock_ocr_response["poly"]
    assert isinstance(json_response, list) and len(json_response) == 2
    common_test(json_response, expected_poly_response)


@pytest.mark.asyncio
async def test_ocr_invalid_file(test_app_asyncio, mock_txt_file):
    headers = {
        "accept": "application/json",
    }
    files = [
        ("files", ("test.txt", mock_txt_file)),
    ]
    response = await test_app_asyncio.post("/ocr", files=files, headers=headers)
    assert response.status_code == 400
