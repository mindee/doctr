import pytest


@pytest.mark.asyncio
async def test_text_recognition(test_app_asyncio, mock_recognition_image, mock_txt_file):
    headers = {
        "accept": "application/json",
    }
    params = {"reco_arch": "crnn_vgg16_bn"}
    files = [
        ("files", ("test.jpg", mock_recognition_image, "image/jpeg")),
        ("files", ("test2.jpg", mock_recognition_image, "image/jpeg")),
    ]
    response = await test_app_asyncio.post("/recognition", params=params, files=files, headers=headers)
    assert response.status_code == 200
    json_response = response.json()
    assert isinstance(json_response, list) and len(json_response) == 2
    for item in json_response:
        assert isinstance(item["name"], str)
        assert isinstance(item["value"], str) and item["value"] == "invite"
        assert isinstance(item["confidence"], (int, float)) and item["confidence"] >= 0.8

    headers = {
        "accept": "application/json",
    }
    files = [
        ("files", ("test.txt", mock_txt_file)),
    ]
    response = await test_app_asyncio.post("/recognition", files=files, headers=headers)
    assert response.status_code == 400
