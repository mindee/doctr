import pytest


@pytest.mark.asyncio
async def test_text_recognition(test_app_asyncio, mock_recognition_image, mock_txt_file):
    response = await test_app_asyncio.post("/recognition", files={"files": [mock_recognition_image] * 2})
    assert response.status_code == 200
    assert response.json() == [{"value": "invite"}, {"value": "invite"}]

    response = await test_app_asyncio.post("/recognition", files={"files": [mock_txt_file]})
    assert response.status_code == 400
