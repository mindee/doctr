# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest


@pytest.mark.asyncio
async def test_text_recognition(test_app_asyncio, mock_recognition_image):

    response = await test_app_asyncio.post("/recognition", files={'file': mock_recognition_image})
    assert response.status_code == 200

    assert response.json() == {"value": "invite"}
