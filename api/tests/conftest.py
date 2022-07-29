# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest_asyncio
import requests
from httpx import AsyncClient

from app.main import app


@pytest_asyncio.fixture(scope="session")
def mock_recognition_image(tmpdir_factory):
    url = 'https://user-images.githubusercontent.com/76527547/117133599-c073fa00-ada4-11eb-831b-412de4d28341.jpeg'
    return requests.get(url).content


@pytest_asyncio.fixture(scope="session")
def mock_detection_image(tmpdir_factory):
    url = 'https://user-images.githubusercontent.com/76527547/117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg'
    return requests.get(url).content


@pytest_asyncio.fixture(scope="function")
async def test_app_asyncio():
    # for httpx>=20, follow_redirects=True (cf. https://github.com/encode/httpx/releases/tag/0.20.0)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac  # testing happens here
