import pytest_asyncio
import requests
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture(scope="session")
def mock_recognition_image(tmpdir_factory):
    url = "https://user-images.githubusercontent.com/76527547/117133599-c073fa00-ada4-11eb-831b-412de4d28341.jpeg"
    return requests.get(url).content


@pytest_asyncio.fixture(scope="session")
def mock_detection_image(tmpdir_factory):
    url = "https://user-images.githubusercontent.com/76527547/117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg"
    return requests.get(url).content


@pytest_asyncio.fixture(scope="session")
def mock_txt_file(tmpdir_factory):
    txt_file = tmpdir_factory.mktemp("data").join("mock.txt")
    txt_file.write("mock text")
    return txt_file.read("rb")


@pytest_asyncio.fixture(scope="function")
async def test_app_asyncio():
    # for httpx>=20, follow_redirects=True (cf. https://github.com/encode/httpx/releases/tag/0.20.0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        yield ac  # testing happens here


@pytest_asyncio.fixture(scope="function")
def mock_detection_response():
    return {
        "box": {
            "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
            "geometries": [
                [0.8203927977629988, 0.181640625, 0.9087770178355502, 0.2041015625],
                [0.7471996155154171, 0.1806640625, 0.8245358080788996, 0.2060546875],
            ],
        },
        "poly": {
            "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
            "geometries": [
                [
                    0.8203927977629988,
                    0.181640625,
                    0.906015010958283,
                    0.181640625,
                    0.906015010958283,
                    0.2021484375,
                    0.8203927977629988,
                    0.2021484375,
                ],
                [
                    0.7482568619833604,
                    0.17938309907913208,
                    0.8208542842026056,
                    0.1819499135017395,
                    0.8193355512950555,
                    0.2034294307231903,
                    0.7467381290758103,
                    0.20086261630058289,
                ],
            ],
        },
    }


@pytest_asyncio.fixture(scope="function")
def mock_kie_response():
    return {
        "box": {
            "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
            "orientation": {"value": None, "confidence": None},
            "language": {"value": None, "confidence": None},
            "dimensions": [2339, 1654],
            "predictions": [
                {
                    "class_name": "words",
                    "items": [
                        {
                            "value": "world!",
                            "geometry": [0.8203927977629988, 0.181640625, 0.9087770178355502, 0.2041015625],
                            "objectness_score": 0.46,
                            "confidence": 0.94,
                            "crop_orientation": {"value": 0, "confidence": None},
                        },
                        {
                            "value": "Hello",
                            "geometry": [0.7471996155154171, 0.1806640625, 0.8245358080788996, 0.2060546875],
                            "objectness_score": 0.46,
                            "confidence": 0.66,
                            "crop_orientation": {"value": 0, "confidence": None},
                        },
                    ],
                }
            ],
        },
        "poly": {
            "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
            "orientation": {"value": None, "confidence": None},
            "language": {"value": None, "confidence": None},
            "dimensions": [2339, 1654],
            "predictions": [
                {
                    "class_name": "words",
                    "items": [
                        {
                            "value": "world!",
                            "geometry": [
                                0.8203927977629988,
                                0.181640625,
                                0.906015010958283,
                                0.181640625,
                                0.906015010958283,
                                0.2021484375,
                                0.8203927977629988,
                                0.2021484375,
                            ],
                            "objectness_score": 0.52,
                            "confidence": 1,
                            "crop_orientation": {"value": 0, "confidence": 1},
                        },
                        {
                            "value": "Hello",
                            "geometry": [
                                0.7482568619833604,
                                0.17938309907913208,
                                0.8208542842026056,
                                0.1819499135017395,
                                0.8193355512950555,
                                0.2034294307231903,
                                0.7467381290758103,
                                0.20086261630058289,
                            ],
                            "objectness_score": 0.57,
                            "confidence": 0.65,
                            "crop_orientation": {"value": 0, "confidence": 1},
                        },
                    ],
                }
            ],
        },
    }


@pytest_asyncio.fixture(scope="function")
def mock_ocr_response():
    return {
        "box": {
            "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
            "orientation": {"value": None, "confidence": None},
            "language": {"value": None, "confidence": None},
            "dimensions": [2339, 1654],
            "items": [
                {
                    "blocks": [
                        {
                            "geometry": [0.7471996155154171, 0.1806640625, 0.9087770178355502, 0.2060546875],
                            "objectness_score": 0.46,
                            "lines": [
                                {
                                    "geometry": [0.7471996155154171, 0.1806640625, 0.9087770178355502, 0.2060546875],
                                    "objectness_score": 0.46,
                                    "words": [
                                        {
                                            "value": "Hello",
                                            "geometry": [
                                                0.7471996155154171,
                                                0.1806640625,
                                                0.8245358080788996,
                                                0.2060546875,
                                            ],
                                            "objectness_score": 0.46,
                                            "confidence": 0.66,
                                            "crop_orientation": {"value": 0, "confidence": None},
                                        },
                                        {
                                            "value": "world!",
                                            "geometry": [
                                                0.8203927977629988,
                                                0.181640625,
                                                0.9087770178355502,
                                                0.2041015625,
                                            ],
                                            "objectness_score": 0.46,
                                            "confidence": 0.94,
                                            "crop_orientation": {"value": 0, "confidence": None},
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ],
        },
        "poly": {
            "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
            "orientation": {"value": None, "confidence": None},
            "language": {"value": None, "confidence": None},
            "dimensions": [2339, 1654],
            "items": [
                {
                    "blocks": [
                        {
                            "geometry": [
                                0.7460642457008362,
                                0.2017778754234314,
                                0.7464945912361145,
                                0.17868199944496155,
                                0.9056554436683655,
                                0.18164771795272827,
                                0.9052250981330872,
                                0.20474359393119812,
                            ],
                            "objectness_score": 0.54,
                            "lines": [
                                {
                                    "geometry": [
                                        0.7460642457008362,
                                        0.2017778754234314,
                                        0.7464945912361145,
                                        0.17868199944496155,
                                        0.9056554436683655,
                                        0.18164771795272827,
                                        0.9052250981330872,
                                        0.20474359393119812,
                                    ],
                                    "objectness_score": 0.54,
                                    "words": [
                                        {
                                            "value": "Hello",
                                            "geometry": [
                                                0.7482568619833604,
                                                0.17938309907913208,
                                                0.8208542842026056,
                                                0.1819499135017395,
                                                0.8193355512950555,
                                                0.2034294307231903,
                                                0.7467381290758103,
                                                0.20086261630058289,
                                            ],
                                            "objectness_score": 0.57,
                                            "confidence": 0.65,
                                            "crop_orientation": {"value": 0, "confidence": 1},
                                        },
                                        {
                                            "value": "world!",
                                            "geometry": [
                                                0.8203927977629988,
                                                0.181640625,
                                                0.906015010958283,
                                                0.181640625,
                                                0.906015010958283,
                                                0.2021484375,
                                                0.8203927977629988,
                                                0.2021484375,
                                            ],
                                            "objectness_score": 0.52,
                                            "confidence": 1,
                                            "crop_orientation": {"value": 0, "confidence": 1},
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ],
        },
    }
