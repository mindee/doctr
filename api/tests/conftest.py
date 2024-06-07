import pytest_asyncio
import requests
from httpx import AsyncClient

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
    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as ac:
        yield ac  # testing happens here


@pytest_asyncio.fixture(scope="function")
def mock_detection_response():
    return {
        "box": {
            "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
            "geometries": [
                [0.8176307908857315, 0.1787109375, 0.9101580212741838, 0.2080078125],
                [0.7471996155154171, 0.1796875, 0.8272978149561669, 0.20703125],
            ],
        },
        "poly": {
            "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
            "geometries": [
                [
                    0.9063061475753784,
                    0.17740710079669952,
                    0.9078840017318726,
                    0.20474515855312347,
                    0.8173396587371826,
                    0.20735852420330048,
                    0.8157618045806885,
                    0.18002046644687653,
                ],
                [
                    0.8233299851417542,
                    0.17740298807621002,
                    0.8250390291213989,
                    0.2027825564146042,
                    0.7470247745513916,
                    0.20540954172611237,
                    0.7453157305717468,
                    0.1800299733877182,
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
                            "value": "Hello",
                            "geometry": [0.7471996155154171, 0.1796875, 0.8272978149561669, 0.20703125],
                            "objectness_score": 0.39,
                            "confidence": 1,
                            "crop_orientation": {"value": 0, "confidence": None},
                        },
                        {
                            "value": "world!",
                            "geometry": [0.8176307908857315, 0.1787109375, 0.9101580212741838, 0.2080078125],
                            "objectness_score": 0.39,
                            "confidence": 1,
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
                            "value": "Hello",
                            "geometry": [
                                0.7453157305717468,
                                0.1800299733877182,
                                0.8233299851417542,
                                0.17740298807621002,
                                0.8250390291213989,
                                0.2027825564146042,
                                0.7470247745513916,
                                0.20540954172611237,
                            ],
                            "objectness_score": 0.5,
                            "confidence": 0.99,
                            "crop_orientation": {"value": 0, "confidence": 1},
                        },
                        {
                            "value": "world!",
                            "geometry": [
                                0.8157618045806885,
                                0.18002046644687653,
                                0.9063061475753784,
                                0.17740710079669952,
                                0.9078840017318726,
                                0.20474515855312347,
                                0.8173396587371826,
                                0.20735852420330048,
                            ],
                            "objectness_score": 0.5,
                            "confidence": 1,
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
                            "geometry": [0.7471996155154171, 0.1787109375, 0.9101580212741838, 0.2080078125],
                            "objectness_score": 0.39,
                            "lines": [
                                {
                                    "geometry": [0.7471996155154171, 0.1787109375, 0.9101580212741838, 0.2080078125],
                                    "objectness_score": 0.39,
                                    "words": [
                                        {
                                            "value": "Hello",
                                            "geometry": [0.7471996155154171, 0.1796875, 0.8272978149561669, 0.20703125],
                                            "objectness_score": 0.39,
                                            "confidence": 1,
                                            "crop_orientation": {"value": 0, "confidence": None},
                                        },
                                        {
                                            "value": "world!",
                                            "geometry": [
                                                0.8176307908857315,
                                                0.1787109375,
                                                0.9101580212741838,
                                                0.2080078125,
                                            ],
                                            "objectness_score": 0.39,
                                            "confidence": 1,
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
                                0.7451040148735046,
                                0.17927837371826172,
                                0.9062581658363342,
                                0.17407986521720886,
                                0.9072266221046448,
                                0.2041015625,
                                0.7460724711418152,
                                0.20930007100105286,
                            ],
                            "objectness_score": 0.5,
                            "lines": [
                                {
                                    "geometry": [
                                        0.7451040148735046,
                                        0.17927837371826172,
                                        0.9062581658363342,
                                        0.17407986521720886,
                                        0.9072266221046448,
                                        0.2041015625,
                                        0.7460724711418152,
                                        0.20930007100105286,
                                    ],
                                    "objectness_score": 0.5,
                                    "words": [
                                        {
                                            "value": "Hello",
                                            "geometry": [
                                                0.7453157305717468,
                                                0.1800299733877182,
                                                0.8233299851417542,
                                                0.17740298807621002,
                                                0.8250390291213989,
                                                0.2027825564146042,
                                                0.7470247745513916,
                                                0.20540954172611237,
                                            ],
                                            "objectness_score": 0.5,
                                            "confidence": 0.99,
                                            "crop_orientation": {"value": 0, "confidence": 1},
                                        },
                                        {
                                            "value": "world!",
                                            "geometry": [
                                                0.8157618045806885,
                                                0.18002046644687653,
                                                0.9063061475753784,
                                                0.17740710079669952,
                                                0.9078840017318726,
                                                0.20474515855312347,
                                                0.8173396587371826,
                                                0.20735852420330048,
                                            ],
                                            "objectness_score": 0.5,
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
