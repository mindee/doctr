import json
from io import BytesIO

import pytest
import requests


@pytest.fixture(scope="session")
def mock_vocab():
    return ('3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|za1ù8,OG€P-kçHëÀÂ2É/ûIJ\'j'
            '(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l')


@pytest.fixture(scope="session")
def mock_pdf_stream():
    url = 'https://arxiv.org/pdf/1911.08947.pdf'
    return requests.get(url).content


@pytest.fixture(scope="session")
def mock_pdf(mock_pdf_stream, tmpdir_factory):
    file = BytesIO(mock_pdf_stream)
    fn = tmpdir_factory.mktemp("data").join("mock_pdf_file.pdf")
    with open(fn, 'wb') as f:
        f.write(file.getbuffer())
    return str(fn)


@pytest.fixture(scope="session")
def mock_image_stream():
    url = "https://miro.medium.com/max/3349/1*mk1-6aYaf_Bes1E3Imhc0A.jpeg"
    return requests.get(url).content


@pytest.fixture(scope="session")
def mock_image_path(mock_image_stream, tmpdir_factory):
    file = BytesIO(mock_image_stream)
    folder = tmpdir_factory.mktemp("images")
    fn = folder.join("mock_image_file.jpeg")
    with open(fn, 'wb') as f:
        f.write(file.getbuffer())
    return str(fn)


@pytest.fixture(scope="session")
def mock_image_folder(mock_image_stream, tmpdir_factory):
    file = BytesIO(mock_image_stream)
    folder = tmpdir_factory.mktemp("images")
    for i in range(5):
        fn = folder.join("mock_image_file_" + str(i) + ".jpeg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())
    return str(folder)


@pytest.fixture(scope="session")
def mock_detection_label(tmpdir_factory):
    folder = tmpdir_factory.mktemp("labels")
    labels = {}
    for idx in range(5):
        labels[f"mock_image_file_{idx}.jpeg"] = {
            "img_dimensions": (800, 600),
            "img_hash": "dummy_hash",
            "polygons": [
                [[1, 2], [1, 3], [2, 1], [2, 3]],
                [[10, 20], [10, 30], [20, 10], [20, 30]],
                [[3, 2], [3, 3], [4, 1], [4, 3]],
                [[30, 20], [30, 30], [40, 10], [40, 30]],
            ],
        }

    labels_path = folder.join('labels.json')
    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    return str(labels_path)


@pytest.fixture(scope="session")
def mock_recognition_label(tmpdir_factory):
    label_file = tmpdir_factory.mktemp("labels").join("labels.json")
    label = {
        "mock_image_file_0.jpeg": "I",
        "mock_image_file_1.jpeg": "am",
        "mock_image_file_2.jpeg": "a",
        "mock_image_file_3.jpeg": "jedi",
        "mock_image_file_4.jpeg": "!",
    }
    with open(label_file, 'w') as f:
        json.dump(label, f)
    return str(label_file)


@pytest.fixture(scope="session")
def mock_ocrdataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp("dataset")
    label_file = root.join("labels.json")
    label = {
        "mock_image_file_0.jpg": {
            "typed_words": [
                {'value': 'I', 'geometry': (.2, .2, .1, .1, 0)},
                {'value': 'am', 'geometry': (.5, .5, .1, .1, 0)},
            ]
        },
        "mock_image_file_1.jpg": {
            "typed_words": [
                {'value': 'a', 'geometry': (.2, .2, .1, .1, 0)},
                {'value': 'jedi', 'geometry': (.5, .5, .1, .1, 0)},
            ]
        },
        "mock_image_file_2.jpg": {
            "typed_words": [
                {'value': '!', 'geometry': (.2, .2, .1, .1, 0)},
            ]
        }
    }
    with open(label_file, 'w') as f:
        json.dump(label, f)

    file = BytesIO(mock_image_stream)
    image_folder = tmpdir_factory.mktemp("images")
    for i in range(3):
        fn = image_folder.join(f"mock_image_file_{i}.jpg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())

    return str(image_folder), str(label_file)
