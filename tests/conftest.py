import json
import shutil
from io import BytesIO

import hdf5storage
import numpy as np
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


@pytest.fixture(scope="session")
def mock_ic13(tmpdir_factory, mock_image_stream):
    file = BytesIO(mock_image_stream)
    image_folder = tmpdir_factory.mktemp("images")
    label_folder = tmpdir_factory.mktemp("labels")
    labels = ["1309, 2240, 1440, 2341, 'I'\n",
              "800, 2240, 1440, 2341, 'am'\n",
              "500, 2240, 1440, 2341, 'a'\n",
              "900, 2240, 1440, 2341, 'jedi'\n",
              "400, 2240, 1440, 2341, '!'"]
    for i in range(5):
        fn_l = label_folder.join(f"gt_mock_image_file_{i}.txt")
        with open(fn_l, 'w') as f:
            f.writelines(labels)
        fn_i = image_folder.join(f"mock_image_file_{i}.jpg")
        with open(fn_i, 'wb') as f:
            f.write(file.getbuffer())
    return str(image_folder), str(label_folder)


@pytest.fixture(scope="session")
def mock_svhn_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    svhn_root = root.mkdir('svhn')
    file = BytesIO(mock_image_stream)
    # ascii image names
    first = np.array([[49], [46], [112], [110], [103]], dtype=np.int16)  # 1.png
    second = np.array([[50], [46], [112], [110], [103]], dtype=np.int16)  # 2.png
    third = np.array([[51], [46], [112], [110], [103]], dtype=np.int16)  # 3.png
    # labels: label is also ascii
    label = {'height': [35, 35, 35, 35], 'label': [1, 1, 3, 7],
             'left': [116, 128, 137, 151], 'top': [27, 29, 29, 26],
             'width': [15, 10, 17, 17]}

    matcontent = {'digitStruct': {'name': [first, second, third], 'bbox': [label, label, label]}}
    # Mock train data
    train_root = svhn_root.mkdir('train')
    hdf5storage.write(matcontent, filename=train_root.join('digitStruct.mat'))
    for i in range(3):
        fn = train_root.join(f'{i+1}.png')
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())
    # Packing data into an archive to simulate the real data set and bypass archive extraction
    shutil.make_archive(svhn_root.join('svhn_train'), 'tar', str(svhn_root))
    return str(root)


@pytest.fixture(scope="session")
def mock_doc_artefacts(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    doc_root = root.mkdir('artefact_detection')
    labels = {
        '0.jpg':
        [
            {'geometry': [0.94375, 0.4013671875, 0.99375, 0.4365234375],
             'label': 'bar_code'},
            {'geometry': [0.03125, 0.6923828125, 0.07875, 0.7294921875],
             'label': 'qr_code'},
            {'geometry': [0.1975, 0.1748046875, 0.39875, 0.2216796875],
             'label': 'bar_code'}
        ],
        '1.jpg': [
            {'geometry': [0.94375, 0.4013671875, 0.99375, 0.4365234375],
             'label': 'bar_code'},
            {'geometry': [0.03125, 0.6923828125, 0.07875, 0.7294921875],
             'label': 'qr_code'},
            {'geometry': [0.1975, 0.1748046875, 0.39875, 0.2216796875],
             'label': 'background'}
        ],
        '2.jpg': [
            {'geometry': [0.94375, 0.4013671875, 0.99375, 0.4365234375],
             'label': 'logo'},
            {'geometry': [0.03125, 0.6923828125, 0.07875, 0.7294921875],
             'label': 'qr_code'},
            {'geometry': [0.1975, 0.1748046875, 0.39875, 0.2216796875],
             'label': 'photo'}
        ]
    }
    train_root = doc_root.mkdir('train')
    label_file = train_root.join("labels.json")

    with open(label_file, 'w') as f:
        json.dump(labels, f)

    image_folder = train_root.mkdir("images")
    file = BytesIO(mock_image_stream)
    for i in range(3):
        fn = image_folder.join(f"{i}.jpg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())
    # Packing data into an archive to simulate the real data set and bypass archive extraction
    shutil.make_archive(doc_root.join('artefact_detection'), 'zip', str(doc_root))
    return str(doc_root)
