import pytest
import json
from io import BytesIO

from doctr import datasets


@pytest.fixture(scope="function")
def mock_ocrdataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp("dataset")
    label_file = root.join("labels.json")
    label = [
        {
            "raw-archive-filepath": "mock_image_file_0.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['I']
        },
        {
            "raw-archive-filepath": "mock_image_file_1.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['am']
        },
        {
            "raw-archive-filepath": "mock_image_file_2.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['a']
        },
        {
            "raw-archive-filepath": "mock_image_file_3.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['jedi']
        },
        {
            "raw-archive-filepath": "mock_image_file_4.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['!']
        }
    ]
    with open(label_file, 'w') as f:
        json.dump(label, f)

    file = BytesIO(mock_image_stream)
    image_folder = tmpdir_factory.mktemp("images")
    for i in range(5):
        fn = image_folder.join(f"mock_image_file_{i}.jpg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())

    return str(image_folder), str(label_file)


def test_ocrdataset(mock_ocrdataset):
    ds = datasets.OCRDataset(*mock_ocrdataset)
    assert len(ds) == 5
