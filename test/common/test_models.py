import pytest
import numpy as np
from io import BytesIO
import requests
import cv2

from doctr.documents import reader
from doctr import models
from doctr.documents import Document, DocumentFile


def test_extract_crops(mock_pdf):  # noqa: F811
    doc_img = DocumentFile.from_pdf(mock_pdf).as_images()[0]
    num_crops = 2
    rel_boxes = np.array([[idx / num_crops, idx / num_crops, (idx + 1) / num_crops, (idx + 1) / num_crops]
                          for idx in range(num_crops)], dtype=np.float32)
    abs_boxes = np.array([[int(idx * doc_img.shape[1] / num_crops),
                           int(idx * doc_img.shape[0]) / num_crops,
                           int((idx + 1) * doc_img.shape[1] / num_crops),
                           int((idx + 1) * doc_img.shape[0] / num_crops)]
                          for idx in range(num_crops)], dtype=np.float32)

    with pytest.raises(AssertionError):
        models.extract_crops(doc_img, np.zeros((1, 5)))

    for boxes in (rel_boxes, abs_boxes):
        croped_imgs = models.extract_crops(doc_img, boxes)
        # Number of crops
        assert len(croped_imgs) == num_crops
        # Data type and shape
        assert all(isinstance(crop, np.ndarray) for crop in croped_imgs)
        assert all(crop.ndim == 3 for crop in croped_imgs)

    # Identity
    assert np.all(doc_img == models.extract_crops(doc_img, np.array([[0, 0, 1, 1]], dtype=np.float32))[0])

    # No box
    assert models.extract_crops(doc_img, np.zeros((0, 4))) == []


def test_extract_rcrops(mock_pdf):  # noqa: F811
    doc_img = DocumentFile.from_pdf(mock_pdf).as_images()[0]
    num_crops = 2
    rel_boxes = np.array([[idx / num_crops + .1, idx / num_crops + .1, .1, .1, 0]
                          for idx in range(num_crops)], dtype=np.float32)
    abs_boxes = np.array([[int((idx / num_crops + .1) * doc_img.shape[1]),
                           int((idx / num_crops + .1) * doc_img.shape[0]),
                           int(.1 * doc_img.shape[1]),
                           int(.1 * doc_img.shape[0]), 0]
                          for idx in range(num_crops)], dtype=np.int)

    with pytest.raises(AssertionError):
        models.extract_crops(doc_img, np.zeros((1, 8)))
    for boxes in (rel_boxes, abs_boxes):
        croped_imgs = models.extract_rcrops(doc_img, boxes)
        # Number of crops
        assert len(croped_imgs) == num_crops
        # Data type and shape
        assert all(isinstance(crop, np.ndarray) for crop in croped_imgs)
        assert all(crop.ndim == 3 for crop in croped_imgs)

    # No box
    assert models.extract_crops(doc_img, np.zeros((0, 5))) == []


def test_documentbuilder():

    words_per_page = 10
    num_pages = 2

    # Don't resolve lines
    doc_builder = models.DocumentBuilder()
    boxes = np.random.rand(words_per_page, 6)
    boxes[:2] *= boxes[2:4]

    out = doc_builder([boxes, boxes], [('hello', 1.0)] * (num_pages * words_per_page), [(100, 200), (100, 200)])
    assert isinstance(out, Document)
    assert len(out.pages) == num_pages
    # 1 Block & 1 line per page
    assert len(out.pages[0].blocks) == 1 and len(out.pages[0].blocks[0].lines) == 1
    assert len(out.pages[0].blocks[0].lines[0].words) == words_per_page

    # Resolve lines
    doc_builder = models.DocumentBuilder(resolve_lines=True, resolve_blocks=True)
    out = doc_builder([boxes, boxes], [('hello', 1.0)] * (num_pages * words_per_page), [(100, 200), (100, 200)])

    # No detection
    boxes = np.zeros((0, 5))
    out = doc_builder([boxes, boxes], [], [(100, 200), (100, 200)])
    assert len(out.pages[0].blocks) == 0

    # Repr
    assert repr(doc_builder) == "DocumentBuilder(resolve_lines=True, resolve_blocks=True, paragraph_break=0.035)"


@pytest.mark.parametrize(
    "input_boxes, sorted_idxs",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # diagonal
        [[[0, 0.5, 0.1, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [0, 1, 2]],  # same line, 2p
        [[[0, 0.5, 0.1, 0.6], [0.2, 0.49, 0.35, 0.59], [0.8, 0.52, 0.9, 0.63]], [0, 1, 2]],  # ~same line
        [[[0, 0.3, 0.4, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
    ],
)
def test_sort_boxes(input_boxes, sorted_idxs):

    doc_builder = models.DocumentBuilder()
    assert doc_builder._sort_boxes(np.asarray(input_boxes)).tolist() == sorted_idxs


@pytest.mark.parametrize(
    "input_boxes, lines",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [[2], [1], [0]]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [[2], [1], [0]]],  # diagonal
        [[[0, 0.5, 0.14, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [[0, 1], [2]]],  # same line, 2p
        [[[0, 0.5, 0.18, 0.6], [0.2, 0.48, 0.35, 0.58], [0.8, 0.52, 0.9, 0.63]], [[0, 1], [2]]],  # ~same line
        [[[0, 0.3, 0.48, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [[0, 1], [2]]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [[0], [1], [2]]],  # 2 lines
    ],
)
def test_resolve_lines(input_boxes, lines):

    doc_builder = models.DocumentBuilder()
    assert doc_builder._resolve_lines(np.asarray(input_boxes)) == lines


@pytest.fixture(scope="function")
def mock_image(tmpdir_factory):
    url = 'https://github.com/mindee/doctr/releases/download/v0.2.1/bitmap30.png'
    file = BytesIO(requests.get(url).content)
    tmp_path = str(tmpdir_factory.mktemp("data").join("mock_bitmap.jpg"))
    with open(tmp_path, 'wb') as f:
        f.write(file.getbuffer())
    image = reader.read_img(tmp_path)
    return image


@pytest.fixture(scope="function")
def mock_bitmap(mock_image):
    bitmap = np.squeeze(cv2.cvtColor(mock_image, cv2.COLOR_BGR2GRAY) / 255.)
    return bitmap


def test_get_bitmap_angle(mock_bitmap):
    angle = models.get_bitmap_angle(mock_bitmap)
    assert abs(angle - 30.) < 1.


def test_estimate_orientation(mock_image):
    angle = models.estimate_orientation(mock_image)
    assert abs(angle - 30.) < 1.


def test_rotate_page(mock_bitmap):
    rotated = models.rotate_page(mock_bitmap, -30.)
    assert abs(models.get_bitmap_angle(rotated) - 0.) < 1.
