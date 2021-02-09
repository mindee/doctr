import requests
import pytest
import fitz
import numpy as np
from io import BytesIO

from doctr import documents


def _mock_words(size=(1., 1.), offset=(0, 0), confidence=0.9):
    return [
        documents.Word("hello", confidence, [
            (offset[0], offset[1]),
            (size[0] / 2 + offset[0], size[1] / 2 + offset[1])
        ]),
        documents.Word("world", confidence, [
            (size[0] / 2 + offset[0], size[1] / 2 + offset[1]),
            (size[0] + offset[0], size[1] + offset[1])
        ])
    ]


def _mock_artefacts(size=(1, 1), offset=(0, 0), confidence=0.8):
    sub_size = (size[0] / 2, size[1] / 2)
    return [
        documents.Artefact("qr_code", confidence, [
            (offset[0], offset[1]),
            (sub_size[0] + offset[0], sub_size[1] + offset[1])
        ]),
        documents.Artefact("qr_code", confidence, [
            (sub_size[0] + offset[0], sub_size[1] + offset[1]),
            (size[0] + offset[0], size[1] + offset[1])
        ]),
    ]


def _mock_lines(size=(1, 1), offset=(0, 0)):
    sub_size = (size[0] / 2, size[1] / 2)
    return [
        documents.Line(_mock_words(size=sub_size, offset=offset)),
        documents.Line(_mock_words(size=sub_size, offset=(offset[0] + sub_size[0], offset[1] + sub_size[1]))),
    ]


def _mock_blocks(size=(1, 1), offset=(0, 0)):
    sub_size = (size[0] / 4, size[1] / 4)
    return [
        documents.Block(
            _mock_lines(size=sub_size, offset=offset),
            _mock_artefacts(size=sub_size, offset=(offset[0] + sub_size[0], offset[1] + sub_size[1]))
        ),
        documents.Block(
            _mock_lines(size=sub_size, offset=(offset[0] + 2 * sub_size[0], offset[1] + 2 * sub_size[1])),
            _mock_artefacts(size=sub_size, offset=(offset[0] + 3 * sub_size[0], offset[1] + 3 * sub_size[1])),
        ),
    ]


def _mock_pages(block_size=(1, 1), block_offset=(0, 0)):
    return [
        documents.Page(_mock_blocks(block_size, block_offset), 0, (300, 200),
                       {"value": 0., "confidence": 1.}, {"value": "EN", "confidence": 0.8}),
        documents.Page(_mock_blocks(block_size, block_offset), 1, (500, 1000),
                       {"value": 0.15, "confidence": 0.8}, {"value": "FR", "confidence": 0.7}),
    ]


def test_word():
    word_str = "hello"
    conf = 0.8
    geom = ((0, 0), (1, 1))
    word = documents.Word(word_str, conf, geom)

    # Attribute checks
    assert word.value == word_str
    assert word.confidence == conf
    assert word.geometry == geom

    # Render
    assert word.render() == word_str

    # Export
    assert word.export() == {"value": word_str, "confidence": conf, "geometry": geom}


def test_line():
    geom = ((0, 0), (0.5, 0.5))
    words = _mock_words(size=geom[1], offset=geom[0])
    line = documents.Line(words)

    # Attribute checks
    assert len(line.words) == len(words)
    assert all(isinstance(w, documents.Word) for w in line.words)
    assert line.geometry == geom

    # Render
    assert line.render() == "hello world"

    # Export
    assert line.export() == {"words": [w.export() for w in words], "geometry": geom}


def test_artefact():
    artefact_type = "qr_code"
    conf = 0.8
    geom = ((0, 0), (1, 1))
    artefact = documents.Artefact(artefact_type, conf, geom)

    # Attribute checks
    assert artefact.type == artefact_type
    assert artefact.confidence == conf
    assert artefact.geometry == geom

    # Render
    assert artefact.render() == "[QR_CODE]"

    # Export
    assert artefact.export() == {"type": artefact_type, "confidence": conf, "geometry": geom}


def test_block():
    geom = ((0, 0), (1, 1))
    sub_size = (geom[1][0] / 2, geom[1][0] / 2)
    lines = _mock_lines(size=sub_size, offset=geom[0])
    artefacts = _mock_artefacts(size=sub_size, offset=sub_size)
    block = documents.Block(lines, artefacts)

    # Attribute checks
    assert len(block.lines) == len(lines)
    assert len(block.artefacts) == len(artefacts)
    assert all(isinstance(w, documents.Line) for w in block.lines)
    assert all(isinstance(a, documents.Artefact) for a in block.artefacts)
    assert block.geometry == geom

    # Render
    assert block.render() == "hello world\nhello world"

    # Export
    assert block.export() == {"lines": [line.export() for line in lines],
                              "artefacts": [artefact.export() for artefact in artefacts], "geometry": geom}


def test_page():
    page_idx = 0
    page_size = (300, 200)
    orientation = {"value": 0., "confidence": 0.}
    language = {"value": "EN", "confidence": 0.8}
    blocks = _mock_blocks()
    page = documents.Page(blocks, page_idx, page_size, orientation, language)

    # Attribute checks
    assert len(page.blocks) == len(blocks)
    assert all(isinstance(b, documents.Block) for b in page.blocks)
    assert page.page_idx == page_idx
    assert page.dimensions == page_size
    assert page.orientation == orientation
    assert page.language == language

    # Render
    assert page.render() == "hello world\nhello world\n\nhello world\nhello world"

    # Export
    assert page.export() == {"blocks": [b.export() for b in blocks], "page_idx": page_idx, "dimensions": page_size,
                             "orientation": orientation, "language": language}


def test_document():
    pages = _mock_pages()
    doc = documents.Document(pages)

    # Attribute checks
    assert len(doc.pages) == len(pages)
    assert all(isinstance(p, documents.Page) for p in doc.pages)

    # Render
    page_export = "hello world\nhello world\n\nhello world\nhello world"
    assert doc.render() == f"{page_export}\n\n\n\n{page_export}"

    # Export
    assert doc.export() == {"pages": [p.export() for p in pages]}


@pytest.fixture(scope="session")
def mock_pdf(tmpdir_factory):
    url = 'https://arxiv.org/pdf/1911.08947.pdf'
    file = BytesIO(requests.get(url).content)
    fn = tmpdir_factory.mktemp("data").join("mock_pdf_file.pdf")
    with open(fn, 'wb') as f:
        f.write(file.getbuffer())
    return str(fn)


def test_convert_page_to_numpy(mock_pdf):
    pdf = fitz.open(mock_pdf)
    # Check correct read
    rgb_page = documents.reader.convert_page_to_numpy(pdf[0])
    assert isinstance(rgb_page, np.ndarray)
    assert rgb_page.shape == (792, 612, 3)

    # Check channel order
    bgr_page = documents.reader.convert_page_to_numpy(pdf[0], rgb_output=False)
    assert np.all(bgr_page == rgb_page[..., ::-1])

    # Check rescaling
    resized_page = documents.reader.convert_page_to_numpy(pdf[0], output_size=(396, 306))
    assert resized_page.shape == (396, 306, 3)


def test_read_pdf(mock_pdf):

    doc_tensors = documents.reader.read_pdf(mock_pdf)

    # 1 doc of 8 pages
    assert(len(doc_tensors) == 8)
    assert all(isinstance(page, np.ndarray) for page in doc_tensors)
    assert all(page.dtype == np.uint8 for page in doc_tensors)


def test_read_img(tmpdir_factory, mock_pdf):

    url = 'https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg'
    file = BytesIO(requests.get(url).content)
    tmp_path = str(tmpdir_factory.mktemp("data").join("mock_img_file.jpg"))
    with open(tmp_path, 'wb') as f:
        f.write(file.getbuffer())

    page = documents.reader.read_img(tmp_path)

    # Data type
    assert isinstance(page, np.ndarray)
    assert page.dtype == np.uint8
    # Shape
    assert page.shape == (606, 517, 3)

    # RGB
    bgr_page = documents.reader.read_img(tmp_path, rgb_output=False)
    assert np.all(page == bgr_page[..., ::-1])

    # Resize
    target_size = (200, 150)
    resized_page = documents.reader.read_img(tmp_path, target_size)
    assert resized_page.shape[:2] == target_size

    # Non-existing file
    with pytest.raises(FileNotFoundError):
        documents.reader.read_img("my_imaginary_file.jpg")
    # Invalid image
    with pytest.raises(ValueError):
        documents.reader.read_img(mock_pdf)
