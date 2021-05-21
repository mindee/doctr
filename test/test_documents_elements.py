import numpy as np
from doctr.documents import elements


def _mock_words(geoms=([.1, .1, .1, .1, 0], [.3, .3, .1, .1, 12]), confidence=0.9):
    return [
        elements.Word("hello", confidence, geoms[0]),
        elements.Word("world", confidence, geoms[1])
    ]


def _mock_artefacts(geoms=([.1, .1, .1, .1, 0], [.3, .3, .1, .1, 12]), confidence=0.8):
    return [
        elements.Artefact("qr_code", confidence, geoms[0]),
        elements.Artefact("qr_code", confidence, geoms[1])
    ]


def _mock_lines(geoms=(
    ([.1, .1, .1, .1, 0], [.3, .1, .1, .1, 1]),
    ([.1, .3, .1, .1, 0], [.3, .3, .1, .1, 3])
    )):
    return [
        elements.Line(_mock_words(geoms=geoms[0])),
        elements.Line(_mock_words(geoms=geoms[1])),
    ]


def _mock_blocks(geoms=(
    (([.1, .1, .1, .1, 0], [.3, .1, .1, .1, 1]),
    ([.1, .3, .1, .1, 0], [.3, .3, .1, .1, 3])),
    ([.1, .1, .1, .1, 0], [.3, .3, .1, .1, 12])
    )):
    return [
        elements.Block(
            _mock_lines(geoms=geoms[0]),
            _mock_artefacts(geoms=geoms[1])
        )
    ]


def _mock_pages():
    return [
        elements.Page(_mock_blocks(), 0, (300, 200),
                      {"value": 0., "confidence": 1.}, {"value": "EN", "confidence": 0.8}),
        elements.Page(_mock_blocks(), 1, (500, 1000),
                      {"value": 0.15, "confidence": 0.8}, {"value": "FR", "confidence": 0.7}),
    ]


def test_word():
    word_str = "hello"
    conf = 0.8
    geom = (.1, .1, .1, .1, 0)
    word = elements.Word(word_str, conf, geom)

    # Attribute checks
    assert word.value == word_str
    assert word.confidence == conf
    assert word.geometry == geom

    # Render
    assert word.render() == word_str

    # Export
    assert word.export() == {"value": word_str, "confidence": conf, "geometry": geom}

    # Repr
    assert word.__repr__() == f"Word(value='hello', confidence={conf:.2})"


def test_line():
    geom = ([.1, .1, .1, .1, 0], [.3, .3, .1, .1, 12])
    words = _mock_words(geoms=geom)
    line = elements.Line(words)

    # Attribute checks
    assert len(line.words) == len(words)
    assert all(isinstance(w, elements.Word) for w in line.words)
    assert line.geometry == geom

    # Render
    assert line.render() == "hello world"

    # Export
    assert line.export() == {"words": [w.export() for w in words], "geometry": geom}

    # Repr
    words_str = ' ' * 4 + ',\n    '.join(repr(word) for word in words) + ','
    assert line.__repr__() == f"Line(\n  (words): [\n{words_str}\n  ]\n)"

    # Ensure that words repr does't span on several lines when there are none
    assert repr(elements.Line([], ((0, 0), (1, 1)))) == "Line(\n  (words): []\n)"


def test_artefact():
    artefact_type = "qr_code"
    conf = 0.8
    geom = (.1, .1, .1, .1, 0)
    artefact = elements.Artefact(artefact_type, conf, geom)

    # Attribute checks
    assert artefact.type == artefact_type
    assert artefact.confidence == conf
    assert artefact.geometry == geom

    # Render
    assert artefact.render() == "[QR_CODE]"

    # Export
    assert artefact.export() == {"type": artefact_type, "confidence": conf, "geometry": geom}

    # Repr
    assert artefact.__repr__() == f"Artefact(type='{artefact_type}', confidence={conf:.2})"


def test_block():
    lines = _mock_lines()
    artefacts = _mock_artefacts()
    block = elements.Block(lines, artefacts)

    # Attribute checks
    assert len(block.lines) == len(lines)
    assert len(block.artefacts) == len(artefacts)
    assert all(isinstance(w, elements.Line) for w in block.lines)
    assert all(isinstance(a, elements.Artefact) for a in block.artefacts)
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
    page = elements.Page(blocks, page_idx, page_size, orientation, language)

    # Attribute checks
    assert len(page.blocks) == len(blocks)
    assert all(isinstance(b, elements.Block) for b in page.blocks)
    assert page.page_idx == page_idx
    assert page.dimensions == page_size
    assert page.orientation == orientation
    assert page.language == language

    # Render
    assert page.render() == "hello world\nhello world\n\nhello world\nhello world"

    # Export
    assert page.export() == {"blocks": [b.export() for b in blocks], "page_idx": page_idx, "dimensions": page_size,
                             "orientation": orientation, "language": language}

    # Repr
    assert '\n'.join(repr(page).split('\n')[:2]) == f'Page(\n  dimensions={repr(page_size)}'

    # Show
    page.show(np.zeros((256, 256, 3), dtype=np.uint8), block=False)


def test_document():
    pages = _mock_pages()
    doc = elements.Document(pages)

    # Attribute checks
    assert len(doc.pages) == len(pages)
    assert all(isinstance(p, elements.Page) for p in doc.pages)

    # Render
    page_export = "hello world\nhello world\n\nhello world\nhello world"
    assert doc.render() == f"{page_export}\n\n\n\n{page_export}"

    # Export
    assert doc.export() == {"pages": [p.export() for p in pages]}

    # Show
    doc.show([np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(len(pages))], block=False)
