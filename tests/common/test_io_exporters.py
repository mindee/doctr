import json

import numpy as np
import pytest

from doctr.file_utils import CLASS_NAME
from doctr.io import elements
from doctr.io.exporters import (
    AsciiDocExporter,
    HTMLExporter,
    MarkdownExporter,
    TextExporter,
    XMLExporter,
    ordered_line_words,
    page_reading_order,
    to_json_safe,
)


def _word_at(text, x0, y0, x1, y1):
    return elements.Word(text, 0.95, ((x0, y0), (x1, y1)), 0.9, {"value": 0, "confidence": None})


def _line_at(text, x0, y0, x1, y1, rtl=False):
    """Build a line whose words are laid out geometrically (leftmost word = last logical word when rtl)"""
    words = text.split()
    step = (x1 - x0) / max(len(words), 1)
    geo_words = words[::-1] if rtl else words
    return elements.Line([
        _word_at(word, x0 + idx * step, y0, x0 + (idx + 0.9) * step, y1) for idx, word in enumerate(geo_words)
    ])


def _reading_order_page():
    """A page in the default builder configuration (single block) with a title, 2 columns & a footer"""
    lines = [_line_at("A Two Column Study", 0.2, 0.05, 0.8, 0.09)]
    lines += [_line_at(f"left line {idx}", 0.08, 0.14 + 0.05 * idx, 0.46, 0.17 + 0.05 * idx) for idx in range(3)]
    lines += [_line_at(f"right line {idx}", 0.54, 0.14 + 0.05 * idx, 0.92, 0.17 + 0.05 * idx) for idx in range(3)]
    lines += [_line_at("- item one", 0.08, 0.4, 0.46, 0.43), _line_at("Page 3 of 12", 0.4, 0.95, 0.6, 0.97)]
    # Shuffle the lines to make sure the export does not rely on the input order
    lines = [lines[idx] for idx in [5, 0, 8, 2, 4, 7, 1, 6, 3]]
    layout = [
        elements.LayoutElement("Title", 0.99, ((0.15, 0.04), (0.85, 0.1))),
        elements.LayoutElement("Text", 0.98, ((0.06, 0.12), (0.48, 0.32))),
        elements.LayoutElement("Text", 0.98, ((0.52, 0.12), (0.94, 0.32))),
        elements.LayoutElement("List-item", 0.97, ((0.06, 0.38), (0.48, 0.45))),
        elements.LayoutElement("Page-footer", 0.97, ((0.35, 0.94), (0.65, 0.98))),
    ]
    return elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800), layout=layout
    )


def test_page_items_in_reading_order():
    page = _reading_order_page()
    items = page.items_in_reading_order()
    assert all(isinstance(item, elements.Block) for item in items)
    rendered = [item.render(line_break=" ") for item in items]
    assert rendered[0] == "A Two Column Study"
    assert rendered[-1] == "Page 3 of 12"
    assert rendered.index("left line 0 left line 1 left line 2") < rendered.index(
        "right line 0 right line 1 right line 2"
    )
    # Multi-block pages are ordered at the block level
    top = elements.Block([_line_at("first words", 0.1, 0.1, 0.9, 0.15)])
    bottom = elements.Block([_line_at("last words", 0.1, 0.5, 0.9, 0.55)])
    page = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [bottom, top], 0, (1000, 800))
    assert [block.render() for block in page.items_in_reading_order()] == ["first words", "last words"]


def test_page_export_as_markdown():
    page = _reading_order_page()
    markdown = page.export_as_markdown()
    parts = markdown.split("\n\n")
    assert parts[0] == "# A Two Column Study"
    assert parts[1] == "left line 0\nleft line 1\nleft line 2"
    # The list item belongs to the left column, hence it is read before the right column
    assert parts[2] == "- \\- item one"  # list item, with the raw OCR dash escaped
    assert parts[3] == "right line 0\nright line 1\nright line 2"
    assert parts[4] == "Page 3 of 12"
    # Page furniture can be dropped
    assert "Page 3 of 12" not in page.export_as_markdown(include_furniture=False)
    # Markdown structural characters are escaped by default
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block([_line_at("*bold* #tag [link]", 0.1, 0.1, 0.9, 0.15)])],
        0,
        (1000, 800),
    )
    assert page.export_as_markdown() == "\\*bold\\* \\#tag \\[link\\]"
    assert page.export_as_markdown(escape=False) == "*bold* #tag [link]"
    # Empty pages export to an empty string
    assert elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [], 0, (1000, 800)).export_as_markdown() == ""


def test_page_export_as_markdown_rtl():
    # Two columns of Arabic text: the right column is read first, and the words of each line are emitted
    # from the rightmost to the leftmost one
    lines = [
        _line_at("النص في العمود الأيمن", 0.54, 0.1, 0.92, 0.14, rtl=True),
        _line_at("النص في العمود الأيسر", 0.08, 0.1, 0.46, 0.14, rtl=True),
    ]
    page = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800))
    markdown = page.export_as_markdown()
    assert markdown == "النص في العمود الأيمن\n\nالنص في العمود الأيسر"
    # An explicit direction takes precedence over the detection
    assert page.export_as_markdown(direction="ltr").startswith("الأيسر")


def test_page_export_with_tables():
    cells = [
        elements.TableCell("Name", 0.9, ((0.1, 0.55), (0.4, 0.6)), 0, 0, 0, 0),
        elements.TableCell("Qty", 0.9, ((0.4, 0.55), (0.7, 0.6)), 0, 0, 1, 1),
        elements.TableCell("Bolt", 0.9, ((0.1, 0.6), (0.4, 0.65)), 1, 1, 0, 0),
        elements.TableCell("12|3", 0.9, ((0.4, 0.6), (0.7, 0.65)), 1, 1, 1, 1),
    ]
    table = elements.Table(cells, 2, 2, ((0.1, 0.55), (0.7, 0.65)), 0.95)
    lines = [
        _line_at("before the table", 0.1, 0.1, 0.9, 0.14),
        _line_at("after the table", 0.1, 0.7, 0.9, 0.74),
    ]
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800), tables=[table]
    )
    markdown = page.export_as_markdown()
    assert markdown.split("\n\n") == [
        "before the table",
        "| Name | Qty |\n| --- | --- |\n| Bolt | 12\\|3 |",
        "after the table",
    ]
    asciidoc = page.export_as_asciidoc()
    assert "|===\n|Name |Qty\n\n|Bolt |12\\|3\n|===" in asciidoc
    assert asciidoc.index("before the table") < asciidoc.index("|===") < asciidoc.index("after the table")


def test_page_export_as_asciidoc():
    page = _reading_order_page()
    asciidoc = page.export_as_asciidoc()
    parts = asciidoc.split("\n\n")
    assert parts[0] == "== A Two Column Study"
    assert parts[2] == "* {empty}- item one"
    assert "Page 3 of 12" not in page.export_as_asciidoc(include_furniture=False)


def test_page_export_as():
    page = _reading_order_page()
    assert page.export_as("markdown") == page.export_as("md") == page.export_as_markdown()
    assert page.export_as("adoc") == page.export_as_asciidoc()
    assert page.export_as("text") == page.render()
    assert page.export_as("json") == page.export()
    assert isinstance(page.export_as("xml")[0], bytes)
    assert page.export_as("markdown", include_furniture=False) == page.export_as_markdown(include_furniture=False)
    with pytest.raises(ValueError):
        page.export_as("yaml")


def test_document_export_as_markdown():
    pages = [
        elements.Page(
            np.zeros((10, 10, 3), dtype=np.uint8),
            [elements.Block([_line_at(f"page {idx} content", 0.1, 0.1, 0.9, 0.15)])],
            idx,
            (1000, 800),
        )
        for idx in range(2)
    ]
    doc = elements.Document(pages)
    assert doc.export_as_markdown() == "page 0 content\n\n---\n\npage 1 content"
    assert doc.export_as_asciidoc() == "page 0 content\n\n<<<\n\npage 1 content"
    assert doc.export_as_markdown(page_break="\n\n") == "page 0 content\n\npage 1 content"
    assert doc.export_as("markdown") == doc.export_as_markdown()
    assert doc.export_as("text") == doc.render()
    assert doc.export_as("json") == doc.export()
    assert len(doc.export_as("xml")) == 2
    with pytest.raises(ValueError):
        doc.export_as("pdf")


def test_kie_page_export_as_markdown():
    predictions = {
        CLASS_NAME: [
            elements.Prediction("second", 0.9, ((0.1, 0.5), (0.9, 0.6)), 0.9, {"value": 0, "confidence": None}),
            elements.Prediction("first", 0.9, ((0.1, 0.1), (0.9, 0.2)), 0.9, {"value": 0, "confidence": None}),
        ]
    }
    page = elements.KIEPage(np.zeros((10, 10, 3), dtype=np.uint8), predictions, 0, (1000, 800))
    assert page.export_as_markdown() == f"**{CLASS_NAME}**\n\n- first\n- second"
    assert page.export_as_asciidoc() == f"*{CLASS_NAME}*\n\n* first\n* second"
    assert page.export_as("md") == page.export_as_markdown()
    with pytest.raises(ValueError):
        page.export_as("yaml")
    doc = elements.KIEDocument([page])
    assert doc.export_as_markdown() == page.export_as_markdown()


def test_page_export_as_markdown_list_items():
    # Three separate single-line list items, each covered by its own List-item region -> three bullets
    lines = [_line_at(f"item number {idx}", 0.1, 0.1 + 0.1 * idx, 0.5, 0.13 + 0.1 * idx) for idx in range(3)]
    layout = [
        elements.LayoutElement("List-item", 0.9, ((0.08, 0.09 + 0.1 * idx), (0.52, 0.14 + 0.1 * idx)))
        for idx in range(3)
    ]
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800), layout=layout
    )
    assert page.export_as_markdown() == "- item number 0\n- item number 1\n- item number 2"
    assert page.export_as_asciidoc() == "* item number 0\n* item number 1\n* item number 2"
    # each list item is its own block in reading order
    items = page.items_in_reading_order()
    assert len(items) == 3
    assert all(len(item.lines) == 1 for item in items)


def test_page_export_as_markdown_wrapped_list_item():
    # A single list item wrapped over three visual lines (one region) must render as ONE bullet, while a
    # second item (another region) is a second bullet.
    lines = [
        _line_at("first item wrapping over", 0.1, 0.10, 0.9, 0.13),
        _line_at("several visual lines here", 0.1, 0.14, 0.9, 0.17),
        _line_at("until it finally ends", 0.1, 0.18, 0.6, 0.21),
        _line_at("second short item", 0.1, 0.26, 0.5, 0.29),
    ]
    layout = [
        elements.LayoutElement("List-item", 0.9, ((0.08, 0.09), (0.92, 0.22))),
        elements.LayoutElement("List-item", 0.9, ((0.08, 0.25), (0.52, 0.30))),
    ]
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800), layout=layout
    )
    assert page.export_as_markdown() == (
        "- first item wrapping over several visual lines here until it finally ends\n- second short item"
    )
    items = page.items_in_reading_order()
    assert len(items) == 2
    assert len(items[0].lines) == 3 and len(items[1].lines) == 1


def test_page_export_as_markdown_rotated_page():
    height, width = 1000, 800

    def _rot_line(text, x0, y0, x1, y1, deg):
        angle = np.deg2rad(deg)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center = np.array([width / 2, height / 2])
        tokens = text.split()
        step = (x1 - x0) / len(tokens)
        words = []
        for idx, token in enumerate(tokens):
            pts = np.array([
                [(x0 + idx * step) * width, y0 * height],
                [(x0 + (idx + 0.9) * step) * width, y0 * height],
                [(x0 + (idx + 0.9) * step) * width, y1 * height],
                [(x0 + idx * step) * width, y1 * height],
            ])
            pts = ((pts - center) @ rot.T + center) / [width, height]
            words.append(
                elements.Word(token, 0.9, tuple(tuple(pt) for pt in pts), 0.9, {"value": 0, "confidence": None})
            )
        return elements.Line(words)

    layout = [
        ("big page title", 0.1, 0.05, 0.9, 0.09),
        ("left one", 0.1, 0.15, 0.45, 0.19),
        ("left two", 0.1, 0.21, 0.45, 0.25),
        ("left three", 0.1, 0.27, 0.45, 0.31),
        ("right one", 0.55, 0.15, 0.9, 0.19),
        ("right two", 0.55, 0.21, 0.9, 0.25),
        ("right three", 0.55, 0.27, 0.9, 0.31),
    ]
    expected = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_rot_line(*args, 0) for args in layout])],
        0,
        (height, width),
    ).export_as_markdown()
    assert expected.split()[:3] == ["big", "page", "title"]
    for deg in (15, 25):
        page = elements.Page(
            np.zeros((10, 10, 3), dtype=np.uint8),
            [elements.Block(lines=[_rot_line(*args, deg) for args in layout])],
            0,
            (height, width),
        )
        assert page.export_as_markdown().split() == expected.split()


def test_page_export_as_markdown_rotated_landscape_page():
    height, width = 800, 1200

    def _rot_line(text, x0, y0, x1, y1, deg):
        angle = np.deg2rad(deg)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center = np.array([width / 2, height / 2])
        tokens = text.split()
        step = (x1 - x0) / len(tokens)
        words = []
        for idx, token in enumerate(tokens):
            pts = np.array([
                [(x0 + idx * step) * width, y0 * height],
                [(x0 + (idx + 0.9) * step) * width, y0 * height],
                [(x0 + (idx + 0.9) * step) * width, y1 * height],
                [(x0 + idx * step) * width, y1 * height],
            ])
            pts = ((pts - center) @ rot.T + center) / [width, height]
            words.append(
                elements.Word(token, 0.9, tuple(tuple(pt) for pt in pts), 0.9, {"value": 0, "confidence": None})
            )
        return elements.Line(words)

    layout = [
        ("big page title", 0.1, 0.05, 0.9, 0.09),
        ("left one", 0.1, 0.15, 0.45, 0.19),
        ("left two", 0.1, 0.21, 0.45, 0.25),
        ("right one", 0.55, 0.15, 0.9, 0.19),
        ("right two", 0.55, 0.21, 0.9, 0.25),
    ]
    expected = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_rot_line(*args, 0) for args in layout])],
        0,
        (height, width),
    ).export_as_markdown()
    for deg in (-35, 35):
        page = elements.Page(
            np.zeros((10, 10, 3), dtype=np.uint8),
            [elements.Block(lines=[_rot_line(*args, deg) for args in layout])],
            0,
            (height, width),
        )
        assert page.export_as_markdown().split() == expected.split()


def test_exporter_classes_direct_use():
    # The exporter classes are usable directly, and export_document dispatches per page type
    page = _reading_order_page()
    md = MarkdownExporter()
    adoc = AsciiDocExporter()
    assert md.export_page(page) == page.export_as_markdown()
    assert adoc.export_page(page) == page.export_as_asciidoc()

    class _Doc:
        pages = [page, page]

    assert md.export_document(_Doc()) == "\n\n---\n\n".join([page.export_as_markdown()] * 2)
    assert adoc.export_document(_Doc(), page_break="\n\n") == "\n\n".join([page.export_as_asciidoc()] * 2)
    # page_reading_order returns (items, labels, direction)
    items, labels, direction = page_reading_order(page)
    assert len(items) == len(labels) and direction == "ltr"


def test_page_export_as_html():
    page = _reading_order_page()
    html = page.export_as_html()
    # title heading, paragraphs in reading order, escaping
    assert html.startswith("<h1>")
    assert page.export_as("html") == html
    assert HTMLExporter().export_page(page) == html
    # list items render as one <li> per item
    lines = [_line_at(f"item {idx} <x>", 0.1, 0.1 + 0.1 * idx, 0.5, 0.13 + 0.1 * idx) for idx in range(2)]
    layout = [
        elements.LayoutElement("List-item", 0.9, ((0.08, 0.09 + 0.1 * idx), (0.52, 0.14 + 0.1 * idx)))
        for idx in range(2)
    ]
    lp = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800), layout=layout
    )
    assert lp.export_as_html() == "<ul>\n<li>item 0 &lt;x&gt;</li>\n<li>item 1 &lt;x&gt;</li>\n</ul>"


def test_export_mixins_carry_full_api():
    # The element export surface comes from the mixins in doctr.io.exporters, with the API unchanged
    from doctr.io.exporters import DocumentExportsMixin, KIEPageExportsMixin, PageExportsMixin

    for method in (
        "render",
        "export_as_xml",
        "export_as_markdown",
        "export_as_asciidoc",
        "export_as_html",
        "export_as",
    ):
        assert getattr(elements.Page, method) is getattr(PageExportsMixin, method)
        assert getattr(elements.KIEPage, method) is getattr(KIEPageExportsMixin, method)
        assert getattr(elements.Document, method) is getattr(DocumentExportsMixin, method)
    assert elements.Page.items_in_reading_order is PageExportsMixin.items_in_reading_order
    page = _reading_order_page()
    # dispatcher covers every format
    for fmt in ("markdown", "md", "asciidoc", "adoc", "html", "text", "txt", "json", "dict", "xml", "hocr"):
        page.export_as(fmt)
    with pytest.raises(ValueError):
        page.export_as("pptx")


def test_page_render():
    left = elements.Block(
        lines=[_line_at("left top", 0.08, 0.1, 0.45, 0.13), _line_at("left low", 0.08, 0.2, 0.45, 0.23)]
    )
    right = elements.Block(
        lines=[_line_at("right top", 0.55, 0.1, 0.92, 0.13), _line_at("right low", 0.55, 0.2, 0.92, 0.23)]
    )
    # The blocks are stored right-first, but render() linearizes them like the other exporters
    page = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [right, left], 0, (1000, 800))
    assert page.render() == "left top\n\nleft low\n\nright top\n\nright low"
    assert page.render(block_break=" | ") == "left top | left low | right top | right low"
    # Single-block and empty pages
    single = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [left], 0, (1000, 800))
    assert single.render(block_break=" | ") == "left top | left low"
    assert elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [], 0, (1000, 800)).render() == ""


def test_page_render_includes_tables_and_furniture_flag():
    page = _reading_order_page()
    assert page.render().splitlines()[0] == "A Two Column Study"
    assert "Page 3 of 12" in page.render()
    # Page furniture can be dropped, exactly like in the Markdown / HTML exports
    assert "Page 3 of 12" not in page.render(include_furniture=False)
    # Recognized tables are part of the plain text render
    cells = [
        elements.TableCell("head", 0.9, ((0.1, 0.6), (0.4, 0.65)), 0, 0, 0, 0),
        elements.TableCell("body", 0.9, ((0.1, 0.66), (0.4, 0.71)), 1, 1, 0, 0),
    ]
    table = elements.Table(cells=cells, num_rows=2, num_cols=1, geometry=((0.1, 0.6), (0.4, 0.71)))
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_line_at("intro line", 0.1, 0.1, 0.9, 0.15)])],
        0,
        (1000, 800),
        tables=[table],
    )
    assert page.render() == "intro line\n\nhead\nbody"


def test_export_is_json_serializable():
    import json

    # Straight boxes built from a detection array carry np.float32 coordinates
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [
            elements.Block(
                lines=[
                    elements.Line([
                        elements.Word(
                            "np",
                            np.float32(0.9),
                            ((np.float32(0.1), np.float32(0.1)), (np.float32(0.4), np.float32(0.2))),
                            np.float32(0.8),
                            {"value": np.int64(0), "confidence": None},
                        )
                    ])
                ]
            )
        ],
        np.int64(0),
        (np.int64(1000), np.int64(800)),
        {"value": np.float32(0.0), "confidence": np.float32(1.0)},
    )
    exported = page.export()
    assert json.loads(json.dumps(exported)) is not None
    word = exported["blocks"][0]["lines"][0]["words"][0]
    assert [coord for point in word["geometry"] for coord in point] == pytest.approx([0.1, 0.1, 0.4, 0.2])
    assert all(type(coord) is float for point in word["geometry"] for coord in point)
    assert type(word["confidence"]) is float and type(word["objectness_score"]) is float
    assert type(word["crop_orientation"]["value"]) is int
    assert exported["page_idx"] == 0 and type(exported["page_idx"]) is int
    assert exported["dimensions"] == (1000, 800) and all(type(dim) is int for dim in exported["dimensions"])
    # Rotated lines / blocks expose their geometry as a numpy array, exported as nested tuples
    poly = np.asarray([[0.1, 0.1], [0.4, 0.1], [0.4, 0.2], [0.1, 0.2]], dtype=np.float32)
    rot_word = elements.Word("rot", 0.9, poly, 0.8, {"value": 0, "confidence": None})
    rot_page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[elements.Line([rot_word])])],
        0,
        (1000, 800),
    )
    assert isinstance(rot_page.blocks[0].geometry, np.ndarray)  # untouched in memory
    rot_export = rot_page.export()
    assert isinstance(rot_export["blocks"][0]["geometry"], tuple)
    json.dumps(rot_export)
    # ... and a whole document round-trips through JSON
    doc = elements.Document([page, rot_page])
    assert elements.Document.from_dict(json.loads(json.dumps(doc.export()))) is not None


def test_export():
    page = _reading_order_page()
    rendered = [
        " ".join(word["value"] for line in block["lines"] for word in line["words"])
        for block in page.export()["blocks"]
    ]
    assert rendered[0] == "A Two Column Study"
    assert rendered[-1] == "Page 3 of 12"
    assert rendered.index("left line 0 left line 1 left line 2") < rendered.index(
        "right line 0 right line 1 right line 2"
    )
    # The linearization can be opted out of
    raw = page.export(reading_order=False)["blocks"]
    assert len(raw) == 1 and len(raw[0]["lines"]) == 9
    # Artefacts survive the regrouping
    artefact = elements.Artefact("qr_code", 0.9, ((0.1, 0.8), (0.2, 0.9)))
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_line_at("a line", 0.1, 0.1, 0.9, 0.15)], artefacts=[artefact])],
        0,
        (1000, 800),
    )
    assert page.export()["blocks"][0]["artefacts"] == [artefact.export()]


def test_xml_export():
    import re

    page = _reading_order_page()
    xml = page.export_as_xml()[0].decode()
    words = re.findall(r'class="ocrx_word"[^>]*>([^<]*)<', xml)
    assert words[:4] == ["A", "Two", "Column", "Study"]
    assert words[-4:] == ["Page", "3", "of", "12"]
    assert words.index("left") < words.index("right")
    # Recognized tables are serialized as an hOCR text area (they used to be dropped)
    cells = [
        elements.TableCell("head", 0.9, ((0.1, 0.6), (0.4, 0.65)), 0, 0, 0, 0),
        elements.TableCell("body", 0.9, ((0.1, 0.66), (0.4, 0.71)), 1, 1, 0, 0),
    ]
    table = elements.Table(cells=cells, num_rows=2, num_cols=1, geometry=((0.1, 0.6), (0.4, 0.71)))
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_line_at("intro line", 0.1, 0.1, 0.9, 0.15)])],
        0,
        (1000, 800),
        tables=[table],
    )
    xml = page.export_as_xml()[0].decode()
    assert 'id="table_1"' in xml and ">head<" in xml and ">body<" in xml


def test_kie_page_export():
    predictions = {
        CLASS_NAME: [
            elements.Prediction("second", 0.9, ((0.1, 0.5), (0.9, 0.6)), 0.9, {"value": 0, "confidence": None}),
            elements.Prediction("first", 0.9, ((0.1, 0.1), (0.9, 0.2)), 0.9, {"value": 0, "confidence": None}),
        ]
    }
    page = elements.KIEPage(np.zeros((10, 10, 3), dtype=np.uint8), predictions, 0, (1000, 800))
    assert [p["value"] for p in page.export()["predictions"][CLASS_NAME]] == ["first", "second"]
    assert [p["value"] for p in page.export(reading_order=False)["predictions"][CLASS_NAME]] == ["second", "first"]
    xml = page.export_as_xml()[0].decode()
    assert xml.index(">first<") < xml.index(">second<")


def test_kie_page_render_keeps_reading_order():
    predictions = {
        CLASS_NAME: [
            elements.Prediction("second", 0.9, ((0.1, 0.5), (0.9, 0.6)), 0.9, {"value": 0, "confidence": None}),
            elements.Prediction("first", 0.9, ((0.1, 0.1), (0.9, 0.2)), 0.9, {"value": 0, "confidence": None}),
        ]
    }
    page = elements.KIEPage(np.zeros((10, 10, 3), dtype=np.uint8), predictions, 0, (1000, 800))
    assert page.render() == f"{CLASS_NAME}: first\n\n{CLASS_NAME}: second"


def test_xml_exporter_class():
    from xml.etree import ElementTree as ET

    page = _reading_order_page()
    xml_bytes, tree = XMLExporter().export_page(page)
    assert isinstance(xml_bytes, bytes) and isinstance(tree, ET.ElementTree)
    # The mixin method delegates to the exporter class, so the output is identical
    assert page.export_as_xml()[0] == xml_bytes
    assert page.export_as("xml")[0] == xml_bytes

    predictions = {
        CLASS_NAME: [elements.Prediction("hi", 0.9, ((0.1, 0.1), (0.9, 0.2)), 0.9, {"value": 0, "confidence": None})]
    }
    kie = elements.KIEPage(np.zeros((10, 10, 3), dtype=np.uint8), predictions, 0, (1000, 800))
    assert kie.export_as_xml()[0] == XMLExporter().export_kie_page(kie)[0]

    doc = elements.Document([page, page])
    doc_xml = XMLExporter().export_document(doc)
    assert len(doc_xml) == 2 and doc.export_as_xml()[0][0] == doc_xml[0][0]


def _table_page(num_rows=2, num_cols=1, rotated=False):
    """A page with an intro line and one recognized table"""
    geo = ((0.1, 0.6), (0.4, 0.66), (0.4, 0.71), (0.1, 0.71)) if rotated else ((0.1, 0.6), (0.4, 0.71))
    cells = [
        elements.TableCell("head", 0.9, ((0.1, 0.6), (0.4, 0.65)), 0, 0, 0, 0),
        elements.TableCell("body", 0.9, ((0.1, 0.66), (0.4, 0.71)), 1, 1, 0, 0),
    ]
    table = elements.Table(cells=cells, num_rows=num_rows, num_cols=num_cols, geometry=geo)
    return elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_line_at("intro line", 0.1, 0.1, 0.9, 0.15)])],
        0,
        (1000, 800),
        tables=[table],
    )


def test_to_json_safe_covers_every_container():
    # 0-d arrays and numpy scalars collapse to their Python equivalent
    assert to_json_safe(np.array(3.5)) == 3.5 and type(to_json_safe(np.array(3.5))) is float
    assert to_json_safe(np.int64(7)) == 7 and type(to_json_safe(np.int64(7))) is int
    assert to_json_safe(np.bool_(True)) is True
    # arrays become nested tuples, so `create_obj_patch` still recognizes an exported geometry
    assert to_json_safe(np.asarray([[0.0, 1.0], [2.0, 3.0]])) == ((0.0, 1.0), (2.0, 3.0))
    # lists stay lists, sets are normalized to lists, dict keys to str
    assert to_json_safe([np.float32(1.0), (np.int8(2),)]) == [1.0, (2,)]
    assert sorted(to_json_safe({np.int64(1), np.int64(2)})) == [1, 2]
    assert to_json_safe({np.int64(1): np.float32(0.5)}) == {"1": 0.5}
    # anything already built-in is returned untouched
    assert to_json_safe("text") == "text" and to_json_safe(None) is None
    json.dumps(to_json_safe({"a": [np.float32(0.1)], "b": np.arange(3)}))


def test_page_artefacts():
    artefact = elements.Artefact("qr_code", 0.9, ((0.7, 0.7), (0.9, 0.9)))
    orphan = elements.Block(lines=[], artefacts=[artefact], geometry=((0.7, 0.7), (0.9, 0.9)), objectness_score=0.9)
    text = elements.Block(lines=[_line_at("some text", 0.1, 0.1, 0.9, 0.15)])
    page = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [orphan, text], 0, (1000, 800))
    items = page.items_in_reading_order()
    assert [artefact.export() for artefact in items[-1].artefacts] == [artefact.export()]
    assert page.export()["blocks"][-1]["artefacts"] == [artefact.export()]


def test_ordered_line_words_directions():
    line = _line_at("one two three", 0.1, 0.1, 0.7, 0.15)
    assert [word.render() for word in ordered_line_words(line, "ltr")] == ["one", "two", "three"]
    assert [word.render() for word in ordered_line_words(line, "rtl")] == ["three", "two", "one"]
    # Vertical pages read their words top to bottom
    stacked = elements.Line([_word_at(value, 0.1, 0.1 * idx, 0.2, 0.1 * idx + 0.05) for idx, value in enumerate("abc")])
    assert [word.render() for word in ordered_line_words(stacked, "ttb-rtl")] == ["a", "b", "c"]
    assert [word.render() for word in ordered_line_words(stacked, "ttb-ltr")] == ["a", "b", "c"]
    # A single-word line short-circuits the per-line detection
    single = elements.Line([_word_at("solo", 0.1, 0.1, 0.2, 0.15)])
    assert [word.render() for word in ordered_line_words(single, "auto", auto=True)] == ["solo"]


def test_vertical_direction_reaches_every_exporter():
    page = _reading_order_page()
    for export in (page.render, page.export_as_markdown, page.export_as_asciidoc, page.export_as_html):
        assert isinstance(export(direction="ttb-ltr"), str)
    assert isinstance(page.export_as_xml(direction="ttb-rtl")[0], bytes)


def test_html_export_with_tables_and_furniture():
    page = _table_page()
    html = page.export_as_html()
    assert "<table>" in html and "<th>head</th>" in html and "<td>body</td>" in html
    assert html.index("<p>intro line</p>") < html.index("<table>")
    # A single-row table has a header but no body
    single = elements.Table(
        cells=[elements.TableCell("only", 0.9, ((0.1, 0.6), (0.4, 0.65)), 0, 0, 0, 0)],
        num_rows=1,
        num_cols=1,
        geometry=((0.1, 0.6), (0.4, 0.65)),
    )
    assert HTMLExporter().render_table(single) == "<table>\n<tr><th>only</th></tr>\n</table>"
    # Page furniture can be dropped from the HTML export too
    assert "Page 3 of 12" in _reading_order_page().export_as_html()
    assert "Page 3 of 12" not in _reading_order_page().export_as_html(include_furniture=False)


def test_render_table_with_empty_grid():
    empty = elements.Table(cells=[], num_rows=0, num_cols=0, geometry=((0.0, 0.0), (0.0, 0.0)))
    for exporter in (MarkdownExporter(), AsciiDocExporter(), HTMLExporter(), TextExporter()):
        assert exporter.render_table(empty) == ""
    # An empty table contributes nothing to the page export
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_line_at("only text", 0.1, 0.1, 0.9, 0.15)])],
        0,
        (1000, 800),
        tables=[empty],
    )
    assert page.export_as_markdown() == "only text"
    assert page.export_as_html() == "<p>only text</p>"


def test_text_exporter_kie_and_document():
    predictions = {
        CLASS_NAME: [
            elements.Prediction("second", 0.9, ((0.1, 0.5), (0.9, 0.6)), 0.9, {"value": 0, "confidence": None}),
            elements.Prediction("first", 0.9, ((0.1, 0.1), (0.9, 0.2)), 0.9, {"value": 0, "confidence": None}),
        ],
        "empty": [],
    }
    kie = elements.KIEPage(np.zeros((10, 10, 3), dtype=np.uint8), predictions, 0, (1000, 800))
    # Classes without any prediction are skipped by every text exporter
    assert TextExporter().export_kie_page(kie) == f"{CLASS_NAME}:\n\nfirst\nsecond"
    assert "empty" not in MarkdownExporter().export_kie_page(kie)
    assert "empty" not in AsciiDocExporter().export_kie_page(kie)
    assert "empty" not in HTMLExporter().export_kie_page(kie)
    # export_document dispatches on the page type and uses the format-specific page break
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_line_at("a page", 0.1, 0.1, 0.9, 0.15)])],
        0,
        (1000, 800),
    )
    mixed = elements.Document([page, kie])
    assert TextExporter().export_document(mixed) == f"a page\n\n\n\n{CLASS_NAME}:\n\nfirst\nsecond"
    assert TextExporter().export_document(mixed, page_break=" // ").startswith("a page // ")


def test_base_text_exporter_is_abstract():
    from doctr.io.exporters import _PageTextExporter

    table = elements.Table(cells=[], num_rows=0, num_cols=0, geometry=((0.0, 0.0), (0.0, 0.0)))
    with pytest.raises(NotImplementedError):
        _PageTextExporter().render_table(table)
    with pytest.raises(NotImplementedError):
        _PageTextExporter().class_header("cls")


def test_blank_lines_are_skipped():
    # Words recognized as empty strings must not produce an empty paragraph / list item
    blank = elements.Block(lines=[elements.Line([_word_at("", 0.1, 0.1, 0.4, 0.15)])])
    text = elements.Block(lines=[_line_at("real text", 0.1, 0.5, 0.9, 0.55)])
    page = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [blank, text], 0, (1000, 800))
    assert page.render() == "real text"
    assert page.export_as_markdown() == "real text"
    assert page.export_as_html() == "<p>real text</p>"


def test_xml_export_edge_cases():
    from doctr.io.exporters import _covering_region_indices

    assert _covering_region_indices([], [((0.0, 0.0), (1.0, 1.0))]) == []
    # Rotated geometries are rejected with an explicit error, for blocks, tables and KIE predictions alike
    poly = np.asarray([[0.1, 0.1], [0.4, 0.1], [0.4, 0.2], [0.1, 0.2]], dtype=np.float32)
    rot_page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [
            elements.Block(
                lines=[elements.Line([elements.Word("rot", 0.9, poly, 0.9, {"value": 0, "confidence": None})])]
            )
        ],
        0,
        (1000, 800),
    )
    with pytest.raises(TypeError):
        rot_page.export_as_xml()
    with pytest.raises(TypeError):
        XMLExporter().export_page(_table_page(rotated=True))
    rot_kie = elements.KIEPage(
        np.zeros((10, 10, 3), dtype=np.uint8),
        {CLASS_NAME: [elements.Prediction("rot", 0.9, poly, 0.9, {"value": 0, "confidence": None})]},
        0,
        (1000, 800),
    )
    with pytest.raises(TypeError):
        rot_kie.export_as_xml()
    # The linearization can be opted out of: blocks first, then tables, in their stored order
    xml = _table_page().export_as_xml(reading_order=False)[0].decode()
    assert xml.index("intro") < xml.index(">head<")


def test_document_forwards_reading_order_options():
    left = elements.Block(lines=[_line_at("left col", 0.08, 0.2, 0.45, 0.23)])
    right = elements.Block(lines=[_line_at("right col", 0.55, 0.2, 0.92, 0.23)])
    footer = elements.Block(lines=[_line_at("Page 1 of 2", 0.4, 0.95, 0.6, 0.97)])
    layout = [
        elements.LayoutElement("Text", 0.98, ((0.06, 0.18), (0.48, 0.26))),
        elements.LayoutElement("Text", 0.98, ((0.52, 0.18), (0.94, 0.26))),
        elements.LayoutElement("Page-footer", 0.97, ((0.35, 0.94), (0.65, 0.98))),
    ]
    page = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [right, left, footer], 0, (1000, 800), layout=layout)
    doc = elements.Document([page, page])

    assert (
        doc.render(page_break=" || ") == "left col\n\nright col\n\nPage 1 of 2 || left col\n\nright col\n\nPage 1 of 2"
    )
    assert "Page 1 of 2" not in doc.render(include_furniture=False)
    assert doc.render(block_break=" ", page_break=" || ").startswith("left col right col")
    assert doc.export_as("text", include_furniture=False) == doc.render(include_furniture=False)

    ordered = doc.export()["pages"][0]["blocks"]
    assert [block["lines"][0]["words"][0]["value"] for block in ordered] == ["left", "right", "Page"]
    raw = doc.export(reading_order=False)["pages"][0]["blocks"]
    assert [block["lines"][0]["words"][0]["value"] for block in raw] == ["right", "left", "Page"]
    assert doc.export_as("json", reading_order=False) == doc.export(reading_order=False)
    json.dumps(doc.export(reading_order=False))


def test_kie_document_forwards_reading_order_options():
    predictions = {
        CLASS_NAME: [
            elements.Prediction("second", 0.9, ((0.1, 0.5), (0.9, 0.6)), 0.9, {"value": 0, "confidence": None}),
            elements.Prediction("first", 0.9, ((0.1, 0.1), (0.9, 0.2)), 0.9, {"value": 0, "confidence": None}),
        ]
    }
    page = elements.KIEPage(np.zeros((10, 10, 3), dtype=np.uint8), predictions, 0, (1000, 800))
    doc = elements.KIEDocument([page])
    assert doc.render() == f"{CLASS_NAME}: first\n\n{CLASS_NAME}: second"
    # The reading direction reaches KIEPage.render through the document
    assert doc.render(direction="ttb-ltr") == f"{CLASS_NAME}: first\n\n{CLASS_NAME}: second"
    assert [p["value"] for p in doc.export()["pages"][0]["predictions"][CLASS_NAME]] == ["first", "second"]
    raw = doc.export(reading_order=False)["pages"][0]["predictions"][CLASS_NAME]
    assert [p["value"] for p in raw] == ["second", "first"]
    json.dumps(doc.export(reading_order=False))


def test_every_format_is_reachable_from_a_document():
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block(lines=[_line_at("page text", 0.1, 0.1, 0.9, 0.15)])],
        0,
        (1000, 800),
    )
    predictions = {
        CLASS_NAME: [elements.Prediction("value", 0.9, ((0.1, 0.1), (0.9, 0.2)), 0.9, {"value": 0, "confidence": None})]
    }
    kie_page = elements.KIEPage(np.zeros((10, 10, 3), dtype=np.uint8), predictions, 0, (1000, 800))
    formats = ("markdown", "md", "asciidoc", "adoc", "html", "text", "txt", "json", "dict", "xml", "hocr")

    for element in (page, kie_page, elements.Document([page]), elements.KIEDocument([kie_page])):
        for fmt in formats:
            assert element.export_as(fmt) is not None
        with pytest.raises(ValueError, match="unsupported export format"):
            element.export_as("pptx")

    # The document-level named methods carry the page break of their format
    doc = elements.Document([page, page])
    assert doc.export_as_html() == "<p>page text</p><hr><p>page text</p>"
    assert doc.export_as_html(page_break="\n") == "<p>page text</p>\n<p>page text</p>"
    assert doc.export_as_markdown().count("\n\n---\n\n") == 1
    assert doc.export_as_asciidoc().count("\n\n<<<\n\n") == 1
    assert len(doc.export_as_xml()) == 2

    kie_doc = elements.KIEDocument([kie_page, kie_page])
    kie_html = f"<h3>{CLASS_NAME}</h3>\n<ul>\n<li>value</li>\n</ul>"
    assert kie_doc.export_as_html() == f"{kie_html}<hr>{kie_html}"
    assert kie_page.export_as_html() == f"<h3>{CLASS_NAME}</h3>\n<ul>\n<li>value</li>\n</ul>"
    assert kie_page.export_as_html(direction="rtl") == kie_page.export_as_html()
