# Exporting results

A {class}`~doctr.io.Document` can be exported to six formats. All of them present the content in the same
**reading order**, which follows columns, places headers and footers, and adapts to right-to-left scripts.

| Format        | Method                                                  | Output                                   | Best for                                    |
| ------------- | ------------------------------------------------------- | ---------------------------------------- | ------------------------------------------- |
| Plain text    | {meth}`~doctr.io.Document.render`                         | `str`                                    | Search indexing, NLP, LLM prompts           |
| JSON          | {meth}`~doctr.io.Document.export`                         | `dict`                                   | Storing every word with its geometry        |
| Markdown      | {meth}`~doctr.io.Document.export_as_markdown`             | `str`                                    | LLMs, RAG pipelines, readable documents     |
| AsciiDoc      | {meth}`~doctr.io.Document.export_as_asciidoc`             | `str`                                    | Technical documentation toolchains          |
| HTML          | {meth}`~doctr.io.Document.export_as_html`                 | `str`                                    | Web display                                 |
| hOCR (XML)    | {meth}`~doctr.io.Document.export_as_xml`                  | `list[tuple[bytes, ElementTree]]`        | Searchable PDFs, OCR interoperability       |

The same methods exist on a single {class}`~doctr.io.Page`. The {meth}`~doctr.io.Document.export_as` dispatcher
selects the format by name, which is convenient when it comes from a configuration:

```python
result.export_as("markdown")  # or "md"
result.export_as("asciidoc")  # or "adoc"
result.export_as("html")
result.export_as("text")  # or "txt", same as render()
result.export_as("json")  # or "dict", same as export()
result.export_as("xml")  # or "hocr", same as export_as_xml()
```

## Plain text

```python
text = result.render()
```

Blocks are separated by a blank line and pages by three blank lines. Both separators are configurable, and page
headers and footers can be left out:

```python
text = result.render(page_break="\f", include_furniture=False)
```

## Markdown, AsciiDoc and HTML

```python
markdown = result.export_as_markdown()
asciidoc = result.export_as_asciidoc()
html = result.export_as_html()
```

The structure of these exports depends on what the predictor detected:

- **Without layout analysis**, blocks become paragraphs.
- **With layout analysis** (`detect_layout=True`), the layout regions are used to render titles and section
  headers as headings, list items as lists, and to place the page furniture (headers, footers, footnotes).
- **With table recognition** (`detect_tables=True`), tables are rendered as real tables.

::::{tab-set}

:::{tab-item} Markdown
```markdown
## ACME Office Supplies

42 Harbour Road, Hamburg

## Bill to:

Jane Doe

12 Market Street, Berlin

INVOICE

No. 2026-0412
Date: 26/10/2026

| Description | Qty | Unit price | Amount |
| --- | --- | --- | --- |
| Recycled paper A4 | 10 | 8.50 | 85.00 |
| Ballpoint pens (box) | 5 | 12.00 | 60.00 |
| Desk lamp | 2 | 45.00 | 90.00 |
| Office chair | 1 | 1,015.00 | 1,015.00 |

Total due

1,250.00 EUR
```
:::

:::{tab-item} HTML
```html
<h2>ACME Office Supplies</h2>
<p>42 Harbour Road, Hamburg</p>
<h2>Bill to:</h2>
<p>Jane Doe</p>
<p>12 Market Street, Berlin</p>
<p>INVOICE</p>
<p>No. 2026-0412<br>
Date: 26/10/2026</p>
<table>
<tr><th>Description</th><th>Qty</th><th>Unit price</th><th>Amount</th></tr>
<tr><td>Recycled paper A4</td><td>10</td><td>8.50</td><td>85.00</td></tr>
...
</table>
<p>Total due</p>
<p>1,250.00 EUR</p>
```
:::
::::

*Output of `ocr_predictor(pretrained=True, detect_tables=True)` on the
{download}`sample invoice <../_static/images/examples/invoice.pdf>`.*

Useful options:

`escape` (Markdown, AsciiDoc)
: Escape the characters that carry a meaning in the markup (`*`, `#`, `|`...). Enabled by default.

`include_furniture`
: Include page headers, page footers and footnotes. Enabled by default.

`page_break`
: The separator inserted between pages: `---` in Markdown, `<<<` in AsciiDoc, `<hr>` in HTML.

`images`
: How figures are rendered, see [](#figures) below.

## hOCR

[hOCR](https://github.com/kba/hocr-spec) is an open standard to store OCR results in XML. docTR produces one hOCR
document per page, as raw bytes and as an {class}`xml.etree.ElementTree.ElementTree`:

```python
for page_idx, (xml_bytes, xml_tree) in enumerate(result.export_as_xml()):
    with open(f"page_{page_idx + 1}.hocr", "wb") as f:
        f.write(xml_bytes)
```

hOCR files are the input of tools such as [hocr-tools](https://github.com/ocropus/hocr-tools) or
[OCRmyPDF](https://ocrmypdf.readthedocs.io). The
[Export as PDF/A notebook](https://github.com/mindee/notebooks/blob/main/doctr/export_as_pdfa.ipynb) shows how to
turn docTR results into **searchable PDFs**.

## Reading order

Every export path shares the same linearization of the page, computed once and memoized. It:

- follows the **columns** of multi-column layouts instead of reading straight across the page,
- infers the **reading direction** from the recognized text (right-to-left for Arabic or Hebrew, for instance),
- uses the **layout regions**, when available, to order blocks, headings, tables and figures and to place the
  page furniture.

Force the direction when automatic detection is not what you want:

```python
result.export_as_markdown(direction="rtl")  # 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
```

To get the elements of a page (blocks, tables and optionally figures) in reading order, for your own exporter:

```python
for item in result.pages[0].items_in_reading_order(include_figures=True):
    print(type(item).__name__, item.render()[:40])
```

The JSON export follows the reading order by default (`export(reading_order=False)` keeps the detection order). To
sort the `Document` tree itself, build the predictor with `keep_reading_order=True`.

## Figures

When the layout model runs (`detect_layout=True` or `detect_tables=True`), the regions it labels as figures
(pictures, charts...) take part in the reading order. The Markdown, AsciiDoc and HTML exports render them according
to their `images` argument:

| Mode                         | Rendering                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `"placeholder"` *(default)*  | A comment marks the figure: `<!-- image -->` (Markdown, HTML) or `// image` (AsciiDoc)     |
| `"none"`                     | Figures are left out                                                                      |
| `"embedded"`                 | Each figure is cropped and inlined as a base64 data URI: the export is self-contained      |
| `"referenced"`               | Each crop is written to a directory and linked by its relative path                        |

```python
from doctr.io import DocumentFile, FigureEncoder
from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True, detect_layout=True)
result = predictor(DocumentFile.from_pdf("report.pdf"))

# A self-contained Markdown file, figures included
markdown = result.export_as_markdown(images="embedded")

# A Markdown file linking to the crops written to `assets/`
markdown = result.export_as_markdown(images=FigureEncoder("referenced", image_dir="assets"))
```

{class}`~doctr.io.FigureEncoder` also sets the image format (`png`, `jpg`, `webp`), the quality of lossy formats,
a padding around each crop, and a `path_prefix` when the export is not written next to the image directory.
Crop file names contain the position and a hash of the figure (e.g. `page1_figure2-3fa2b1c9.png`), so several
documents can share a directory.

:::{note}
Once a figure carries its pixels (`"embedded"` and `"referenced"`), the text recognized inside it is left out of
the export, since the image shows it, and its caption becomes the alternative text of the image. Plain text never
contains figures, and hOCR positions them as `ocr_photo` areas.
:::

## From the command line

The [CLI](cli.md) exposes all these formats; the format is inferred from the extension of the output file:

```bash
doctr-cli --input_path report.pdf --detect_layout --output report.md --images referenced
```
