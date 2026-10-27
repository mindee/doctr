# Command-line interface

`doctr-cli` runs the complete OCR pipeline on images and PDFs from your terminal, and writes the result as JSON,
plain text, Markdown, AsciiDoc, HTML or hOCR. It is installed with the package.

```bash
doctr-cli --input_path invoice.pdf --output invoice.json
```

## Common recipes

::::{tab-set}

:::{tab-item} Text
```bash
doctr-cli --input_path scan.jpg --output scan.txt
```
:::

:::{tab-item} Markdown with tables
```bash
doctr-cli --input_path report.pdf --detect_tables --output report.md
```
:::

:::{tab-item} Several files
Every page of every input is appended to a single document:

```bash
doctr-cli --input_path page1.jpg page2.jpg annex.pdf --output dossier.json
```
:::

:::{tab-item} Rotated scans
```bash
doctr-cli --input_path skewed.pdf --straighten_pages --output skewed.txt
```
:::

:::{tab-item} GPU & hOCR
On a CUDA device of compute capability 8.0 or newer, the models run in bfloat16:

```bash
doctr-cli --input_path doc.pdf --device cuda:0 --output doc.xml
```
:::

:::{tab-item} Web page
Requires the `html` extra:

```bash
doctr-cli --input_path https://www.example.com --output page.md
```
:::
::::

The format is **inferred from the extension** of `--output` (`.json`, `.txt`, `.md`, `.adoc`, `.html`, `.xml`) and
falls back to JSON. Use `--format` to set it explicitly.

## Arguments

Run `doctr-cli --help` for the full list with default values.

### Input

| Argument          | Default | Description                                                                                 |
| ----------------- | ------- | ------------------------------------------------------------------------------------------- |
| `--input_path`    | —       | **Required.** One or more images, PDF files, or `http(s)://` URLs of web pages               |
| `--pdf_password`  | `None`  | Password of encrypted PDF files                                                             |
| `--pdf_scale`     | `2`     | PDF rendering scale (1 corresponds to 72 dpi)                                               |

### Models

| Argument        | Default           | Description                                                                       |
| --------------- | ----------------- | --------------------------------------------------------------------------------- |
| `--det_arch`    | `fast_base`       | Detection architecture                                                            |
| `--reco_arch`   | `crnn_vgg16_bn`   | Recognition architecture                                                          |
| `--layout_arch` | `lw_detr_s`       | Layout architecture                                                               |
| `--device`      | `auto`            | `auto`, `cpu`, `cuda`, `cuda:1`, `mps`... (`auto` picks a GPU when available)       |

### Processing

| Argument                                         | Default   | Description                                                            |
| ------------------------------------------------ | --------- | ---------------------------------------------------------------------- |
| `--assume_straight_pages` / `--no-assume_straight_pages` | on | Handle pages as straight (see [](rotated_documents.md))          |
| `--straighten_pages`                             | off       | Straighten rotated pages before reading them                           |
| `--preserve_aspect_ratio` / `--no-preserve_aspect_ratio` | on | Pad pages instead of stretching them                             |
| `--symmetric_pad` / `--no-symmetric_pad`         | on        | Pad on all sides                                                       |
| `--det_bs`, `--reco_bs`                          | `2`, `128`| Batch sizes of the detection and recognition models                    |
| `--bin_thresh`, `--box_thresh`                   | model     | [Detection thresholds](ocr_predictor.md#tune-the-text-detection), between 0 and 1 |
| `--confidence_aggregation`                       | model     | `mean`, `min`, `max`, `median`, `geometric_mean` or `harmonic_mean`     |
| `--detect_orientation`                           | off       | Estimate the orientation of each page                                  |
| `--detect_language`                              | off       | Estimate the language of each page                                     |
| `--detect_layout`                                | off       | Attach the layout regions to each page                                 |
| `--detect_tables`                                | off       | Recognize tables (enables the layout model)                            |
| `--ignore_regions CLASS [CLASS ...]`             | none      | Layout classes to mask before OCR, e.g. `Picture Table`                |
| `--export_as_straight_boxes`                     | off       | Export straight boxes with `--no-assume_straight_pages`                |
| `--preserve_original_coords`                     | off       | With `--straighten_pages`, map the boxes back to the original pages    |
| `--disable_page_orientation`                     | off       | Skip the page orientation classifier                                   |
| `--disable_crop_orientation`                     | off       | Skip the crop orientation classifier                                   |

### Document assembly

| Argument                                  | Default | Description                                         |
| ----------------------------------------- | ------- | --------------------------------------------------- |
| `--resolve_lines` / `--no-resolve_lines`  | on      | Group words into lines                              |
| `--resolve_blocks`                        | off     | Group lines into blocks                             |
| `--paragraph_break`                       | `0.035` | Relative space that splits a line                   |
| `--keep_reading_order`                    | off     | Sort the content of each page in reading order      |

### Output

| Argument                                          | Default                       | Description                                                                 |
| ------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------- |
| `--output`                                        | `results.json`                | Output file                                                                 |
| `--format`                                        | from extension                | `json`/`dict`, `txt`/`text`, `md`/`markdown`, `adoc`/`asciidoc`, `html`, `xml`/`hocr` |
| `--direction`                                     | `auto`                        | Reading direction: `auto`, `ltr`, `rtl`, `ttb-rtl`, `ttb-ltr`               |
| `--reading_order` / `--no-reading_order`          | on                            | Linearize JSON and hOCR exports in reading order                            |
| `--escape` / `--no-escape`                        | on                            | Escape markup characters (Markdown, AsciiDoc)                               |
| `--include_furniture` / `--no-include_furniture`  | on                            | Include page headers, footers and footnotes                                 |
| `--images`                                        | `placeholder`                 | Figures in Markdown, AsciiDoc and HTML: `none`, `placeholder`, `embedded`, `referenced` (see [figures](exporting.md#figures)) |
| `--image_dir`                                     | `<output name>_images`        | Directory of the figures with `--images referenced`                         |
| `--image_format`                                  | `png`                         | `png`, `jpg`, `jpeg` or `webp`                                              |
| `--file_title`                                    | `docTR - XML export (hOCR)`   | Title of the hOCR files                                                     |
| `--indent`                                        | `4`                           | Indentation of the JSON export                                              |
| `--quiet`                                         | off                           | Only log errors                                                             |
| `--version`                                       |                               | Print the docTR version and exit                                            |

## Output files

- **JSON** contains the full structure: pages (dimensions, orientation, language, layout), blocks, lines and words
  with their confidence and geometry, and the tables with `--detect_tables`.
- **Text-like formats** (`txt`, `md`, `adoc`, `html`) are written to a single file.
- **hOCR** describes one page per file: `--output doc.xml` on a three-page document writes `doc_page_1.xml`,
  `doc_page_2.xml` and `doc_page_3.xml`.
