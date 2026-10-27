# User guide

The user guide explains how to get the most out of docTR on your own documents, one topic at a time. Each page
is self-contained and full of examples you can copy.

## The essentials

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`file;1.1em` Loading documents
:link: loading_documents
:link-type: doc

Read PDFs, images, web pages and in-memory data into pages.
:::

:::{grid-item-card} {octicon}`gear;1.1em` Running OCR
:link: ocr_predictor
:link-type: doc

Choose the models and configure the end-to-end OCR predictor.
:::

:::{grid-item-card} {octicon}`list-unordered;1.1em` Working with results
:link: results
:link-type: doc

Navigate the `Document`, convert coordinates, visualize and save predictions.
:::

:::{grid-item-card} {octicon}`file-code;1.1em` Exporting results
:link: exporting
:link-type: doc

Text, JSON, Markdown, AsciiDoc, HTML and hOCR, in reading order.
:::
::::

## Beyond plain text

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` Layout analysis & tables
:link: layout_and_tables
:link-type: doc

Detect titles, figures and tables, and extract tables as row/column grids.
:::

:::{grid-item-card} {octicon}`sync;1.1em` Rotated documents
:link: rotated_documents
:link-type: doc

Read skewed scans, photos and pages in any orientation.
:::

:::{grid-item-card} {octicon}`tag;1.1em` Key information extraction
:link: kie
:link-type: doc

Detect several classes of fields (dates, amounts...) and read them in one pass.
:::

:::{grid-item-card} {octicon}`package;1.1em` Task-specific predictors
:link: standalone_predictors
:link-type: doc

Use the detection, recognition, layout, table and orientation models on their own.
:::

:::{grid-item-card} {octicon}`image;1.1em` Artefact detection
:link: artefacts
:link-type: doc

Find logos, QR codes, barcodes and photos with the `contrib` module.
:::

:::{grid-item-card} {octicon}`terminal;1.1em` Command-line interface
:link: cli
:link-type: doc

Run OCR on files from your terminal with `doctr-cli`.
:::
::::

## When something goes wrong

:::{card} {octicon}`question;1.1em` Troubleshooting & FAQ
:link: faq
:link-type: doc

Solutions to the most common installation, input and accuracy issues.
:::

```{toctree}
:hidden:
:caption: The essentials

loading_documents
ocr_predictor
results
exporting
```

```{toctree}
:hidden:
:caption: Beyond plain text

layout_and_tables
rotated_documents
kie
standalone_predictors
artefacts
cli
```

```{toctree}
:hidden:
:caption: Help

faq
```
