# Core concepts

A few ideas come back on every page of this documentation: **predictors**, the **`Document`** tree they return,
and the way docTR expresses **positions**. Ten minutes here will make everything else easier to follow.

## The OCR pipeline

OCR is split into two consecutive tasks, each solved by a dedicated deep learning model:

1. **Text detection** localizes every word on the page (*where is text?*).
2. **Text recognition** transcribes each detected word from its crop (*what does it say?*).

The predictions are then assembled into a structured document. Optional stages plug into the same pipeline when
you need them: orientation handling for rotated pages, layout analysis, and table structure recognition.

```{raw} html
:file: ../_includes/pipeline.html
```

## Predictors and models

You rarely call a model directly. Instead you use a **predictor**: a callable that takes care of everything
around the neural network so you can feed it raw pages.

| Layer             | Responsibility                                                                     | Example                          |
| ----------------- | ---------------------------------------------------------------------------------- | -------------------------------- |
| **Pre-processor** | Resizes, pads, normalizes and batches the input images                              | {class}`~doctr.models.preprocessor.PreProcessor` |
| **Model**         | A regular PyTorch `nn.Module` that runs the network                                  | `fast_base`, `crnn_vgg16_bn`     |
| **Post-processor**| Turns raw network outputs into boxes, strings and scores                             | decoding of the segmentation map |

docTR exposes a factory function for every task. Each one accepts an architecture **name** (or a model
**instance**, e.g. one you trained yourself) and returns a ready-to-use predictor:

| Task                        | Factory                                                     | Returns                                   |
| --------------------------- | ----------------------------------------------------------- | ----------------------------------------- |
| End-to-end OCR              | {func}`~doctr.models.ocr_predictor`                           | a {class}`~doctr.io.Document`              |
| Key information extraction  | {func}`~doctr.models.kie_predictor`                           | a {class}`~doctr.io.KIEDocument`           |
| Text detection only         | {func}`~doctr.models.detection.detection_predictor`                     | boxes per page                            |
| Text recognition only       | {func}`~doctr.models.recognition.recognition_predictor`                   | `(text, confidence)` per crop              |
| Layout analysis             | {func}`~doctr.models.layout.layout_predictor`                        | labelled regions per page                 |
| Table structure recognition | {func}`~doctr.models.table_structure.table_predictor`                         | cells with row/column spans per table     |

The end-to-end {func}`~doctr.models.ocr_predictor` composes the task predictors, which stay reachable as attributes:

```python
from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True)

predictor.det_predictor  # text detection (pre-processor + model + post-processor)
predictor.reco_predictor  # text recognition
predictor.doc_builder  # assembles words into lines, blocks and pages
```

:::{important}
Always pass `pretrained=True` unless you load your own weights: without it, the models are randomly initialized
and return meaningless predictions.
:::

## The `Document` tree

The OCR predictor returns a {class}`~doctr.io.Document`: a tree of elements that mirrors the structure of what was
read.

```{raw} html
<div class="doctr-tree"><span class="k">Document</span>
└── pages: list[<span class="k">Page</span>]                  <span class="c"># one per input page</span>
    ├── blocks: list[<span class="k">Block</span>]            <span class="c"># groups of lines (paragraphs, addresses...)</span>
    │   ├── lines: list[<span class="k">Line</span>]          <span class="c"># words read together</span>
    │   │   └── words: list[<span class="k">Word</span>]      <span class="c"># value, confidence, geometry</span>
    │   └── artefacts: list[<span class="k">Artefact</span>]  <span class="c"># non-textual elements</span>
    ├── layout: list[<span class="k">LayoutElement</span>]    <span class="c"># with detect_layout=True</span>
    └── tables: list[<span class="k">Table</span>]            <span class="c"># with detect_tables=True</span>
        └── cells: list[<span class="k">TableCell</span>]</div>
```

| Element                                  | What it represents                                                                                   |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| {class}`~doctr.io.Word`                    | An uninterrupted sequence of characters, with its text `value`, recognition `confidence` and `geometry` |
| {class}`~doctr.io.Line`                    | Words aligned and meant to be read together. On a two-column page, a horizontal row holds two lines   |
| {class}`~doctr.io.Block`                   | A group of lines (e.g. an address written on several lines) and artefacts                              |
| {class}`~doctr.io.Artefact`                | A non-textual element: a picture, a logo, a QR code, a signature...                                   |
| {class}`~doctr.io.LayoutElement`           | A region found by the layout model: `Title`, `Text`, `Table`, `Picture`, `Page-header`...             |
| {class}`~doctr.io.Table` / {class}`~doctr.io.TableCell` | A recognized table, as cells spanning rows and columns                                    |
| {class}`~doctr.io.Page`                    | Everything found on one page, plus its `dimensions`, `orientation` and `language`                     |

Every element can be **rendered** to text with `.render()` and **exported** to a JSON-serializable dict with
`.export()`. See [](../user_guide/results.md) for a hands-on tour.

## Coordinates

All geometries are expressed in **relative coordinates**, between `0` and `1`, as a fraction of the page width (x)
and height (y), with the origin at the **top-left** corner. This makes them independent of the resolution the page
was rendered at.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Straight boxes (default)
Two points: the top-left and the bottom-right corners.

```python
((xmin, ymin), (xmax, ymax))
# e.g. ((0.12, 0.05), (0.31, 0.08))
```
:::

:::{grid-item-card} Rotated boxes
With `assume_straight_pages=False`: a polygon of four `(x, y)` points, as a NumPy array of shape `(4, 2)`.

```python
array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
```
:::
::::

To get pixel coordinates, multiply by the page size. Note that `page.dimensions` is `(height, width)`:

```python
height, width = page.dimensions
(xmin, ymin), (xmax, ymax) = word.geometry
box_in_pixels = (int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height))
```

## Scores

docTR attaches several scores to its predictions. They answer different questions:

`Word.confidence`
: How sure the **recognition** model is about the text, between 0 and 1. By default it is the lowest
  probability among the characters of the word (see [confidence aggregation](../user_guide/ocr_predictor.md#confidence-scores)).

`Word.objectness_score`
: How sure the **detection** model is that the box contains text.

`Word.crop_orientation`
: The predicted rotation of the word crop, `{"value": angle, "confidence": score}`, when crop orientation is
  classified (rotated documents).

`Page.orientation` / `Page.language`
: The estimated page rotation and language, filled with `detect_orientation=True` / `detect_language=True`.

## Vocabularies

A recognition model can only output characters from the **vocabulary** it was trained on. The pretrained recognition
models use the `french` vocabulary, which covers Latin characters with the most common accents, digits and
punctuation. docTR ships [100+ vocabularies](../models/vocabs.md) for training models on other languages and
alphabets, and lets you [restrict a model to a subset](../user_guide/ocr_predictor.md#restrict-the-output-characters)
of its vocabulary.

```python
from doctr.models import recognition_predictor

predictor = recognition_predictor("crnn_vgg16_bn", pretrained=True)
print(predictor.model.cfg["vocab"])
```

## Next steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book;1.1em` User guide
:link: ../user_guide/index
:link-type: doc

Every option of the OCR predictor, explained with examples.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Model zoo
:link: ../models/index
:link-type: doc

Compare the architectures and pick the right one for your documents.
:::
::::
