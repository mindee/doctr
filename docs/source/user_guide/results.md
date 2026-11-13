# Working with results

The OCR predictor returns a {class}`~doctr.io.Document`. This page shows how to navigate it, turn its geometries
into pixels, visualize the predictions, and save them for later.

:::{seealso}
[](../getting_started/concepts.md#the-document-tree) introduces the elements of the tree and
[](exporting.md) covers the text-oriented export formats (Markdown, HTML, hOCR...).
:::

## Navigate the tree

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True)
result = predictor(DocumentFile.from_pdf("invoice.pdf"))

page = result.pages[0]
page.dimensions  # (height, width) of the page image, in pixels: (1684, 1191)
page.page  # the page image itself, as a NumPy array
page.blocks  # list of Block
page.orientation  # {"value": ..., "confidence": ...} with detect_orientation=True
page.language  # {"value": ..., "confidence": ...} with detect_language=True
page.layout  # list of LayoutElement with detect_layout=True
page.tables  # list of Table with detect_tables=True
```

Most of the time, you want a flat list of words. A small generator does the job:

```python
def iter_words(document):
    for page_idx, page in enumerate(document.pages):
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    yield page_idx, word


words = [word.value for _, word in iter_words(result)]
```

Every element also has a `render()` method that returns its text: `line.render()` joins the words of a line with
spaces, `page.render()` returns the whole page.

## Filter by confidence

`word.confidence` tells how sure the recognition model is. Use it to send doubtful words to a human, or to drop
noise:

```python
doubtful = [(page_idx, word.value, word.confidence) for page_idx, word in iter_words(result) if word.confidence < 0.5]
```

:::{tip}
The way character probabilities are combined into `word.confidence` is configurable: see
[confidence scores](ocr_predictor.md#confidence-scores).
:::

## Convert coordinates to pixels

Geometries are relative to the page size, with `(0, 0)` at the top-left corner. Multiply them by the page
dimensions to get pixels:

::::{tab-set}

:::{tab-item} Straight boxes
```python
import numpy as np

height, width = page.dimensions
(xmin, ymin), (xmax, ymax) = word.geometry

x0, y0 = int(xmin * width), int(ymin * height)
x1, y1 = int(xmax * width), int(ymax * height)

word_image = page.page[y0:y1, x0:x1]  # crop the word from the page
```
:::

:::{tab-item} Rotated boxes
With `assume_straight_pages=False`, geometries are `(4, 2)` arrays of points:

```python
import numpy as np

height, width = page.dimensions
polygon = np.asarray(word.geometry) * np.array([width, height])  # (4, 2) pixel coordinates
```
:::
::::

## Find a value next to a keyword

Combining positions and text lets you build simple extraction rules. For instance, to read the amount written to
the right of *Total*:

```python
def value_right_of(page, keyword):
    words = [w for block in page.blocks for line in block.lines for w in line.words]
    anchor = next(w for w in words if w.value.lower().startswith(keyword.lower()))
    (_, ay0), (ax1, ay1) = anchor.geometry
    center_y = (ay0 + ay1) / 2
    # Words on the same horizontal band, to the right of the keyword, sorted left to right
    candidates = [w for w in words if w.geometry[0][1] < center_y < w.geometry[1][1] and w.geometry[0][0] > ax1]
    return " ".join(w.value for w in sorted(candidates, key=lambda w: w.geometry[0][0]))


print(value_right_of(result.pages[0], "Total"))  # due 1,250.00 EUR
```

For forms and invoices with many fields, a [KIE predictor](kie.md) trained on your fields is more robust than
hand-written rules.

## Visualize the predictions

::::{tab-set}

:::{tab-item} Interactive window
Requires the `viz` extra. Hover a box to see the word and its confidence:

```python
result.show()  # all pages; or result.pages[0].show()
```

When the layout model ran, its regions are drawn as well (`display_layout=False` hides them).
:::

:::{tab-item} Image file
{func}`~doctr.utils.visualization.visualize_page` returns a Matplotlib figure that you can save, e.g. on a
server without display:

```python
import matplotlib

matplotlib.use("Agg")  # no display needed
from doctr.utils.visualization import visualize_page

page = result.pages[0]
fig = visualize_page(page.export(), page.page, interactive=False, add_labels=False)
fig.savefig("predictions.png", bbox_inches="tight")
```
:::
::::

```{image} ../_static/images/examples/invoice_words.png
:alt: Detected words overlaid on the invoice
:width: 520px
:align: center
```

## Rebuild the page from its predictions

{meth}`~doctr.io.Page.synthesize` draws the recognized words at their position on a blank page. It is a quick
way to eyeball what was read, and where. Arabic and Hebrew words need the `rtl` extra
(`pip install "python-doctr[rtl]"`) to be drawn with their letters joined and in reading order, unless Pillow was
built with Raqm:

```python
import matplotlib.pyplot as plt

synthetic_pages = result.synthesize()  # one image per page
plt.imshow(synthetic_pages[0])
plt.axis("off")
plt.show()
```

```{image} ../_static/images/examples/invoice_synthesized.png
:alt: The invoice rebuilt from the predictions
:width: 520px
:align: center
```

## Save and reload

{meth}`~doctr.io.Document.export` converts the result into a nested dictionary of plain Python types, ready for
JSON. {meth}`~doctr.io.Document.from_dict` rebuilds the `Document` from it:

```python
import json

from doctr.io import Document

# Save
with open("invoice.json", "w") as f:
    json.dump(result.export(), f)

# ... later: reload
with open("invoice.json") as f:
    restored = Document.from_dict(json.load(f))

print(restored.render())
```

The page images are not part of the export. Pass them back with `Document.from_dict(data, pages=pages)` if you
need them, e.g. to call `show()`.

:::{dropdown} What does the JSON export look like?
```json
{
  "pages": [
    {
      "page_idx": 0,
      "dimensions": [1684, 1191],
      "orientation": {"value": null, "confidence": null},
      "language": {"value": null, "confidence": null},
      "blocks": [
        {
          "geometry": [[0.0706, 0.0537], [0.3992, 0.0732]],
          "objectness_score": 0.83,
          "lines": [
            {
              "geometry": [[0.0706, 0.0537], [0.3992, 0.0732]],
              "objectness_score": 0.83,
              "words": [
                {
                  "value": "ACME",
                  "confidence": 0.99995,
                  "geometry": [[0.0706, 0.0537], [0.1603, 0.0713]],
                  "objectness_score": 0.80,
                  "crop_orientation": {"value": 0, "confidence": null}
                }
              ]
            }
          ],
          "artefacts": []
        }
      ],
      "layout": [],
      "tables": []
    }
  ]
}
```
:::
