# Running OCR

{func}`~doctr.models.ocr_predictor` builds the end-to-end pipeline: it detects the words of each page, reads them,
and assembles the result into a {class}`~doctr.io.Document`. This page walks through its options, from choosing
the models to fine-tuning the output.

## Create a predictor

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True)

pages = DocumentFile.from_pdf("path/to/report.pdf")
result = predictor(pages)
```

Building a predictor loads the models (and downloads their weights the first time), so **create it once and reuse
it** for all your documents. The predictor runs in inference mode and accepts any number of pages per call.

## Choose the architectures

By default, `ocr_predictor` uses `fast_base` for text detection and `crnn_vgg16_bn` for text recognition. Pass the
names of other architectures to trade speed for accuracy:

```python
predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True)
```

| You want...                        | Detection (`det_arch`)                  | Recognition (`reco_arch`)                        |
| ---------------------------------- | --------------------------------------- | ------------------------------------------------ |
| A good default                     | `fast_base`                             | `crnn_vgg16_bn`                                  |
| Maximum speed (CPU, edge devices)  | `fast_tiny`, `db_mobilenet_v3_large`    | `crnn_mobilenet_v3_small`, `viptr_tiny`          |
| Maximum accuracy                   | `db_resnet50`, `fast_base`              | `parseq`, `vitstr_base`, `master`                 |

The [model zoo](../models/index.md) lists every architecture with its benchmark. Instead of a name, you can also pass
a model **instance**, which is how you plug in [your own trained models](../training/custom_models.md):

```python
from doctr.models import ocr_predictor, fast_base, parseq

predictor = ocr_predictor(
    det_arch=fast_base(pretrained=True),
    reco_arch=parseq(pretrained=True),
)
```

## Options at a glance

All options are keyword arguments of {func}`~doctr.models.ocr_predictor`.

::::{tab-set}

:::{tab-item} Page geometry
| Option                     | Default | Effect                                                                                                   |
| -------------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `assume_straight_pages`    | `True`  | Assume upright pages with horizontal text: fastest, returns straight boxes. Set to `False` for rotated or skewed documents. |
| `straighten_pages`         | `False` | Estimate the page rotation and straighten the pages before running the models                            |
| `export_as_straight_boxes` | `False` | With `assume_straight_pages=False`, still export straight boxes instead of polygons                      |
| `preserve_aspect_ratio`    | `True`  | Pad the pages instead of stretching them when resizing them for the models                               |
| `symmetric_pad`            | `True`  | With `preserve_aspect_ratio=True`, pad on all sides instead of bottom-right only                          |

See [](rotated_documents.md) for a detailed guide.
:::

:::{tab-item} Extra predictions
| Option               | Default      | Effect                                                                                              |
| -------------------- | ------------ | --------------------------------------------------------------------------------------------------- |
| `detect_orientation` | `False`      | Estimate the general orientation of each page, available as `page.orientation`                    |
| `detect_language`    | `False`      | Estimate the language of each page, available as `page.language`                                  |
| `detect_layout`      | `False`      | Run a layout model and attach its regions to each page (`page.layout`)                           |
| `layout_arch`        | `"lw_detr_s"`| Layout architecture name or model instance                                                          |
| `detect_tables`      | `False`      | Recognize the structure of tables (`page.tables`); enables the layout model                         |
| `ignore_regions`     | `None`       | Layout classes to mask out before OCR, e.g. `["Picture"]`; requires the layout model (`detect_layout=True`) |

See [](layout_and_tables.md).
:::

:::{tab-item} Output structure
| Option               | Default | Effect                                                                                  |
| -------------------- | ------- | --------------------------------------------------------------------------------------- |
| `resolve_lines`      | `True`  | Group words into lines                                                                  |
| `resolve_blocks`     | `False` | Group lines into blocks (by default, all the lines of a page form a single block)       |
| `paragraph_break`    | `0.035` | Minimum horizontal gap, relative to the page width, that splits two words into different lines |
| `keep_reading_order` | `False` | Sort the blocks of every page in reading order                                          |
:::

:::{tab-item} Performance
| Option                     | Default | Effect                                                                                  |
| -------------------------- | ------- | --------------------------------------------------------------------------------------- |
| `det_bs`                   | `2`     | Batch size of the detection model (also used by the layout and table models)            |
| `reco_bs`                  | `128`   | Batch size of the recognition model (number of word crops per batch)                   |
| `disable_page_orientation` | `False` | Skip the page orientation classifier (see [rotated documents](rotated_documents.md))     |
| `disable_crop_orientation` | `False` | Skip the word crop orientation classifier                                               |

See [](../deployment/performance.md) for GPU and half-precision inference.
:::
::::

## Tune the text detection

The detection models output a probability map that is binarized, then split into boxes. Two thresholds of the
post-processor control this step:

`bin_thresh`
: Minimum probability for a pixel to be considered as text. Lower it to catch faint text, raise it to separate words
  that are close to each other.

`box_thresh`
: Minimum mean score of a box to be kept. Lower it to keep more (uncertain) words, raise it to drop noise.

```python
predictor = ocr_predictor(pretrained=True)

predictor.det_predictor.model.postprocessor.bin_thresh = 0.3
predictor.det_predictor.model.postprocessor.box_thresh = 0.1
```

## Confidence scores

The recognition model predicts a probability for each character of a word. They are aggregated into
`word.confidence` with one of the following methods:

| Method                                | Behavior                                                                            |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| `"min"`                               | The least confident character (default for CRNN, VIPTR, SAR and MASTER)            |
| `"mean"`                              | Average probability (default for ViTSTR and PARSeq)                                 |
| `"geometric_mean"`, `"harmonic_mean"` | Means that penalize a single uncertain character more than the arithmetic mean      |
| `"median"`, `"max"`                   | Ignore most uncertain characters: they barely separate misread words from correct ones |
| a callable                            | Any function taking the 1D array of probabilities and returning a float             |

For the probabilities `[0.99, 0.98, 0.12, 0.99]` (one doubtful character), the mean is `0.77`, the geometric mean
`0.58`, the harmonic mean `0.35` and the minimum `0.12`. Pick a method that makes uncertain words stand out if
you use the confidence to flag words for human review.

```python
from doctr.models import ocr_predictor, parseq

# When building the predictor (modifies a model instance passed as `reco_arch`)
predictor = ocr_predictor(pretrained=True, confidence_aggregation="geometric_mean")

# Or afterwards, on the post-processor of the recognition model
predictor.reco_predictor.model.postprocessor.confidence_aggregation = "mean"

# Or on a model instance
predictor = ocr_predictor(reco_arch=parseq(pretrained=True, confidence_aggregation="min"), pretrained=True)
```

:::{dropdown} Details: which probabilities are aggregated?
Only the probabilities of the predicted characters: the CTC blank frames (CRNN, VIPTR) and the positions from the
end-of-sequence token onwards (SAR, MASTER) are ignored, and a CTC character has the highest probability of its
frames. An empty prediction has a confidence of `0`.

Very wide crops are split into parts that are read separately; their confidences are combined with the
`split_confidence_aggregation` of the recognition predictor (`"min"` by default):

```python
predictor.reco_predictor.split_confidence_aggregation = "mean"
```
:::

## Restrict the output characters

If you know which languages to expect, restrict the recognition model to their characters with
{func}`~doctr.models.utils.add_whitelist`. Forbidden characters can no longer be predicted, which removes a whole
class of errors (e.g. `ł` read instead of `l` on an English document).

```python
from doctr.datasets import VOCABS
from doctr.models import ocr_predictor
from doctr.models.utils import add_whitelist

predictor = ocr_predictor(pretrained=True)

# Only German and Polish characters can be predicted from now on
handle = add_whitelist(predictor, [VOCABS["german"], VOCABS["polish"]])
result = predictor(pages)

# Restore the original decoding
handle.remove()
```

The handle is also a context manager, which removes the whitelist when the block ends:

```python
with add_whitelist(predictor, VOCABS["german"]):
    result = predictor(pages)
```

By default, forbidden characters are dropped (`strategy="mask"`) and the model falls back to its best allowed
character. With `strategy="nearest"`, each forbidden character is first folded onto the closest allowed one,
which is handy to normalize accents onto a base alphabet:

```python
# é -> e, ł -> l, ß -> ss...: an ASCII-only output
handle = add_whitelist(predictor, VOCABS["latin"], strategy="nearest")
```

The closest character is found by transliteration by default. Pass `mapping="weights"` to derive it from the
model's own confusions instead, or a `{"forbidden": "allowed"}` dict to override specific characters.

:::{note}
A whitelist can only *restrict* a model: characters it was not trained on are silently ignored. To read a new
alphabet, use a model trained on a matching [vocabulary](../models/vocabs.md).
:::

## Hooks

Hooks let you edit the detected boxes **before they are read**. A hook is a callable that receives the location
predictions right after text detection and returns them, possibly modified. The recognition model then reads the
crops of the boxes the hook returned:

- **remove** boxes to skip regions you don't need (they are not read, which also saves time);
- **move or resize** boxes, e.g. to pad crops that are too tight;
- **add** boxes, e.g. for fields at a known position on a form.

The location predictions are a list with one NumPy array per page, in relative coordinates: `(N, 4)` straight boxes
as `xmin, ymin, xmax, ymax`, or `(N, 4, 2)` polygons with `assume_straight_pages=False`. A hook returns the same
structure, with any number of boxes per page.

::::{tab-set}

:::{tab-item} Skip headers & footers
```python
import numpy as np

from doctr.models import ocr_predictor


class SkipMargins:
    """Ignore the words in the top and bottom bands of the page (headers, footers)"""

    def __init__(self, band: float = 0.05):
        self.band = band

    def __call__(self, loc_preds: list[np.ndarray]) -> list[np.ndarray]:
        kept = []
        for boxes in loc_preds:  # one array per page, relative coordinates
            if boxes.ndim == 2:  # straight boxes (N, 4): xmin, ymin, xmax, ymax
                center_y = (boxes[:, 1] + boxes[:, 3]) / 2
            else:  # rotated boxes (N, 4, 2): polygons
                center_y = boxes[:, :, 1].mean(axis=1)
            keep = (center_y > self.band) & (center_y < 1 - self.band)
            kept.append(boxes[keep])
        return kept


predictor = ocr_predictor(pretrained=True)
predictor.add_hook(SkipMargins())
```

On the {download}`sample invoice <../_static/images/examples/invoice.pdf>`, the footer line
(`ACME Office Supplies GmbH - VAT ID ...`) is no longer read.
:::

:::{tab-item} Pad the boxes
```python
import numpy as np

from doctr.models import ocr_predictor


def pad_boxes(loc_preds: list[np.ndarray], pad: float = 0.002) -> list[np.ndarray]:
    """Enlarge the straight boxes slightly, so that the crops keep the character edges"""
    offsets = np.array([-pad, -pad, pad, pad])
    return [np.clip(boxes + offsets.astype(boxes.dtype), 0, 1) for boxes in loc_preds]


predictor = ocr_predictor(pretrained=True)
predictor.add_hook(pad_boxes)
```
:::
::::

Good to know:

- Hooks run in the order they were added, each one receiving the output of the previous one.
- Each returned box keeps the `objectness_score` of the closest detected box of its page; a box added on a page
  without any detection gets a score of `1`.
- With the [KIE predictor](kie.md), hooks receive and return a dict with one such list per class.
- An output in the wrong format (not one array per page, or boxes of another shape) raises a `ValueError`.

## Process many documents

- **Reuse the predictor**: building it is the expensive part.
- **Batch your pages**: one call on a list of pages is faster than one call per page, as the models process
  several pages and word crops at once (see `det_bs` and `reco_bs` above).
- **Open long documents lazily**: with `lazy=True`, pages are only decoded when they are processed, and the predictor
  works chunk by chunk (`page_chunk_size` pages at a time). A 1,000-page PDF or a folder of 100k images can then be
  passed in a single call (see [](loading_documents.md#long-documents-large-collections)).
- **Stream the results**: {meth}`~doctr.models.predictor.OCRPredictor.stream` yields the pages as soon as they are
  processed, so that the results can be written as you go.

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True, page_chunk_size=32)

# Every PDF and image of a folder, as one lazy document
doc = DocumentFile.from_files("invoices/", lazy=True)

for page in predictor.stream(doc):
    source_idx, page_idx = doc.locate(page.page_idx)
    print(f"input #{source_idx}, page {page_idx + 1}: {page.render()[:80]}")
```

Inputs longer than `page_chunk_size` pages, page sources and iterators (e.g. a generator of frames) are all processed
chunk by chunk, and `page_idx` is always the index of the page in the whole input. Hooks are therefore called once
per chunk, with the location predictions of its pages.

## Next steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`list-unordered;1.1em` Working with results
:link: results
:link-type: doc

Navigate the `Document`, read words and boxes, and save the predictions.
:::

:::{grid-item-card} {octicon}`rocket;1.1em` Performance
:link: ../deployment/performance
:link-type: doc

Run on GPU, in half precision, and compile the models.
:::
::::
