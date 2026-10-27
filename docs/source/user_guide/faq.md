# Troubleshooting & FAQ

## Installation and setup

:::{dropdown} `ModuleNotFoundError: No module named 'matplotlib'` when calling `.show()`
:open:

Visualization relies on optional dependencies. Install the `viz` extra:

```bash
pip install "python-doctr[viz]"
```

Similarly, `DocumentFile.from_url` needs the `html` extra and the `contrib` module needs the `contrib` extra
(see [](../getting_started/installation.md#optional-extras)).
:::

:::{dropdown} The pretrained weights cannot be downloaded
Weights are downloaded on first use. Behind a proxy or on a machine without internet access, download them once
on a connected machine and copy the cache directory (`~/.cache/doctr` by default), or point `DOCTR_CACHE_DIR` to
a shared location. See [](../deployment/configuration.md).
:::

:::{dropdown} Errors in a restricted environment (AWS Lambda, read-only file systems)
docTR writes its model cache to `~/.cache/doctr` and uses multiprocessing for some data operations. Set
`DOCTR_CACHE_DIR` to a writable directory and `DOCTR_MULTIPROCESSING_DISABLE=TRUE`
(see [](../deployment/configuration.md#aws-lambda)).
:::

## Inputs

:::{dropdown} `ValueError: incorrect input shape`
Predictors expect a **list** of pages, each a 3-channel array of shape `(H, W, 3)`. Wrap a single page in a list
(`predictor([page])`) and convert grayscale images to 3 channels:

```python
import numpy as np

page = np.stack([gray] * 3, axis=-1)
```
:::

:::{dropdown} The results are wrong on images loaded with OpenCV
OpenCV loads images in **BGR** order, while docTR expects **RGB**. Convert them first, or use
`DocumentFile.from_images`, which handles it for you:

```python
import cv2

page = cv2.cvtColor(cv2.imread("scan.jpg"), cv2.COLOR_BGR2RGB)
```
:::

:::{dropdown} Does docTR use the text layer of digital PDFs?
No: every page is rendered to an image and read by the models, so scanned and digital-born PDFs are processed the
same way. If your PDFs have a reliable text layer and you only need the text, a PDF text extractor is faster.
:::

## Accuracy

:::{dropdown} Some words are missing
- Small print: render PDFs at a higher resolution, e.g. `DocumentFile.from_pdf(path, scale=3)`.
- Faint or low-contrast text: lower the detection thresholds (`bin_thresh`, `box_thresh`), see
  [](ocr_predictor.md#tune-the-text-detection).
- Very large images: keep `preserve_aspect_ratio=True` (default) so the page is not distorted.
:::

:::{dropdown} The output contains odd characters (accents, symbols) that are not in my documents
Restrict the recognition model to the characters you expect with a whitelist, see
[](ocr_predictor.md#restrict-the-output-characters).
:::

:::{dropdown} My documents are in a language with another alphabet
The pretrained recognition models are trained on the `french` vocabulary (Latin script). For other scripts, look for
a [community model](../training/sharing_models.md#community-models) on the Hugging Face Hub, or
[train a model](../training/train.md) on a matching [vocabulary](../models/vocabs.md).
:::

:::{dropdown} The text is read in the wrong order
Use an export method (`render()`, Markdown, HTML...), which applies the reading-order algorithm, rather than
iterating the blocks. Enable `detect_layout=True` for complex layouts, and set `direction` if the automatic
detection fails. See [](exporting.md#reading-order).
:::

:::{dropdown} Rotated pages give poor results
The default mode assumes upright pages. Use `straighten_pages=True` or `assume_straight_pages=False`
(see [](rotated_documents.md)).
:::

## Performance

:::{dropdown} OCR is slow
- Run on a GPU, and use half precision (see [](../deployment/performance.md)).
- Create the predictor **once** and reuse it.
- Pass several pages per call and increase `det_bs` / `reco_bs` if your hardware allows it.
- Try lighter architectures (`fast_tiny`, `crnn_mobilenet_v3_small`), see the [model zoo](../models/index.md).
- Keep `assume_straight_pages=True` when your pages are upright.
:::

:::{dropdown} I run out of GPU memory
Lower the batch sizes (`det_bs`, `reco_bs`), process fewer pages per call, or render PDFs at a lower `scale`.
:::

## Still stuck?

Search the [GitHub issues](https://github.com/mindee/doctr/issues) and
[discussions](https://github.com/mindee/doctr/discussions), or ask the community (see [](../community/index.md)).
When opening an issue, include the output of:

```bash
python scripts/collect_env.py
```
