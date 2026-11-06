# Task-specific predictors

The OCR predictor chains several models. Each of them is also available on its own, wrapped in a predictor that
handles pre-processing, batching and post-processing. Use them when you only need one step: locating text,
reading pre-cropped words, classifying orientations...

| Predictor                                              | Input                     | Output per input image                                 |
| ------------------------------------------------------ | ------------------------- | ------------------------------------------------------ |
| {func}`~doctr.models.detection.detection_predictor`                | Pages                     | `{class_name: array of boxes + scores}`                |
| {func}`~doctr.models.recognition.recognition_predictor`              | Word crops                | `(text, confidence)`                                   |
| {func}`~doctr.models.layout.layout_predictor`                   | Pages                     | `{"class_names", "boxes", "scores"}`                   |
| {func}`~doctr.models.table_structure.table_predictor`                    | Table images              | `{"cells", "num_rows", "num_cols"}`                    |
| {func}`~doctr.models.classification.page_orientation_predictor`         | Pages                     | Orientation class, angle and probability               |
| {func}`~doctr.models.classification.crop_orientation_predictor`         | Word crops                | Orientation class, angle and probability               |

All of them take a **list of images** (NumPy arrays of shape `(H, W, 3)`, `uint8`, RGB) and return one result per
image. They also accept a [lazy page source](loading_documents.md#long-documents--large-collections) or any iterable
of images (see [below](#long-inputs)). As with the OCR predictor, pass `pretrained=True` to load trained weights.

## Text detection

```python
from doctr.io import DocumentFile
from doctr.models import detection_predictor

predictor = detection_predictor("fast_base", pretrained=True)
pages = DocumentFile.from_pdf("invoice.pdf")
out = predictor(pages)

boxes = out[0]["words"]  # (N, 5): xmin, ymin, xmax, ymax, score (relative coordinates)
```

With `assume_straight_pages=False`, each box is a rotated polygon: the array has shape `(N, 5, 2)`, with the
four corners followed by the score. Multi-class detection models (see [](kie.md)) return one entry per class.

The predictor accepts the same input options as the OCR predictor: `assume_straight_pages`,
`preserve_aspect_ratio`, `symmetric_pad` and `batch_size`.

## Text recognition

The recognition predictor reads **one word per image**. Crops are resized to the input size of the model (32 px
high), so they should be tightly cropped around the word:

```python
from doctr.models import recognition_predictor

predictor = recognition_predictor("crnn_vgg16_bn", pretrained=True)
out = predictor([word_crop])  # e.g. the crop of "ACME" from the sample invoice

print(out)  # [('ACME', 0.9996)]
```

:::{note}
The pretrained vocabularies have no space character: feed the predictor single words, not whole lines. To read
lines or pages, use the OCR predictor, which detects the words first.
:::

Very wide crops are split into overlapping parts that are read separately and merged. `batch_size` (default
`128`) sets how many crops are processed at once.

## Layout analysis & table structure

See [](layout_and_tables.md#standalone-predictors).

## Orientation classifiers

Two lightweight classifiers predict the orientation of a page or a word crop among 0°, 90°, 180° and 270°:

```python
from doctr.models import crop_orientation_predictor, page_orientation_predictor

page_classifier = page_orientation_predictor(pretrained=True)
class_idxs, angles, probs = page_classifier([upright_page, rotated_page])
print(angles)  # [0, -90]

crop_classifier = crop_orientation_predictor(pretrained=True)
class_idxs, angles, probs = crop_classifier([word_crop, upside_down_crop])
print(angles)  # [0, 180]
```

They are used by the OCR predictor for [rotated documents](rotated_documents.md), and can be
[retrained on your data](../training/custom_models.md#orientation-classifiers).

## Long inputs

Like the OCR predictor, every task-specific predictor takes a lazy page source (`DocumentFile.from_pdf(...,
lazy=True)`, `from_images`, `from_files`), a slice of one, a tuple or a generator, and processes long inputs **chunk by
chunk**: `chunk_size` images are decoded and processed at a time, and the next chunk of a page source is decoded in a
background thread while the model runs on the current one. The memory footprint therefore does not grow with the
number of images, and the output is the same as for a list:

```python
from doctr.io import DocumentFile
from doctr.models import layout_predictor, recognition_predictor

# The layout regions of every page of a long PDF
predictor = layout_predictor("lw_detr_s", pretrained=True)
regions = predictor(DocumentFile.from_pdf("path/to/1000_pages.pdf", lazy=True))

# A folder of 100k word crops
predictor = recognition_predictor("crnn_vgg16_bn", pretrained=True, chunk_size=2048)
words = predictor(DocumentFile.from_images("path/to/word_crops/", lazy=True))
```

| Predictors                                                      | `chunk_size` default |
| --------------------------------------------------------------- | -------------------- |
| Detection, layout, table structure, page orientation            | `32` images          |
| Recognition, crop orientation (much smaller crops)              | `1024` crops         |

`chunk_size` is rounded up to a multiple of `batch_size`, so that the model processes the same batches as for a list,
and the predictions do not change (the recognition predictor splits very wide crops within each chunk, which can only
shift its batch boundaries). Lists and tuples of at most `chunk_size` images are processed at once, exactly as before.

:::{note}
Unlike the predictions of the OCR predictor, the outputs of the task-specific predictors are plain lists, built once
the whole input is processed. With `return_maps=True`, the detection predictor returns the segmentation map of every
page: on a long input, these maps add up.
:::

## Accessing the models

Every predictor exposes its model as `.model`, a regular PyTorch module:

```python
predictor = detection_predictor("db_resnet50", pretrained=True)

predictor.model  # the DBNet module
predictor.model.cfg  # its configuration (input shape, mean/std, url...)
predictor.model.postprocessor.bin_thresh  # post-processing parameters
```

The [Using standalone predictors](https://colab.research.google.com/github/mindee/notebooks/blob/main/doctr/using_standalone_predictors.ipynb)
notebook shows these predictors in action.
