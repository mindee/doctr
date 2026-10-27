# Key information extraction

The OCR predictor finds *all* the words of a page. Often you only care about a few fields: the date, the total
amount, the invoice number... The **KIE predictor** (Key Information Extraction) uses a detection model trained on
several classes, so that each detected box comes with the *type* of information it contains.

## How it differs from OCR

| | {func}`~doctr.models.ocr_predictor` | {func}`~doctr.models.kie_predictor` |
| --- | --- | --- |
| Detection model | One class: `words` | Any number of classes (`date`, `total`, `address`...) |
| Output | {class}`~doctr.io.Document` of pages, blocks, lines and words | {class}`~doctr.io.KIEDocument` of pages with predictions grouped **by class** |
| Typical use | Read everything | Extract specific fields |

## Run a KIE predictor

The KIE predictor takes the same arguments as the OCR predictor (architectures, orientation, layout...):

```python
from doctr.io import DocumentFile
from doctr.models import kie_predictor

predictor = kie_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
result = predictor(DocumentFile.from_pdf("invoice.pdf"))

page = result.pages[0]
for class_name, predictions in page.predictions.items():
    for prediction in predictions:
        print(class_name, prediction.value, prediction.confidence, prediction.geometry)
```

Each {class}`~doctr.io.Prediction` is a word with an extra `crop_orientation` field. `render()` and the exports
group the predictions by class:

```python
print(result.render())
# words: ACME
#
# words: Office
# ...
```

:::{important}
The pretrained detection models only know the `words` class, so out of the box the KIE predictor behaves like the
OCR predictor with a flat output. Its value comes with a detection model **trained on your own classes**.
:::

## Use your own classes

1. Annotate your documents with one polygon list per class (see the
   [multi-class detection format](../training/train.md#text-detection)), e.g. with the
   [docTR Labeler](../community/tools.md#doctr-labeler).
2. Train a detection model with these classes using the
   [detection training script](../training/train.md#text-detection).
3. Load it with its class names and plug it into the KIE predictor:

```python
from doctr.models import db_resnet50, kie_predictor

det_model = db_resnet50(pretrained=False, pretrained_backbone=False, class_names=["date", "total", "invoice_id"])
det_model.from_pretrained("path/to/your/checkpoint.pt")

predictor = kie_predictor(det_arch=det_model, reco_arch="crnn_vgg16_bn", pretrained=True)
result = predictor(pages)

dates = [prediction.value for prediction in result.pages[0].predictions["date"]]
```

## Visualize and export

{class}`~doctr.io.KIEDocument` offers the same tools as `Document`: `show()` draws the boxes with one color per
class, `export()` returns a JSON-ready dict with a `predictions` key per page, and the
[export formats](exporting.md) are available as well.
