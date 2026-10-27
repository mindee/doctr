# Using your trained models

Every predictor factory accepts a model **instance** instead of an architecture name. To use your checkpoint,
build the architecture with the same settings as for training, load the weights with `from_pretrained`, and pass
the model to the predictor. Pretrained and custom models can be mixed freely.

:::{tip}
`from_pretrained` accepts a local path or a URL. The `<experiment name>.json` file written next to the checkpoint
records the settings to rebuild the model with (vocabulary, class names...).
:::

## Detection and recognition models

::::{tab-set}

:::{tab-item} Custom detection
```python
from doctr.models import db_resnet50, ocr_predictor

det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
det_model.from_pretrained("path/to/detection.pt", map_location="cpu")

predictor = ocr_predictor(det_arch=det_model, reco_arch="crnn_vgg16_bn", pretrained=True)
```
:::

:::{tab-item} Custom recognition
```python
from doctr.models import crnn_vgg16_bn, ocr_predictor

reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
reco_model.from_pretrained("path/to/recognition.pt", map_location="cpu")

predictor = ocr_predictor(det_arch="fast_base", reco_arch=reco_model, pretrained=True)
```
:::

:::{tab-item} Both
```python
from doctr.models import crnn_vgg16_bn, db_resnet50, ocr_predictor

det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
det_model.from_pretrained("path/to/detection.pt")
reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
reco_model.from_pretrained("path/to/recognition.pt")

# pretrained=True still loads the pretrained orientation classifiers when they are needed
predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model, pretrained=True)
```
:::
::::

### Another vocabulary

A recognition model trained on another vocabulary than the default (`french`) must be built with that vocabulary,
so that its output layer matches the checkpoint:

```python
from doctr.datasets import VOCABS
from doctr.models import crnn_vgg16_bn, ocr_predictor

reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["german"])
reco_model.from_pretrained("path/to/recognition.pt")

predictor = ocr_predictor(reco_arch=reco_model, pretrained=True)
```

### KIE classes

A multi-class detection model is built with its class names, and used with the
[KIE predictor](../user_guide/kie.md):

```python
from doctr.models import db_resnet50, kie_predictor

det_model = db_resnet50(pretrained=False, pretrained_backbone=False, class_names=["total", "date"])
det_model.from_pretrained("path/to/detection.pt")

predictor = kie_predictor(det_arch=det_model, reco_arch="crnn_vgg16_bn", pretrained=True)
```

## Layout models

A layout model trained on your own classes is passed through `layout_arch`. The detected regions carry your class
names:

```python
from doctr.models import lw_detr_s, ocr_predictor

layout_model = lw_detr_s(pretrained=False, class_names=["heading", "paragraph", "figure", "table"])
layout_model.from_pretrained("path/to/layout.pt")

predictor = ocr_predictor(pretrained=True, detect_layout=True, layout_arch=layout_model)
result = predictor(pages)
print([(region.type, region.confidence) for region in result.pages[0].layout])
```

:::{note}
Table recognition (`detect_tables=True`) crops the regions labelled `Table`: keep this class name if your layout
model should feed the table model.
:::

## Table structure models

```python
from doctr.models import ocr_predictor, tablecenternet
from doctr.models.table_structure.zoo import table_predictor

table_model = tablecenternet(pretrained=False)
table_model.from_pretrained("path/to/table.pt")

predictor = ocr_predictor(pretrained=True, detect_tables=True)
predictor.table_predictor = table_predictor(table_model)  # replace the default table model
```

## Orientation classifiers

The orientation classifiers are used with `assume_straight_pages=False`, `straighten_pages=True` or
`detect_orientation=True`. Only `mobilenet_v3_small` based models are supported. Replace the default ones after
building the predictor:

```python
from doctr.models import (
    crop_orientation_predictor,
    mobilenet_v3_small_crop_orientation,
    mobilenet_v3_small_page_orientation,
    ocr_predictor,
    page_orientation_predictor,
)

page_model = mobilenet_v3_small_page_orientation(pretrained=False)
page_model.from_pretrained("path/to/page_orientation.pt")
crop_model = mobilenet_v3_small_crop_orientation(pretrained=False)
crop_model.from_pretrained("path/to/crop_orientation.pt")

predictor = ocr_predictor(pretrained=True, assume_straight_pages=False, detect_orientation=True)
predictor.page_orientation_predictor = page_orientation_predictor(page_model)
predictor.crop_orientation_predictor = crop_orientation_predictor(crop_model)
```

## Custom pre-processing

To change how images are resized and normalized (e.g. a model trained with another input size or other
statistics), assemble the predictors yourself:

```python
from doctr.models import crnn_vgg16_bn, db_resnet50
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.predictor import OCRPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models.recognition.predictor import RecognitionPredictor

det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
det_model.from_pretrained("path/to/detection.pt")
reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
reco_model.from_pretrained("path/to/recognition.pt")

det_predictor = DetectionPredictor(
    PreProcessor((1024, 1024), batch_size=1, mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)),
    det_model,
)
reco_predictor = RecognitionPredictor(
    PreProcessor(
        (32, 128), preserve_aspect_ratio=True, batch_size=32, mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301)
    ),
    reco_model,
)

predictor = OCRPredictor(det_predictor, reco_predictor)
```

## Models from the Hugging Face Hub

Models shared on the Hub are loaded with {func}`~doctr.models.factory.from_hub`, which restores the architecture and its
settings automatically. See [](sharing_models.md).
