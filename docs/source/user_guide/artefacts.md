# Artefact detection

The `doctr.contrib` module gathers complementary features that are useful in document pipelines but are not part
of the core OCR. Its first member, {class}`~doctr.contrib.ArtefactDetector`, finds non-textual elements on a page:
**bar codes, QR codes, logos and photos**.

:::{note}
Contrib modules run on [ONNX Runtime](https://onnxruntime.ai) instead of PyTorch. Install the extra first:

```bash
pip install "python-doctr[contrib]"
# or, for GPU inference: pip install onnxruntime-gpu
```
:::

## Detect artefacts

```python
from doctr.contrib import ArtefactDetector
from doctr.io import DocumentFile

pages = DocumentFile.from_images(["path/to/page.jpg"])

detector = ArtefactDetector(batch_size=2, conf_threshold=0.5, iou_threshold=0.5)
results = detector(pages)

for artefact in results[0]:  # one list per page
    print(artefact["label"], artefact["confidence"], artefact["box"])
# e.g. qr_code 0.91 [812, 64, 1003, 255]

detector.show()  # requires the `viz` extra
```

Each artefact is a dict with a `label`, a `confidence` and a `box` as `[xmin, ymin, xmax, ymax]` in **pixels**.

| Parameter         | Default | Description                                                    |
| ----------------- | ------- | -------------------------------------------------------------- |
| `batch_size`      | `2`     | Number of pages processed at once                              |
| `conf_threshold`  | `0.5`   | Minimum confidence of a detection                              |
| `iou_threshold`   | `0.5`   | Overlap above which duplicate detections are merged (NMS)      |

## Use your own YOLOv8 model

The detector is based on [YOLOv8](https://github.com/ultralytics/ultralytics). You can run any YOLOv8 detection
model exported to ONNX, e.g. one trained to find tables, stamps or signatures:

```python
from doctr.contrib import ArtefactDetector

detector = ArtefactDetector(model_path="path/to/your/model.onnx", labels=["table", "figure"])
```

:::{important}
The model must be exported to ONNX **with a dynamic batch size**. Oriented bounding box (OBB) models are not
supported.
:::
