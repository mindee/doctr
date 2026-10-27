# ONNX & OnnxTR

[ONNX](https://onnx.ai) is an open format for machine learning models, supported by many runtimes and hardware
vendors. Exporting docTR models to ONNX lets you run them **without PyTorch**, with a smaller footprint and
often a faster CPU inference.

## Export a model

{func}`~doctr.models.utils.export_model_to_onnx` traces a model with a dummy input. Build the model with
`exportable=True`, which makes it return raw outputs instead of post-processed predictions:

```python
import torch

from doctr.models import vitstr_small
from doctr.models.utils import export_model_to_onnx

model = vitstr_small(pretrained=True, exportable=True)  # or your own fine-tuned model

batch_size = 1
input_shape = (3, 32, 128)  # (C, H, W), the input shape of the model
dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)

model_path = export_model_to_onnx(model, model_name="vitstr_small", dummy_input=dummy_input)
```

The batch dimension of the exported model is dynamic. Find the input shape of any model in `model.cfg["input_shape"]`.

## Run it with OnnxTR

[OnnxTR](https://github.com/felixdittrich92/OnnxTR) is a lightweight package built on
[ONNX Runtime](https://onnxruntime.ai), with the same interface as docTR. It ships ONNX versions of the pretrained
models and loads your exported ones:

::::{tab-set}

:::{tab-item} CPU
```bash
pip install "onnxtr[cpu]"
```
:::

:::{tab-item} NVIDIA GPU
```bash
pip install "onnxtr[gpu]"
```
:::

:::{tab-item} Intel (OpenVINO)
```bash
pip install "onnxtr[openvino]"
```
:::
::::

```python
from onnxtr.io import DocumentFile
from onnxtr.models import linknet_resnet18, ocr_predictor, parseq

pages = DocumentFile.from_images("path/to/page.jpg")

# Pretrained models, exactly like in docTR
predictor = ocr_predictor(det_arch="linknet_resnet18", reco_arch="parseq")

# Or your exported models
det_model = linknet_resnet18("path/to/detection.onnx")
reco_model = parseq("path/to/recognition.onnx", vocab="ABC")  # the vocabulary used for training
predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model)

result = predictor(pages)
```

See the OnnxTR documentation for [installation options](https://github.com/felixdittrich92/OnnxTR#installation)
and [loading custom models](https://github.com/felixdittrich92/OnnxTR#loading-custom-exported-models).
