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

### Formula recognition

:::{important}
`formulanet` must be exported with `dynamo=True` (requires `pip install onnxscript`). The default exporter
(`dynamo=False`) cannot express its decoding loop, which stops once every formula of the batch is complete: it unrolls
`max_decode_length` steps (2048 by default), a graph far too large to be loaded.
:::

With `dynamo=True`, the decoding is exported as ONNX loops, which stop at the end of the formulas as in PyTorch:

```python
from doctr.models import formulanet

model = formulanet(pretrained=True, exportable=True)
dummy_input = torch.rand((1, *model.cfg["input_shape"]), dtype=torch.float32)

model_path = export_model_to_onnx(model, model_name="formulanet", dummy_input=dummy_input, dynamo=True)
```

The exported logits always have `max_decode_length + 1` steps, with zeros after the end of every formula.

The autoregressive text recognition models `parseq` and `master` can also be exported with `dynamo=True`: their
decoding then stops at the end of the words (up to 20% faster for `master` on CPU, but about 2x slower on GPU with
ONNX Runtime), and the `master` logits are zeros after the end of every word, as in PyTorch. With the default exporter,
they always run the full decoding length.

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
