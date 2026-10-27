# Performance

docTR runs on the CPU out of the box. This page lists the levers to make it faster, from the most effective to the
most specialized.

## Run on a GPU

Predictors are PyTorch modules: move them to a device like any model. Inputs are moved automatically.

::::{tab-set}

:::{tab-item} NVIDIA (CUDA)
```python
import torch

from doctr.models import ocr_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = ocr_predictor(pretrained=True).to(device)
# or simply: ocr_predictor(pretrained=True).cuda()
```
:::

:::{tab-item} Apple Silicon (MPS)
```python
import torch

from doctr.models import ocr_predictor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
predictor = ocr_predictor(pretrained=True).to(device)
```
:::
::::

The same applies to every predictor: `detection_predictor`, `recognition_predictor`, `layout_predictor`,
`table_predictor`, `crop_orientation_predictor` and `page_orientation_predictor`.

## Half precision

On a GPU, half-precision inference uses 16-bit floats instead of 32-bit: it is faster and uses less memory, with
the same results in practice.

```python
from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True).cuda().bfloat16()  # or .half() for float16
result = predictor(pages)
```

| Format | Method | Notes |
| --- | --- | --- |
| **bfloat16** {bdg-success}`recommended` | `.bfloat16()` | Same range as float32, so no overflow. Requires an Ampere GPU or newer (compute capability 8.0+) |
| float16 | `.half()` | More precision but a much narrower range. Use it on older GPUs |

:::{note}
Half precision is only supported on GPU devices. The [CLI](../user_guide/cli.md) enables bfloat16 automatically on
compatible GPUs.
:::

## Batch sizes

The predictor processes pages and word crops in batches: `det_bs` pages at a time for detection (default `2`) and
`reco_bs` crops at a time for recognition (default `128`). On a GPU with free memory, larger batches are faster;
lower them if you run out of memory:

```python
predictor = ocr_predictor(pretrained=True, det_bs=8, reco_bs=512)
```

Pass all the pages of a document (or several documents) in **one call** so that batches are full.

## Skip what you don't need

- Keep `assume_straight_pages=True` (default) for upright documents: it avoids the orientation classifiers and
  the more expensive rotated-box post-processing.
- With rotated documents, `disable_page_orientation=True` and `disable_crop_orientation=True` skip the orientation
  classifiers when you know the text is horizontal (see [](../user_guide/rotated_documents.md)).
- Only enable `detect_layout`, `detect_tables`, `detect_orientation` and `detect_language` when you use them.
- Use lighter architectures, see the [model zoo](../models/index.md).

## Compile the models

[`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) converts the models into
optimized graphs. Compile the models first, then build the predictor with them:

```python
import torch

from doctr.models import (
    crop_orientation_predictor,
    fast_base,
    mobilenet_v3_small_crop_orientation,
    mobilenet_v3_small_page_orientation,
    ocr_predictor,
    page_orientation_predictor,
    vitstr_small,
)

detection_model = torch.compile(fast_base(pretrained=True).eval())
recognition_model = torch.compile(vitstr_small(pretrained=True).eval())

predictor = ocr_predictor(detection_model, recognition_model, assume_straight_pages=False)

# Only needed when the orientation classifiers are used (rotated documents)
predictor.crop_orientation_predictor = crop_orientation_predictor(
    torch.compile(mobilenet_v3_small_crop_orientation(pretrained=True).eval())
)
predictor.page_orientation_predictor = page_orientation_predictor(
    torch.compile(mobilenet_v3_small_page_orientation(pretrained=True).eval())
)

result = predictor(pages)
```

:::{important}
- The first calls are slow while the graphs are compiled: warm the predictor up before measuring.
- The `master` recognition architecture does not support compilation yet.
- Only the default `inductor` backend is officially supported.
:::

## CPU inference

- PyTorch uses all the cores by default. When several workers share a machine, limit each of them with
  `torch.set_num_threads(n)`.
- For the smallest footprint, export the models to ONNX and run them with [OnnxTR](onnx.md), optionally with
  8-bit quantized models.

## Measure

The `latency.py` scripts of the [references](https://github.com/mindee/doctr/tree/main/references) benchmark any
architecture on your hardware:

```bash
python references/detection/latency.py fast_base --it 100
```
