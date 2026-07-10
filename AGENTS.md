# Notes for AI coding agents

This file collects behaviors that are easy for an automated agent (or a
human moving quickly) to miss, based on hands-on verification on an
NVIDIA T4 (Turing, 16GB).

## GPU is opt-in, with no warning

`ocr_predictor(...)` (and the standalone `detection_predictor` /
`recognition_predictor`) return a plain `nn.Module`-based predictor that stays
on **CPU by default** — including on machines where
`torch.cuda.is_available()` is `True`. There is no warning when this happens.
You must explicitly move it yourself:

```python
model = ocr_predictor(pretrained=True).cuda()  # or .to(device)
```

Measured on a T4: CPU 2.657s/img → GPU (fp32) 0.155s/img (**17.16x**).

There is no `device=` keyword on `ocr_predictor` / `_predictor`. Passing one
(e.g. `ocr_predictor(pretrained=True, device="cuda")`, a natural guess) raises:

```
TypeError: DocumentBuilder.__init__() got an unexpected keyword argument 'device'
```

The error surfaces from an internal class (`DocumentBuilder`), not from
`ocr_predictor` itself, which makes the root cause harder to trace. The
standard PyTorch `nn.Module` API (`.cuda()` / `.to(device)`) is the only
supported way to select a device; `scripts/analyze.py` also has no
`--device`/`--cuda` flag and is CPU-only unless you edit it.

## `.half()` can silently corrupt recognition output (CRNN only)

`.half()` (fp16) is officially documented for inference speedup. On a T4,
applying it to the **default README recognizer, `crnn_vgg16_bn`**
(`nn.LSTM` decoder) silently corrupts recognition output — no error, no
warning, and the corruption is deterministic (100% reproducible across
repeated runs), so it's easy to mistake for a real, faster result:

```
fp32: ['Mr.', 'Anjum', 'Hameed,', 'We', 'are', 'pleased', 'inform', 'that', 'your', 'salary']
fp16: ['Mr',  'Anjnn', 'nra,',    'Wn', 'arn', 'plansa', 'infnrn', 'tt',   'your', 'salay']
```

Cross-checked with the same detector/image against the transformer-based
recognizers `master` and `parseq`: both produced output identical to fp32
under `.half()`. The corruption appears specific to CRNN's `nn.LSTM` decoder,
not to fp16 inference generally.

**Before using `.half()` for recognition, confirm your `reco_arch` is not a
CRNN variant, or validate the output yourself.**

## No bf16 option (not a T4 trap here)

The library only documents fp32/fp16 for inference; there is no bf16 code
path (`torch.bfloat16` does not appear in the codebase). Nothing to trip on
for this axis.

## Attention has no backend to misconfigure

`master` / `parseq` / ViTSTR all use a self-contained matmul+softmax
attention implementation (`doctr/models/modules/transformer/pytorch.py`),
not `torch.nn.functional.scaled_dot_product_attention` or
`nn.MultiheadAttention`. There's no SDPA/flash-attention backend selection,
so there's no flash-attn-unsupported-GPU crash risk — but also no backend
to switch for a speedup.

## int8 quantization is out of scope for this repo

docTR itself only supports exporting to ONNX (`export_model_to_onnx()`).
Quantized runtime inference is handled by the sister project
[OnnxTR](https://github.com/felixdittrich92/OnnxTR), not by docTR.

## Developer-mode install can silently disable CUDA

`pip install -e doctr/.` (per README "Developer mode") only constrains
`torch>=2.0.0,<3.0.0`, with no CUDA index pinned. If PyPI resolves a torch
build newer than what your GPU driver supports, `torch.cuda.is_available()`
returns `False` with no exception — you simply run on CPU. Reinstalling with
the CUDA index that matches your driver (e.g.
`pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124`)
fixes it.
