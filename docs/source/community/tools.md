# Community tools

Projects built by the community on top of docTR. Built something useful? Open a pull request to add it to this page,
with a short description and what makes it unique.

## docTR Labeler

[github.com/text2knowledge/docTR-Labeler](https://github.com/text2knowledge/docTR-Labeler)

An annotation tool to create and correct OCR datasets for training and fine-tuning docTR models.

- **Polygon editing**: draw and edit polygons around text regions, with automatic shape correction
- **Auto-annotation**: suggestions and polygon refinement powered by OnnxTR
- **Fast workflow**: keyboard shortcuts for selection, zoom, undo/redo and saving
- **Flexible**: launch it with the `doctr-labeler` command or use its Python API
- **Private**: everything runs locally, no account needed

## OnnxTR

[github.com/felixdittrich92/OnnxTR](https://github.com/felixdittrich92/OnnxTR)

An [ONNX Runtime](https://onnxruntime.ai) backend for docTR models: the same interface, without PyTorch. See
[](../deployment/onnx.md).

- **Minimal dependencies**: no PyTorch or TensorFlow
- **Fast inference**: CPU, GPU and accelerators such as OpenVINO
- **Quantization**: 8-bit models for a smaller memory footprint
- **Familiar API**: `onnxtr.models.ocr_predictor`, just like docTR
- **Production ready**: Docker images and Hugging Face Hub integration

## docling-OCR-OnnxTR

[github.com/felixdittrich92/docling-OCR-OnnxTR](https://github.com/felixdittrich92/docling-OCR-OnnxTR)

A plugin that brings the OnnxTR OCR engine to the [Docling](https://github.com/docling-project/docling) document
parsing framework.

- **Native integration**: use it in Docling pipelines through `OnnxtrOcrOptions`
- **Drop-in replacement** for the other OCR engines of Docling
- **Configurable**: model choice, languages, confidence thresholds, batch sizes, orientation correction and 8-bit
  models
