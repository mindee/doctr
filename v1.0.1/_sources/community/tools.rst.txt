Community Tools
===============

This section highlights notable tools developed by the docTR community.


docTR-Labeler
-------------

:Link: https://github.com/text2knowledge/docTR-Labeler

**Overview**

``docTR-Labeler`` is a dedicated annotation tool tailored for creating and editing OCR datasets to train and fine-tune docTR models. It offers a user-friendly graphical interface, featuring polygon-based text labeling, automatic annotation suggestions via OnnxTR, and convenient label export capabilities.

**Key Features**

* Interactive Polygon Editing: Draw and edit polygons around text regions with precision
* AI-Powered Auto-Annotation: Automatic annotation suggestions and polygon refinement powered by OnnxTR
* Auto-Correction: Automatic correction of polygon shapes to ensure accurate text region representation
* Efficient Workflow: Keyboard shortcuts for selection, zooming, undo/redraw, and saving operations
* Flexible Access: CLI launch with ``doctr-labeler`` command and full programmatic Python API integration
* Privacy-First: No authentication required - everything runs locally on your machine
* Real-Time Rendering: Live image rendering with helpful visual feedback


OnnxTR
------

:Link: https://github.com/felixdittrich92/OnnxTR

**Overview**

``OnnxTR`` provides an ONNX-based backend for docTR models, enabling fast, cross-platform inference using ONNX Runtime. It's a core refactored library that enhances the performance and flexibility of OCR tasks without relying on heavy frameworks like PyTorch or TensorFlow.

**Key Features**

* Minimal Dependencies: No PyTorch or TensorFlow requirements
* Fast Inference: Optimized with ONNX Runtime for production environments
* Quantization Support: Reduced memory usage and faster inference through model quantization
* Batch Processing: Efficient batch inference capabilities
* Multi-Platform: CPU, GPU, and specialized accelerator runtimes like OpenVINO
* Flexible Installation: Separate install options for different runtime requirements
* Familiar API: One-line inference via ``onnxtr.models.ocr_predictor`` (similar to docTR)
* Docker Ready: Production-ready Docker images available
* Hugging Face Integration: Seamless model sharing and loading
* Server Optimized: OpenCV headless installation options for server environments


docling-OCR-OnnxTR
------------------

:Link: https://github.com/felixdittrich92/docling-OCR-OnnxTR

**Overview**

``docling-OCR-OnnxTR`` is a high-performance plugin that integrates the OnnxTR OCR engine into the Docling document parsing framework. By leveraging ONNX Runtime, it delivers superior accuracy and efficiency compared to traditional OCR engines across various hardware configurations.

**Key Features**

* Native Docling Support: Direct integration with Docling pipelines using ``OnnxtrOcrOptions``
* Drop-in Replacement: Easy migration from existing OCR engines
* Model Selection: Control over detection and recognition model choices
* Multi-Language Support: Configurable language settings
* Quality Control: Adjustable confidence thresholds
* Performance Tuning: Batch size optimization
* Enhanced Processing: Orientation correction and 8-bit model loading options


Contribute Your Tool
--------------------

**Share Your Innovation**

Have you built something amazing on top of docTR ?

We'd love to showcase your work! Whether it's a useful plugin, dataset preparation tool, or any other docTR-based project, the community would benefit from learning about it.

**How to Contribute**

To contribute your tool to the docTR community, please follow these steps:

1. **GitHub**: Open a pull request with your tool information
2. **Format**: Follow the structure above with clear descriptions and key features


.. tip::
   Include a clear tool description and highlight what makes your tool unique or particularly useful to the docTR community.

   This helps others quickly understand its value and how to use it effectively.
