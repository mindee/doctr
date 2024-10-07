Preparing your model for inference
==================================

A well-trained model is a good achievement but you might want to tune a few things to make it production-ready!

.. currentmodule:: doctr.models.utils


Model optimization
------------------

This section is meant to help you perform inference with optimized versions of your model.


Half-precision
^^^^^^^^^^^^^^

**NOTE:** We support half-precision inference for PyTorch and TensorFlow models only on **GPU devices**.

Half-precision (or FP16) is a binary floating-point format that occupies 16 bits in computer memory.

Advantages:

- Faster inference
- Less memory usage

.. tabs::

    .. tab:: TensorFlow

        .. code:: python3

            import tensorflow as tf
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            predictor = ocr_predictor(reco_arch="crnn_mobilenet_v3_small", det_arch="linknet_resnet34", pretrained=True)

    .. tab:: PyTorch

        .. code:: python3

            import torch
            predictor = ocr_predictor(reco_arch="crnn_mobilenet_v3_small", det_arch="linknet_resnet34", pretrained=True).cuda().half()
            res = predictor(doc)


Export to ONNX
^^^^^^^^^^^^^^

ONNX (Open Neural Network Exchange) is an open and interoperable format for representing and exchanging machine learning models.
It defines a common format for representing models, including the network structure, layer types, parameters, and metadata.

.. tabs::

    .. tab:: TensorFlow

        .. code:: python3

            import tensorflow as tf
            from doctr.models import vitstr_small
            from doctr.models.utils import export_model_to_onnx

            batch_size = 16
            input_shape = (3, 32, 128)
            model = vitstr_small(pretrained=True, exportable=True)
            dummy_input = [tf.TensorSpec([batch_size, input_shape], tf.float32, name="input")]
            model_path, output = export_model_to_onnx(model, model_name="vitstr.onnx", dummy_input=dummy_input)


    .. tab:: PyTorch

        .. code:: python3

            import torch
            from doctr.models import vitstr_small
            from doctr.models.utils import export_model_to_onnx

            batch_size = 16
            input_shape = (32, 128, 3)
            model = vitstr_small(pretrained=True, exportable=True)
            dummy_input = torch.rand((batch_size, input_shape), dtype=torch.float32)
            model_path = export_model_to_onnx(model, model_name="vitstr.onnx, dummy_input=dummy_input)


Using your ONNX exported model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use your exported model, we have build a dedicated lightweight package called `OnnxTR <https://github.com/felixdittrich92/OnnxTR>`_.
The package doesn't require PyTorch or TensorFlow to be installed - build on top of ONNXRuntime.
It is simple and easy-to-use (with the same interface you know already from docTR), that allows you to perform inference with your exported model.

- `Installation <https://github.com/felixdittrich92/OnnxTR#installation>`_
- `Loading custom exported model <https://github.com/felixdittrich92/OnnxTR#loading-custom-exported-models>`_

.. code:: shell

    pip install onnxtr[cpu]

.. code:: python3

    from onnxtr.io import DocumentFile
    from onnxtr.models import ocr_predictor, parseq, linknet_resnet18
    # Load your documents
    single_img_doc = DocumentFile.from_images("path/to/your/img.jpg")

    # Load your exported model/s
    reco_model = parseq("path_to_custom_model.onnx", vocab="ABC")
    det_model = linknet_resnet18("path_to_custom_model.onnx")
    predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
    # Or use any of the pre-trained models
    predictor = ocr_predictor(det_arch="linknet_resnet18", reco_arch="parseq")

    # Get your results
    res = predictor(single_img_doc)
