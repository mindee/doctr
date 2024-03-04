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


Using your ONNX exported model in docTR
---------------------------------------

**Coming soon**
