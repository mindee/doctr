doctr.models
=============

The full Optical Character Recognition task can be seen as two consecutive tasks: text detection and text recognition.
Either performed at once or separately, to each task corresponds a type of deep learning architecture.

.. currentmodule:: doctr.models

For a given task, DocTR provides a Predictor, which is composed of 3 components:

* PreProcessor: a module in charge of making inputs directly usable by the TensorFlow model.
* Model: a deep learning model, implemented with TensorFlow backend.
* PostProcessor: making model outputs structured and reusable.


Pre-processing
--------------
Operations that need to be carried out before passing data to actual models

.. currentmodule:: doctr.models.preprocessor

.. autoclass:: PreProcessor


Text Detection
--------------
Architectures to localize text elements

.. autofunction:: doctr.models.detection.db_resnet50


Text Recognition
----------------
Architectures to identify strings inside the localized boxes

.. autofunction:: doctr.models.recognition.crnn_vgg16_bn
.. autofunction:: doctr.models.recognition.sar_vgg16_bn


Model export
------------
Utility functions to make the most of document analysis models.

.. currentmodule:: doctr.models.export


.. autofunction:: convert_to_tflite

.. autofunction:: convert_to_fp16

.. autofunction:: quantize_model
