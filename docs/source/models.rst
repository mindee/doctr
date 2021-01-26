doctr.models
=============


.. currentmodule:: doctr.models


Pre-processing
--------------
Operations that need to be carried out before passing data to actual models

.. currentmodule:: doctr.models.preprocessor

.. autoclass:: Preprocessor


Text Detection
--------------
Architectures to localize text elements

.. autoclass:: doctr.models.detection.DBResNet50


Text Recognition
----------------
Architectures to identify strings inside the localized boxes

.. autoclass:: doctr.models.recognition.CRNN
.. autoclass:: doctr.models.recognition.SAR

Model export
------------
Utility functions to make the most of document analysis models.

.. currentmodule:: doctr.models.utils


.. autofunction:: convert_to_tflite

.. autofunction:: convert_to_fp16

.. autofunction:: quantize_model
