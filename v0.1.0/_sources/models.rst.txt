doctr.models
============

The full Optical Character Recognition task can be seen as two consecutive tasks: text detection and text recognition.
Either performed at once or separately, to each task corresponds a type of deep learning architecture.

.. currentmodule:: doctr.models

For a given task, DocTR provides a Predictor, which is composed of 3 components:

* PreProcessor: a module in charge of making inputs directly usable by the TensorFlow model.
* Model: a deep learning model, implemented with TensorFlow backend.
* PostProcessor: making model outputs structured and reusable.


Text Detection
--------------
Localizing text elements in images

.. list-table:: Text detection model zoo
   :widths: 20 20 15 10 10 10
   :header-rows: 1

   * - Architecture
     - Input shape
     - # params
     - Recall
     - Precision
     - FPS
   * - db_resnet50
     - (1024, 1024, 3)
     -
     -
     -
     -

Pre-processing for detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In DocTR, the pre-processing scheme for detection is the following:

1. resize each input image to the target size (bilinear interpolation by default) with potential deformation.
2. batch images together
3. normalize the batch using the training data statistics


Detection models
^^^^^^^^^^^^^^^^
Models expect a TensorFlow tensor as input and produces one in return. DocTR includes implementations and pretrained versions of the following models:

.. autofunction:: doctr.models.detection.db_resnet50


Post-processing outputs
^^^^^^^^^^^^^^^^^^^^^^^
The purpose of this block is to turn the model output (binary segmentation map for instance), into a set of bounding boxes.


Detection predictors
^^^^^^^^^^^^^^^^^^^^
Combining the right components around a given architecture for easier usage, predictors lets you pass numpy images as inputs and return structured information.

.. autofunction:: doctr.models.detection.db_resnet50_predictor


Text Recognition
----------------
Identifying strings in images

.. list-table:: Text recognition model zoo
   :widths: 20 20 15 10 10
   :header-rows: 1

   * - Architecture
     - Input shape
     - # params
     - Accuracy
     - FPS
   * - crnn_vgg16_bn
     - (32, 128, 3)
     -
     -
     -
   * - sar_vgg16_bn
     - (64, 256, 3)
     -
     -
     -


Pre-processing for recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In DocTR, the pre-processing scheme for recognition is the following:

1. resize each input image to the target size (bilinear interpolation by default) without deformation.
2. pad the image to the target size (with zeros by default)
3. batch images together
4. normalize the batch using the training data statistics

Recognition models
^^^^^^^^^^^^^^^^^^
Models expect a TensorFlow tensor as input and produces one in return. DocTR includes implementations and pretrained versions of the following models:


.. autofunction:: doctr.models.recognition.crnn_vgg16_bn
.. autofunction:: doctr.models.recognition.sar_vgg16_bn


Post-processing outputs
^^^^^^^^^^^^^^^^^^^^^^^
The purpose of this block is to turn the model output (symbol classification for the sequence), into a set of strings.

Recognition predictors
^^^^^^^^^^^^^^^^^^^^^^
Combining the right components around a given architecture for easier usage.

.. autofunction:: doctr.models.recognition.crnn_vgg16_bn_predictor
.. autofunction:: doctr.models.recognition.sar_vgg16_bn_predictor


End-to-End OCR
--------------
Predictors that localize and identify text elements in images

Two-stage approaches
^^^^^^^^^^^^^^^^^^^^
Those architectures involve one stage of text detection, and one stage of text recognition. The text detection will be used to produces cropped images that will be passed into the text recognition block.

.. autofunction:: doctr.models.zoo.ocr_db_crnn


Model export
------------
Utility functions to make the most of document analysis models.

.. currentmodule:: doctr.models.export


.. autofunction:: convert_to_tflite

.. autofunction:: convert_to_fp16

.. autofunction:: quantize_model
