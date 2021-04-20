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

+---------------------------------------------------+----------------------------+----------------------------+---------+
|                                                   |        FUNSD               |        CORD                |         |
+==================+=================+==============+============+===============+============+===============+=========+
| **Architecture** | **Input shape** | **# params** | **Recall** | **Precision** | **Recall** | **Precision** | **FPS** |
+------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+
| db_resnet50      | (1024, 1024, 3) |              |   0.733    |     0.817     |   0.745    |     0.875     |   2.1   |
+------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+

All text detection models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combine have 199 pages which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed this way: we instantiate the model, we feed the model with 100 random tensors of shape [1, 1024, 1024, 3] as a warm-up. Then, we measure the average speed of the model on 1000 batches of 1 frame (random tensors of shape [1, 1024, 1024, 3]).
We used a c5.x12large from AWS instances (CPU Xeon Platinum 8275L) to perform experiments.

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
.. autofunction:: doctr.models.detection.linknet


Post-processing detections
^^^^^^^^^^^^^^^^^^^^^^^^^^
The purpose of this block is to turn the model output (binary segmentation map for instance), into a set of bounding boxes.


Detection predictors
^^^^^^^^^^^^^^^^^^^^
Combining the right components around a given architecture for easier usage, predictors lets you pass numpy images as inputs and return structured information.

.. autofunction:: doctr.models.detection.detection_predictor


Text Recognition
----------------
Identifying strings in images

.. list-table:: Text recognition model zoo
   :widths: 20 20 15 10 10 10
   :header-rows: 1

   * - Architecture
     - Input shape
     - # params
     - FUNSD
     - CORD
     - FPS
   * - crnn_vgg16_bn
     - (32, 128, 3)
     -
     - 0.860
     - 0.913
     - 12.8
   * - sar_vgg16_bn
     - (32, 128, 3)
     -
     - 0.862
     - 0.917
     - 3.3
   * - sar_resnet31
     - (32, 128, 3)
     -
     - **0.863**
     - **0.921**
     - 2.7

All text recognition models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combine have 30595 word-level crops which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed this way: we instantiate the model, we feed the model with 100 random tensors of shape [1, 32, 128, 3] as a warm-up. Then, we measure the average speed of the model on 1000 batches of 1 frame (random tensors of shape [1, 32, 128, 3]).
We used a c5.x12large from AWS instances (CPU Xeon Platinum 8275L) to perform experiments.

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
.. autofunction:: doctr.models.recognition.sar_resnet31

Post-processing outputs
^^^^^^^^^^^^^^^^^^^^^^^
The purpose of this block is to turn the model output (symbol classification for the sequence), into a set of strings.

Recognition predictors
^^^^^^^^^^^^^^^^^^^^^^
Combining the right components around a given architecture for easier usage.

.. autofunction:: doctr.models.recognition.recognition_predictor


End-to-End OCR
--------------
Predictors that localize and identify text elements in images

+--------------------------------------------------------------+--------------------------------------+--------------------------------------+
|                                                              |                  FUNSD               |                  CORD                |
+=============================+=================+==============+============+===============+=========+============+===============+=========+
|      **Architecture**       | **Input shape** | **# params** | **Recall** | **Precision** | **FPS** | **Recall** | **Precision** | **FPS** |
+-----------------------------+-----------------+--------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + crnn_vgg16_bn | (1024, 1024, 3) |              |   0.629    |     0.701     |  0.85   |    0.664   |     0.780     |   1.6   |
+-----------------------------+-----------------+--------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + sar_vgg16_bn  | (1024, 1024, 3) |              |   0.630    |     0.702     |  0.49   |    0.666   |     0.783     |   1.0   |
+-----------------------------+-----------------+--------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + sar_resnet31  | (1024, 1024, 3) |              |   0.640    |     0.713     |  0.27   |    0.672   |   **0.789**   |  0.83   |
+-----------------------------+-----------------+--------------+------------+---------------+---------+------------+---------------+---------+
| Gvision text detection      |        NA       |              |   0.595    |     0.625     |         |    0.753   |     0.700     |         |
+-----------------------------+-----------------+--------------+------------+---------------+---------+------------+---------------+---------+
| Gvision doc. text detection |        NA       |              |   0.640    |     0.533     |         |    0.689   |     0.611     |         |
+-----------------------------+-----------------+--------------+------------+---------------+---------+------------+---------------+---------+
| aws textract                |        NA       |              | **0.781**  |   **0.830**   |         |  **0.875** |     0.660     |         |
+-----------------------------+-----------------+--------------+------------+---------------+---------+------------+---------------+---------+

All OCR models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combine have 199 pages which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed this way: we instantiate the predictor, we warm-up the model and then we measure the average speed of the end-to-end predictor on the datasets, with a batch size of 1.
We used a c5.x12large from AWS instances (CPU Xeon Platinum 8275L) to perform experiments.

Two-stage approaches
^^^^^^^^^^^^^^^^^^^^
Those architectures involve one stage of text detection, and one stage of text recognition. The text detection will be used to produces cropped images that will be passed into the text recognition block.

.. autofunction:: doctr.models.zoo.ocr_predictor


Model export
------------
Utility functions to make the most of document analysis models.

.. currentmodule:: doctr.models.export


.. autofunction:: convert_to_tflite

.. autofunction:: convert_to_fp16

.. autofunction:: quantize_model
