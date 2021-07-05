doctr.models
============

The full Optical Character Recognition task can be seen as two consecutive tasks: text detection and text recognition.
Either performed at once or separately, to each task corresponds a type of deep learning architecture.

.. currentmodule:: doctr.models

For a given task, DocTR provides a Predictor, which is composed of 2 components:

* PreProcessor: a module in charge of making inputs directly usable by the TensorFlow model.
* Model: a deep learning model, implemented with TensorFlow backend along with its specific post-processor to make outputs structured and reusable.


Text Detection
--------------
Localizing text elements in images

+---------------------------------------------------+----------------------------+----------------------------+---------+
|                                                   |        FUNSD               |        CORD                |         |
+==================+=================+==============+============+===============+============+===============+=========+
| **Architecture** | **Input shape** | **# params** | **Recall** | **Precision** | **Recall** | **Precision** | **FPS** |
+------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+
| db_resnet50      | (1024, 1024, 3) | 25.2 M       | 82.14      | 87.64         | 92.49      | 89.66         | 2.1     |
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
.. autofunction:: doctr.models.detection.linknet16

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
     - 15.8M
     - 86.02
     - 91.3
     - 12.8
   * - sar_vgg16_bn
     - (32, 128, 3)
     - 21.5M
     - 86.2
     - 91.7
     - 3.3
   * - sar_resnet31
     - (32, 128, 3)
     - 53.1M
     - **86.3**
     - **92.1**
     - 2.7

All text recognition models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

All these recognition models are trained with our french vocab (cf. :ref:`vocabs`).

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
.. autofunction:: doctr.models.recognition.master


Recognition predictors
^^^^^^^^^^^^^^^^^^^^^^
Combining the right components around a given architecture for easier usage.

.. autofunction:: doctr.models.recognition.recognition_predictor


End-to-End OCR
--------------
Predictors that localize and identify text elements in images

+-----------------------------+--------------------------------------+--------------------------------------+
|                             |                  FUNSD               |                  CORD                |
+=============================+============+===============+=========+============+===============+=========+
| **Architecture**            | **Recall** | **Precision** | **FPS** | **Recall** | **Precision** | **FPS** |
+-----------------------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + crnn_vgg16_bn | 70.08      | 74.77         | 0.85    | 82.19      | **79.67**     | 1.6     |
+-----------------------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + sar_vgg16_bn  | N/A        | N/A           | 0.49    | N/A        | N/A           | 1.0     |
+-----------------------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + sar_resnet31  | N/A        | N/A           | 0.27    | N/A        | N/A           | 0.83    |
+-----------------------------+------------+---------------+---------+------------+---------------+---------+
| Gvision text detection      | 59.50      | 62.50         |         | 75.30      | 70.00         |         |
+-----------------------------+------------+---------------+---------+------------+---------------+---------+
| Gvision doc. text detection | 64.00      | 53.30         |         | 68.90      | 61.10         |         |
+-----------------------------+------------+---------------+---------+------------+---------------+---------+
| AWS textract                | **78.10**  | **83.00**     |         | **87.50**  | 66.00         |         |
+-----------------------------+------------+---------------+---------+------------+---------------+---------+

All OCR models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

All recognition models of predictors are trained with our french vocab (cf. :ref:`vocabs`).

*Disclaimer: both FUNSD subsets combine have 199 pages which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed this way: we instantiate the predictor, we warm-up the model and then we measure the average speed of the end-to-end predictor on the datasets, with a batch size of 1.
We used a c5.x12large from AWS instances (CPU Xeon Platinum 8275L) to perform experiments.

Results on private ocr datasets

+------------------------------------+----------------------------+----------------------------+----------------------------+
|                                    |          Receipts          |            Invoices        |            IDs             |
+====================================+============+===============+============+===============+============+===============+
| **Architecture**                   | **Recall** | **Precision** | **Recall** | **Precision** | **Recall** | **Precision** |
+------------------------------------+------------+---------------+------------+---------------+------------+---------------+
| db_resnet50 + crnn_vgg16_bn (ours) | **78.90**  | **81.01**     | 65.68      | **69.86**     | **49.48**  | **50.46**     |
+------------------------------------+------------+---------------+------------+---------------+------------+---------------+
| Gvision doc. text detection        | 68.91      | 59.89         | 63.20      | 52.85         | 43.70      | 29.21         |
+------------------------------------+------------+---------------+------------+---------------+------------+---------------+
| AWS textract                       | 75.77      | 77.70         | **70.47**  | 69.13         | 46.39      | 43.32         |
+------------------------------------+------------+---------------+------------+---------------+------------+---------------+


Two-stage approaches
^^^^^^^^^^^^^^^^^^^^
Those architectures involve one stage of text detection, and one stage of text recognition. The text detection will be used to produces cropped images that will be passed into the text recognition block.

.. autofunction:: doctr.models.zoo.ocr_predictor

Export model output
^^^^^^^^^^^^^^^^^^^^

The ocr_predictor returns a `Document` object with a nested structure (with `Page`, `Block`, `Line`, `Word`, `Artefact`). 
To get a better understanding of our document model, check our :ref:`document_structure` section

Here is a typical `Document` layout::

  Document(
    (pages): [Page(
      dimensions=(340, 600)
      (blocks): [Block(
        (lines): [Line(
          (words): [
            Word(value='No.', confidence=0.91),
            Word(value='RECEIPT', confidence=0.99),
            Word(value='DATE', confidence=0.96),
          ]
        )]
        (artefacts): []
      )]
    )]
  )

You can also export them as a nested dict, more appropriate for JSON format::

  json_output = result.export()

For reference, here is the JSON export for the same `Document` as above::

  {
    'pages': [
        {
            'page_idx': 0,
            'dimensions': (340, 600),
            'orientation': {'value': None, 'confidence': None},
            'language': {'value': None, 'confidence': None},
            'blocks': [
                {
                    'geometry': ((0.1357421875, 0.0361328125), (0.8564453125, 0.8603515625)),
                    'lines': [
                        {
                            'geometry': ((0.1357421875, 0.0361328125), (0.8564453125, 0.8603515625)),
                            'words': [
                                {
                                    'value': 'No.',
                                    'confidence': 0.914085328578949,
                                    'geometry': ((0.5478515625, 0.06640625), (0.5810546875, 0.0966796875))
                                },
                                {
                                    'value': 'RECEIPT',
                                    'confidence': 0.9949972033500671,
                                    'geometry': ((0.1357421875, 0.0361328125), (0.51171875, 0.1630859375))
                                },
                                {
                                    'value': 'DATE',
                                    'confidence': 0.9578408598899841,
                                    'geometry': ((0.1396484375, 0.3232421875), (0.185546875, 0.3515625))
                                }
                            ]
                        }
                    ],
                    'artefacts': []
                }
            ]
        }
    ]
  }

Model export
------------
Utility functions to make the most of document analysis models.

.. currentmodule:: doctr.models.export

Model compression
^^^^^^^^^^^^^^^^^

.. autofunction:: convert_to_tflite

.. autofunction:: convert_to_fp16

.. autofunction:: quantize_model

Using SavedModel
^^^^^^^^^^^^^^^^

Additionally, models in DocTR inherit TensorFlow 2 model properties and can be exported to
`SavedModel <https://www.tensorflow.org/guide/saved_model>`_ format as follows:


    >>> import tensorflow as tf
    >>> from doctr.models import db_resnet50
    >>> model = db_resnet50(pretrained=True)
    >>> input_t = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> _ = model(input_t, training=False)
    >>> tf.saved_model.save(model, 'path/to/your/folder/db_resnet50/')

And loaded just as easily:


    >>> import tensorflow as tf
    >>> model = tf.saved_model.load('path/to/your/folder/db_resnet50/')
