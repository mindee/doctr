Choosing the right model
========================

The full Optical Character Recognition task can be seen as two consecutive tasks: text detection and text recognition.
Either performed at once or separately, to each task corresponds a type of deep learning architecture.

For a given task, docTR provides a Predictor, which is composed of 2 components:

* PreProcessor: a module in charge of making inputs directly usable by the deep learning model.
* Model: a deep learning model, implemented with all supported deep learning backends (TensorFlow & PyTorch) along with its specific post-processor to make outputs structured and reusable.


Text Detection
--------------

The task consists of localizing textual elements in a given image.
While those text elements can represent many things, in docTR, we will consider uninterrupted character sequences (words). Additionally, the localization can take several forms: from straight bounding boxes (delimited by the 2D coordinates of the top-left and bottom-right corner), to polygons, or binary segmentation (flagging which pixels belong to this element, and which don't).
Our latest detection models works with rotated and skewed documents!

Available architectures
^^^^^^^^^^^^^^^^^^^^^^^

The following architectures are currently supported:

* :py:meth:`linknet_resnet18 <doctr.models.detection.linknet_resnet18>`
* :py:meth:`linknet_resnet34 <doctr.models.detection.linknet_resnet34>`
* :py:meth:`linknet_resnet50 <doctr.models.detection.linknet_resnet50>`
* :py:meth:`db_resnet50 <doctr.models.detection.db_resnet50>`
* :py:meth:`db_mobilenet_v3_large <doctr.models.detection.db_mobilenet_v3_large>`

We also provide 2 models working with any kind of rotated documents:

* :py:meth:`linknet_resnet18_rotation <doctr.models.detection.linknet_resnet18_rotation>` (TensorFlow)
* :py:meth:`db_resnet50_rotation <doctr.models.detection.differentiable_binarization.pytorch.db_resnet50_rotation>` (PyTorch)

For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:


+-----------------------------------------------------------------------------------+----------------------------+----------------------------+--------------------+
|                                                                                   |        FUNSD               |        CORD                |                    |
+================+=================================+=================+==============+============+===============+============+===============+====================+
| **Backend**    | **Architecture**                | **Input shape** | **# params** | **Recall** | **Precision** | **Recall** | **Precision** | **sec/it (B: 1)**  |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | db_resnet50                     | (1024, 1024, 3) | 25.2 M       | 81.22      | 86.66         | 92.46      | 89.62         | 1.2                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| Tensorflow     | db_mobilenet_v3_large           | (1024, 1024, 3) | 4.2 M        | 78.27      | 82.77         | 80.99      | 66.57         | 0.5                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | linknet_resnet18                | (1024, 1024, 3) | 11.5 M       | 78.23      | 83.77         | 82.88      | 82.42         | 0.7                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| Tensorflow     | linknet_resnet18_rotation       | (1024, 1024, 3) | 11.5 M       | 81.12      | 82.13         | 83.55      | 80.14         | 0.6                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | linknet_resnet34                | (1024, 1024, 3) | 21.6 M       | 82.14      | 87.64         | 85.55      | 86.02         | 0.8                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| Tensorflow     | linknet_resnet50                | (1024, 1024, 3) | 28.8 M       | 79.00      | 84.79         | 85.89      | 65.75         | 1.1                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | db_resnet34                     | (1024, 1024, 3) | 22.4 M       |            |               |            |               |                    |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | db_resnet50                     | (1024, 1024, 3) | 25.4 M       | 79.17      | 86.31         | 92.96      | 91.23         | 1.1                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | db_resnet50_rotation            | (1024, 1024, 3) | 25.4 M       | 83.30      | 91.07         | 91.63      | 90.53         | 1.6                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | db_mobilenet_v3_large           | (1024, 1024, 3) | 4.2 M        | 80.06      | 84.12         | 80.51      | 66.51         | 0.5                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | linknet_resnet18                | (1024, 1024, 3) | 11.5 M       |            |               |            |               |                    |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | linknet_resnet34                | (1024, 1024, 3) | 21.6 M       |            |               |            |               |                    |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | linknet_resnet50                | (1024, 1024, 3) | 28.8 M       |            |               |            |               |                    |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+


All text detection models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combined have 199 pages which might not be representative enough of the model capabilities*

Seconds per iteration (with a batch size of 1) is computed after a warmup phase of 100 tensors, by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz`.


Detection predictors
^^^^^^^^^^^^^^^^^^^^

:py:meth:`detection_predictor <doctr.models.detection.detection_predictor>` wraps your detection model to make it easily useable with your favorite deep learning framework seamlessly.

.. code:: python3

    import numpy as np
    from doctr.models import detection_predictor
    predictor = detection_predictor('db_resnet50')
    dummy_img = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    out = model([dummy_img])

You can pass specific boolean arguments to the predictor:

* `assume_straight_pages`: if you work with straight documents only, it will fit straight bounding boxes to the text areas.
* `preserve_aspect_ratio`: if you want to preserve the aspect ratio of your documents while resizing before sending them to the model.
* `symmetric_pad`: if you choose to preserve the aspect ratio, it will pad the image symmetrically and not from the bottom-right.

For instance, this snippet will instantiates a detection predictor able to detect text on rotated documents while preserving the aspect ratio:

.. code:: python3

    from doctr.models import detection_predictor
    predictor = detection_predictor('db_resnet50_rotation', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)

NB: for the moment, `db_resnet50_rotation` is pretrained in Pytorch only and `linknet_resnet18_rotation` in Tensorflow only.


Text Recognition
----------------

The task consists of transcribing the character sequence in a given image.


Available architectures
^^^^^^^^^^^^^^^^^^^^^^^

The following architectures are currently supported:

* :py:meth:`crnn_vgg16_bn <doctr.models.recognition.crnn_vgg16_bn>`
* :py:meth:`crnn_mobilenet_v3_small <doctr.models.recognition.crnn_mobilenet_v3_small>`
* :py:meth:`crnn_mobilenet_v3_large <doctr.models.recognition.crnn_mobilenet_v3_large>`
* :py:meth:`sar_resnet31 <doctr.models.recognition.sar_resnet31>`
* :py:meth:`master <doctr.models.recognition.master>`
* :py:meth:`vitstr_small <doctr.models.recognition.vitstr_small>`
* :py:meth:`vitstr_base <doctr.models.recognition.vitstr_base>`
* :py:meth:`parseq <doctr.models.recognition.parseq>`


For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:


+-----------------------------------------------------------------------------------+----------------------------+----------------------------+--------------------+
|                                                                                   |        FUNSD               |        CORD                |                    |
+================+=================================+=================+==============+============+===============+============+===============+====================+
| **Backend**    | **Architecture**                | **Input shape** | **# params** | **Exact**  | **Partial**   | **Exact**  | **Partial**   | **sec/it (B: 64)** |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | crnn_vgg16_bn                   | (32, 128, 3)    | 15.8 M       | 88.12      | 88.85         | 94.68      | 95.10         | 0.9                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| Tensorflow     | crnn_mobilenet_v3_small         | (32, 128, 3)    | 2.1 M        | 86.88      | 87.61         | 92.28      | 92.73         | 0.25               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | crnn_mobilenet_v3_large         | (32, 128, 3)    | 4.5 M        | 87.44      | 88.12         | 94.14      | 94.55         | 0.34               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| Tensorflow     | master                          | (32, 128, 3)    | 58.8 M       | 87.44      | 88.21         | 93.83      | 94.25         | 22.3               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | sar_resnet31                    | (32, 128, 3)    | 57.2 M       | 87.67      | 88.48         | 94.21      | 94.66         | 7.1                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| Tensorflow     | vitstr_small                    | (32, 128, 3)    | 21.4 M       | 83.01      | 83.84         | 86.57      | 87.00         | 2.0                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | vitstr_base                     | (32, 128, 3)    | 85.2 M       | 85.98      | 86.70         | 90.47      | 90.95         | 5.8                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | parseq                          | (32, 128, 3)    | 23.8 M       | 81.62      | 82.29         | 79.13      | 79.52         | 3.6                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | crnn_vgg16_bn                   | (32, 128, 3)    | 15.8 M       | 86.54      | 87.41         | 94.29      | 94.69         | 0.6                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | crnn_mobilenet_v3_small         | (32, 128, 3)    | 2.1 M        | 87.25      | 87.99         | 93.91      | 94.34         | 0.05               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | crnn_mobilenet_v3_large         | (32, 128, 3)    | 4.5 M        | 87.38      | 88.09         | 94.46      | 94.92         | 0.08               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | master                          | (32, 128, 3)    | 58.7 M       |            |               |            |               | 17.6               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | sar_resnet31                    | (32, 128, 3)    | 55.4 M       |            |               |            |               | 4.9                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | vitstr_small                    | (32, 128, 3)    | 21.4 M       |            |               |            |               | 1.5                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | vitstr_base                     | (32, 128, 3)    | 85.2 M       |            |               |            |               | 4.1                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | parseq                          | (32, 128, 3)    | 23.8 M       |            |               |            |               | 2.2                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+


All text recognition models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metric being used (exact match) are available in :ref:`metrics`.

While most of our recognition models were trained on our french vocab (cf. :ref:`vocabs`), you can easily access the vocab of any model as follows:

.. code:: python3

    from doctr.models import recognition_predictor
    predictor = recognition_predictor('crnn_vgg16_bn')
    print(predictor.model.cfg['vocab'])


*Disclaimer: both FUNSD subsets combine have 30595 word-level crops which might not be representative enough of the model capabilities*

Seconds per iteration (with a batch size of 64) is computed after a warmup phase of 100 tensors, by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz`.


Recognition predictors
^^^^^^^^^^^^^^^^^^^^^^
:py:meth:`recognition_predictor <doctr.models.recognition.recognition_predictor>` wraps your recognition model to make it easily useable with your favorite deep learning framework seamlessly.

.. code:: python3

    import numpy as np
    from doctr.models import recognition_predictor
    predictor = recognition_predictor('crnn_vgg16_bn')
    dummy_img = (255 * np.random.rand(50, 150, 3)).astype(np.uint8)
    out = model([dummy_img])


End-to-End OCR
--------------

The task consists of both localizing and transcribing textual elements in a given image.

Available architectures
^^^^^^^^^^^^^^^^^^^^^^^

You can use any combination of detection and recognition models supported by docTR.

For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:

+---------------------------------------------------------------------------+----------------------------+----------------------------+
|                                                                           |        FUNSD               |        CORD                |
+================+==========================================================+============================+============+===============+
| **Backend**    | **Architecture**                                         | **Recall** | **Precision** | **Recall** | **Precision** |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + crnn_vgg16_bn                              | 70.82      | 75.56         | 83.97      | 81.40         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + crnn_mobilenet_v3_small                    | 69.63      | 74.29         | 81.08      | 78.59         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + crnn_mobilenet_v3_large                    | 70.01      | 74.70         | 83.28      | 80.73         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + sar_resnet31                               | 68.75      | 73.76         | 78.56      | 76.24         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + master                                     | 68.75      | 73.76         | 78.56      | 76.24         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + vitstr_small                               | 64.58      | 68.91         | 74.66      | 72.37         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + vitstr_base                                | 66.89      | 71.37         | 79.11      | 76.68         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + parseq                                     | 65.77      | 70.18         | 71.57      | 69.37         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + crnn_vgg16_bn                              | 67.82      | 73.35         | 84.84      | 83.27         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + crnn_mobilenet_v3_small                    | 67.89      | 74.01         | 84.43      | 82.85         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + crnn_mobilenet_v3_large                    | 68.45      | 74.63         | 84.86      | 83.27         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + sar_resnet31                               |            |               |            |               |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + master                                     |            |               |            |               |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + vitstr_small                               |            |               |            |               |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + vitstr_base                                |            |               |            |               |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + parseq                                     |            |               |            |               |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| None           | Gvision text detection                                   | 59.50      | 62.50         | 75.30      | 59.03         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| None           | Gvision doc. text detection                              | 64.00      | 53.30         | 68.90      | 61.10         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| None           | AWS textract                                             | 78.10      | 83.00         | 87.50      | 66.00         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| None           | Azure Form Recognizer (v3.2)                             | 79.42      | 85.89         | 89.62      | 88.93         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+


All OCR models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combine have 199 pages which might not be representative enough of the model capabilities*


Two-stage approaches
^^^^^^^^^^^^^^^^^^^^
Those architectures involve one stage of text detection, and one stage of text recognition. The text detection will be used to produces cropped images that will be passed into the text recognition block. Everything is wrapped up with :py:meth:`ocr_predictor <doctr.models.ocr_predictor>`.

.. code:: python3

    import numpy as np
    from doctr.models import ocr_predictor
    model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    input_page = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    out = model([input_page])


You can pass specific boolean arguments to the predictor:

* `assume_straight_pages`
* `preserve_aspect_ratio`
* `symmetric_pad`

Those 3 are going straight to the detection predictor, as mentioned above (in the detection part).

* `export_as_straight_boxes`: If you work with rotated and skewed documents but you still want to export straight bounding boxes and not polygons, set it to True.

For instance, this snippet instantiates an end-to-end ocr_predictor working with rotated documents, which preserves the aspect ratio of the documents, and returns polygons:

.. code:: python3

    from doctr.model import ocr_predictor
    model = ocr_predictor('linknet_resnet18_rotation', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)


What should I do with the output?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

For reference, here is the export for the same `Document` as above::

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

To export the outpout as XML (hocr-format) you can use the `export_as_xml` method:

.. code-block:: python

  xml_output = result.export_as_xml()
  for output in xml_output:
      xml_bytes_string = output[0]
      xml_element = output[1]

For reference, here is a sample XML byte string output:

.. code-block:: xml

  <?xml version="1.0" encoding="UTF-8"?>
  <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
    <head>
      <title>docTR - hOCR</title>
      <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
      <meta name="ocr-system" content="doctr 0.5.0" />
      <meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />
    </head>
    <body>
      <div class="ocr_page" id="page_1" title="image; bbox 0 0 3456 3456; ppageno 0" />
      <div class="ocr_carea" id="block_1_1" title="bbox 857 529 2504 2710">
        <p class="ocr_par" id="par_1_1" title="bbox 857 529 2504 2710">
          <span class="ocr_line" id="line_1_1" title="bbox 857 529 2504 2710; baseline 0 0; x_size 0; x_descenders 0; x_ascenders 0">
            <span class="ocrx_word" id="word_1_1" title="bbox 1552 540 1778 580; x_wconf 99">Hello</span>
            <span class="ocrx_word" id="word_1_2" title="bbox 1782 529 1900 583; x_wconf 99">XML</span>
            <span class="ocrx_word" id="word_1_3" title="bbox 1420 597 1684 641; x_wconf 81">World</span>
          </span>
        </p>
      </div>
    </body>
  </html>
