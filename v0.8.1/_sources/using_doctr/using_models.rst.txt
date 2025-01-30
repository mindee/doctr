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
* :py:meth:`fast_tiny <doctr.models.detection.fast_tiny>`
* :py:meth:`fast_small <doctr.models.detection.fast_small>`
* :py:meth:`fast_base <doctr.models.detection.fast_base>`

For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:


+------------------------------------------------------------------------------------+----------------------------+----------------------------+--------------------+
|                                                                                    |        FUNSD               |        CORD                |                    |
+================+=================================+=================+===============+============+===============+============+===============+====================+
| **Backend**    | **Architecture**                | **Input shape** | **# params**  | **Recall** | **Precision** | **Recall** | **Precision** | **sec/it (B: 1)**  |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | db_resnet50                     | (1024, 1024, 3) | 25.2 M        | 84.39      | 85.86         | 93.70      | 83.24         | 1.2                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | db_mobilenet_v3_large           | (1024, 1024, 3) | 4.2 M         | 80.29      | 70.90         | 84.70      | 67.76         | 0.5                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | linknet_resnet18                | (1024, 1024, 3) | 11.5 M        | 81.37      | 84.08         | 85.71      | 83.70         | 0.7                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | linknet_resnet34                | (1024, 1024, 3) | 21.6 M        | 82.20      | 85.49         | 87.63      | 87.17         | 0.8                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | linknet_resnet50                | (1024, 1024, 3) | 28.8 M        | 80.70      | 83.51         | 86.46      | 84.94         | 1.1                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | fast_tiny                       | (1024, 1024, 3) | 13.5 M (8.5M) | 85.29      | 85.34         | 93.46      | 75.99         | 0.7 (0.4)          |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | fast_small                      | (1024, 1024, 3) | 14.7 M (9.7M) | 85.50      | 86.89         | 94.05      | 78.33         | 0.7 (0.5)          |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | fast_base                       | (1024, 1024, 3) | 16.3 M (10.6M)| 85.22      | 86.97         | 94.18      | 84.74         | 0.8 (0.5)          |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | db_resnet34                     | (1024, 1024, 3) | 22.4 M        | 82.76      | 76.75         | 89.20      | 71.74         | 0.8                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | db_resnet50                     | (1024, 1024, 3) | 25.4 M        | 83.56      | 86.68         | 92.61      | 86.39         | 1.1                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | db_mobilenet_v3_large           | (1024, 1024, 3) | 4.2 M         | 82.69      | 84.63         | 94.51      | 70.28         | 0.5                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | linknet_resnet18                | (1024, 1024, 3) | 11.5 M        | 81.64      | 85.52         | 88.92      | 82.74         | 0.6                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | linknet_resnet34                | (1024, 1024, 3) | 21.6 M        | 81.62      | 82.95         | 86.26      | 81.06         | 0.7                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | linknet_resnet50                | (1024, 1024, 3) | 28.8 M        | 81.78      | 82.47         | 87.29      | 85.54         | 1.0                |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | fast_tiny                       | (1024, 1024, 3) | 13.5 M (8.5M) | 84.90      | 85.04         | 93.73      | 76.26         | 0.7 (0.4)          |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | fast_small                      | (1024, 1024, 3) | 14.7 M (9.7M) | 85.36      | 86.68         | 94.09      | 78.53         | 0.7 (0.5)          |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | fast_base                       | (1024, 1024, 3) | 16.3 M (10.6M)| 84.95      | 86.73         | 94.39      | 85.36         | 0.8 (0.5)          |
+----------------+---------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+


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
    model = detection_predictor('db_resnet50')
    dummy_img = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    out = model([dummy_img])

You can pass specific boolean arguments to the predictor:
* `pretrained`: if you want to use a model that has been pretrained on a specific dataset, setting `pretrained=True` this will load the corresponding weights. If `pretrained=False`, which is the default, would otherwise lead to a random initialization and would lead to no/useless results.
* `assume_straight_pages`: if you work with straight documents only, it will fit straight bounding boxes to the text areas.
* `preserve_aspect_ratio`: if you want to preserve the aspect ratio of your documents while resizing before sending them to the model.
* `symmetric_pad`: if you choose to preserve the aspect ratio, it will pad the image symmetrically and not from the bottom-right.

For instance, this snippet will instantiates a detection predictor able to detect text on rotated documents while preserving the aspect ratio:

.. code:: python3

    from doctr.models import detection_predictor
    predictor = detection_predictor('db_resnet50', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)


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
| TensorFlow     | crnn_mobilenet_v3_small         | (32, 128, 3)    | 2.1 M        | 86.88      | 87.61         | 92.28      | 92.73         | 0.25               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | crnn_mobilenet_v3_large         | (32, 128, 3)    | 4.5 M        | 87.44      | 88.12         | 94.14      | 94.55         | 0.34               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | master                          | (32, 128, 3)    | 58.8 M       | 87.44      | 88.21         | 93.83      | 94.25         | 22.3               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | sar_resnet31                    | (32, 128, 3)    | 57.2 M       | 87.67      | 88.48         | 94.21      | 94.66         | 7.1                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| TensorFlow     | vitstr_small                    | (32, 128, 3)    | 21.4 M       | 83.01      | 83.84         | 86.57      | 87.00         | 2.0                |
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
| PyTorch        | master                          | (32, 128, 3)    | 58.7 M       | 88.57      | 89.39         | 95.73      | 96.21         | 17.6               |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | sar_resnet31                    | (32, 128, 3)    | 55.4 M       | 88.10      | 88.88         | 94.83      | 95.29         | 4.9                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | vitstr_small                    | (32, 128, 3)    | 21.4 M       | 88.00      | 88.82         | 95.40      | 95.78         | 1.5                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | vitstr_base                     | (32, 128, 3)    | 85.2 M       | 88.33      | 89.09         | 95.32      | 95.71         | 4.1                |
+----------------+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| PyTorch        | parseq                          | (32, 128, 3)    | 23.8 M       | 88.53      | 89.24         | 95.56      | 95.91         | 2.2                |
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
    model = recognition_predictor('crnn_vgg16_bn')
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
| TensorFlow     | db_resnet50 + crnn_vgg16_bn                              | 73.45      | 74.73         | 85.79      | 76.21         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + crnn_mobilenet_v3_small                    | 72.66      | 73.93         | 83.43      | 74.11         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + crnn_mobilenet_v3_large                    | 72.86      | 74.13         | 85.16      | 75.65         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + master                                     | 72.73      | 74.00         | 84.13      | 75.05         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + sar_resnet31                               | 73.23      | 74.51         | 85.34      | 76.03         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + vitstr_small                               | 68.57      | 69.77         | 78.24      | 69.51         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + vitstr_base                                | 70.96      | 72.20         | 82.10      | 72.94         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| TensorFlow     | db_resnet50 + parseq                                     | 68.85      | 70.05         | 72.38      | 64.30         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + crnn_vgg16_bn                              | 72.43      | 75.13         | 85.05      | 79.33         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + crnn_mobilenet_v3_small                    | 73.06      | 75.79         | 84.64      | 78.94         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + crnn_mobilenet_v3_large                    | 73.17      | 75.90         | 84.96      | 79.25         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + master                                     | 73.90      | 76.66         | 85.84      | 80.07         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + sar_resnet31                               | 73.58      | 76.33         | 85.64      | 79.88         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + vitstr_small                               | 73.06      | 75.79         | 85.95      | 80.17         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + vitstr_base                                | 73.70      | 76.46         | 85.76      | 79.99         |
+----------------+----------------------------------------------------------+------------+---------------+------------+---------------+
| PyTorch        | db_resnet50 + parseq                                     | 73.52      | 76.27         | 85.91      | 80.13         |
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

* `assume_straight_pages`: if you work with straight documents only, it will fit straight bounding boxes to the text areas.
* `preserve_aspect_ratio`: if you want to preserve the aspect ratio of your documents while resizing before sending them to the model.
* `symmetric_pad`: if you choose to preserve the aspect ratio, it will pad the image symmetrically and not from the bottom-right.

Those 3 are going straight to the detection predictor, as mentioned above (in the detection part).

Additional arguments which can be passed to the `ocr_predictor` are:

* `export_as_straight_boxes`: If you work with rotated and skewed documents but you still want to export straight bounding boxes and not polygons, set it to True.
* `straighten_pages`: If you want to straighten the pages before sending them to the detection model, set it to True.

For instance, this snippet instantiates an end-to-end ocr_predictor working with rotated documents, which preserves the aspect ratio of the documents, and returns polygons:

.. code:: python3

    from doctr.models import ocr_predictor
    model = ocr_predictor('linknet_resnet18', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)


Additionally, you can change the batch size of the underlying detection and recognition predictors to optimize the performance depending on your hardware:

* `det_bs`: batch size for the detection model (default: 2)
* `reco_bs`: batch size for the recognition model (default: 128)

.. code:: python3

    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True, det_bs=4, reco_bs=1024)

To modify the output structure you can pass the following arguments to the predictor which will be handled by the underlying `DocumentBuilder`:

* `resolve_lines`: whether words should be automatically grouped into lines (default: True)
* `resolve_blocks`: whether lines should be automatically grouped into blocks (default: False)
* `paragraph_break`: relative length of the minimum space separating paragraphs (default: 0.035)

For example to disable the automatic grouping of lines into blocks:

.. code:: python3

    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True, resolve_blocks=False)


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

To get only the text content of the `Document`, you can use the `render` method::

  text_output = result.render()

For reference, here is the output for the `Document` above::

  No. RECEIPT DATE

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
                                    'geometry': ((0.5478515625, 0.06640625), (0.5810546875, 0.0966796875)),
                                    'objectness_score': 0.96,
                                    'crop_orientation': {'value': 0, 'confidence': None},
                                },
                                {
                                    'value': 'RECEIPT',
                                    'confidence': 0.9949972033500671,
                                    'geometry': ((0.1357421875, 0.0361328125), (0.51171875, 0.1630859375)),
                                    'objectness_score': 0.99,
                                    'crop_orientation': {'value': 0, 'confidence': None},
                                },
                                {
                                    'value': 'DATE',
                                    'confidence': 0.9578408598899841,
                                    'geometry': ((0.1396484375, 0.3232421875), (0.185546875, 0.3515625)),
                                    'objectness_score': 0.99,
                                    'crop_orientation': {'value': 0, 'confidence': None},
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


Advanced options
^^^^^^^^^^^^^^^^
We provide a few advanced options to customize the behavior of the predictor to your needs:

* Modify the binarization threshold for the detection model.
* Modify the box threshold for the detection model.

This is useful to detect (possible less) text regions more accurately with a higher threshold, or to detect more text regions with a lower threshold.


.. code:: python3

    import numpy as np
    from doctr.models import ocr_predictor
    predictor = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)

    # Modify the binarization threshold and the box threshold
    predictor.det_predictor.model.postprocessor.bin_thresh = 0.5
    predictor.det_predictor.model.postprocessor.box_thresh = 0.2

    input_page = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    out = predictor([input_page])


* Disable page orientation classification

If you deal with documents which contains only small rotations (~ -45 to 45 degrees), you can disable the page orientation classification to speed up the inference.

This will only have an effect with `assume_straight_pages=False` and/or `straighten_pages=True` and/or `detect_orientation=True`.

.. code:: python3

    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True, assume_straight_pages=False, disable_page_orientation=True)


* Disable crop orientation classification

If you deal with documents which contains only horizontal text, you can disable the crop orientation classification to speed up the inference.

This will only have an effect with `assume_straight_pages=False` and/or `straighten_pages=True`.

.. code:: python3

    from doctr.models import ocr_predictor
    model = ocr_predictor(pretrained=True, assume_straight_pages=False, disable_crop_orientation=True)


* Add a hook to the `ocr_predictor` to manipulate the location predictions before the crops are passed to the recognition model.

.. code:: python3

    from doctr.models import ocr_predictor

    class CustomHook:
        def __call__(self, loc_preds):
            # Manipulate the location predictions here
            # 1. The outpout structure needs to be the same as the input location predictions
            # 2. Be aware that the coordinates are relative and needs to be between 0 and 1
            return loc_preds

    my_hook = CustomHook()

    predictor = ocr_predictor(pretrained=True)
    # Add a hook in the middle of the pipeline
    predictor.add_hook(my_hook)
    # You can also add multiple hooks which will be executed sequentially
    for hook in [my_hook, my_hook, my_hook]:
        predictor.add_hook(hook)
