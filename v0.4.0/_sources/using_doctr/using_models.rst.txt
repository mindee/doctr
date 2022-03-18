Choosing the right model
========================

The full Optical Character Recognition task can be seen as two consecutive tasks: text detection and text recognition.
Either performed at once or separately, to each task corresponds a type of deep learning architecture.

.. currentmodule:: doctr.models

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

* `linknet_resnet18 <models.html#doctr.models.detection.linknet_resnet18>`_
* `db_resnet50 <models.html#doctr.models.detection.db_resnet50>`_
* `db_mobilenet_v3_large <models.html#doctr.models.detection.db_mobilenet_v3_large>`_

We also provide 2 models working with any kind of rotated documents:

* `linknet_resnet18_rotation <models.html#doctr.models.detection.linknet_resnet18_rotation>`_
* `db_resnet50_rotation <models.html#doctr.models.detection.db_resnet50_rotation>`_

For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:


+------------------------------------------------------------------+----------------------------+----------------------------+---------+
|                                                                  |        FUNSD               |        CORD                |         |
+=================================+=================+==============+============+===============+============+===============+=========+
| **Architecture**                | **Input shape** | **# params** | **Recall** | **Precision** | **Recall** | **Precision** | **FPS** |
+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+
| db_resnet50                     | (1024, 1024, 3) | 25.2 M       | 82.14      | 87.64         | 92.49      | 89.66         | 2.1     |
+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+
| db_mobilenet_v3_large           | (1024, 1024, 3) |  4.2 M       | 79.35      | 84.03         | 81.14      | 66.85         |         |
+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+


All text detection models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combined have 199 pages which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed after a warmup phase of 100 tensors (where the batch size is 1), by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `c5.x12large <https://aws.amazon.com/ec2/instance-types/c5/>`_ AWS instance (CPU Xeon Platinum 8275L).


Detection predictors
^^^^^^^^^^^^^^^^^^^^

`detection_predictor <models.html#doctr.models.detection.detection_predictor>`_ wraps your detection model to make it easily useable with your favorite deep learning framework seamlessly.

    >>> import numpy as np
    >>> from doctr.models import detection_predictor
    >>> predictor = detection_predictor('db_resnet50')
    >>> dummy_img = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    >>> out = model([dummy_img])

You can pass specific boolean arguments to the predictor:

* `assume_straight_pages`: if you work with straight documents only, it will fit straight bounding boxes to the text areas.
* `preserve_aspect_ratio`: if you want to preserve the aspect ratio of your documents while resizing before sending them to the model.
* `symmetric_pad`: if you choose to preserve the aspect ratio, it will pad the image symmetrically and not from the bottom-right.

For instance, this snippet will instantiates a detection predictor able to detect text on rotated documents while preserving the aspect ratio:

    >>> from doctr.models import detection_predictor
    >>> predictor = detection_predictor('db_resnet50_rotation', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)

NB: for the moment, `db_resnet50_rotation` is pretrained in Pytorch only and `linknet_resnet18_rotation` in Tensorflow only.


Text Recognition
----------------

The task consists of transcribing the character sequence in a given image.


Available architectures
^^^^^^^^^^^^^^^^^^^^^^^

The following architectures are currently supported:

* `crnn_vgg16_bn <models.html#doctr.models.recognition.crnn_vgg16_bn>`_
* `crnn_mobilenet_v3_small <models.html#doctr.models.recognition.crnn_mobilenet_v3_small>`_
* `crnn_mobilenet_v3_large <models.html#doctr.models.recognition.crnn_mobilenet_v3_large>`_
* `sar_resnet31 <models.html#doctr.models.recognition.sar_resnet31>`_
* `master <models.html#doctr.models.recognition.master>`_


For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:


.. list-table:: Text recognition model zoo
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
     - 87.18
     - 92.93
     - 12.8
   * - crnn_mobilenet_v3_small
     - (32, 128, 3)
     - 2.1M
     - 86.21
     - 90.56
     -
   * - crnn_mobilenet_v3_large
     - (32, 128, 3)
     - 4.5M
     - 86.95
     - 92.03
     -
   * - sar_resnet31
     - (32, 128, 3)
     - 56.2M
     - **87.70**
     - **93.41**
     - 2.7
   * - master
     - (32, 128, 3)
     - 67.7M
     - 87.62
     - 93.27
     -

All text recognition models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metric being used (exact match) are available in :ref:`metrics`.

While most of our recognition models were trained on our french vocab (cf. :ref:`vocabs`), you can easily access the vocab of any model as follows:

    >>> from doctr.models import recognition_predictor
    >>> predictor = recognition_predictor('crnn_vgg16_bn')
    >>> print(predictor.model.cfg['vocab'])


*Disclaimer: both FUNSD subsets combine have 30595 word-level crops which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed after a warmup phase of 100 tensors (where the batch size is 1), by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `c5.x12large <https://aws.amazon.com/ec2/instance-types/c5/>`_ AWS instance (CPU Xeon Platinum 8275L).


Recognition predictors
^^^^^^^^^^^^^^^^^^^^^^
`recognition_predictor <models.html#doctr.models.recognition.recognition_predictor>`_ wraps your recognition model to make it easily useable with your favorite deep learning framework seamlessly.

    >>> import numpy as np
    >>> from doctr.models import recognition_predictor
    >>> predictor = recognition_predictor('crnn_vgg16_bn')
    >>> dummy_img = (255 * np.random.rand(50, 150, 3)).astype(np.uint8)
    >>> out = model([dummy_img])


End-to-End OCR
--------------

The task consists of both localizing and transcribing textual elements in a given image.

Available architectures
^^^^^^^^^^^^^^^^^^^^^^^

You can use any combination of detection and recognition models supporte by docTR.

For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:

+----------------------------------------+--------------------------------------+--------------------------------------+
|                                        |                  FUNSD               |                  CORD                |
+========================================+============+===============+=========+============+===============+=========+
| **Architecture**                       | **Recall** | **Precision** | **FPS** | **Recall** | **Precision** | **FPS** |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + crnn_vgg16_bn            | 71.25      | 76.02         | 0.85    | 84.00      |   81.42       | 1.6     |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + master                   | 71.03      | 76.06         |         | 84.49      |   81.94       |         |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + sar_resnet31             | 71.25      | 76.29         | 0.27    | 84.50      | **81.96**     | 0.83    |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + crnn_mobilenet_v3_small  | 69.85      | 74.80         |         | 80.85      | 78.42         | 0.83    |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| db_resnet50 + crnn_mobilenet_v3_large  | 70.57      | 75.57         |         | 82.57      | 80.08         | 0.83    |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| db_mobilenet_v3_large + crnn_vgg16_bn  | 67.73      | 71.73         |         | 71.65      | 59.03         |         |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| Gvision text detection                 | 59.50      | 62.50         |         | 75.30      | 70.00         |         |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| Gvision doc. text detection            | 64.00      | 53.30         |         | 68.90      | 61.10         |         |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+
| AWS textract                           | **78.10**  | **83.00**     |         | **87.50**  | 66.00         |         |
+----------------------------------------+------------+---------------+---------+------------+---------------+---------+

All OCR models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combine have 199 pages which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed after a warmup phase of 100 tensors (where the batch size is 1), by measuring the average number of processed frames per second over 1000 samples. Those results were obtained on a `c5.x12large <https://aws.amazon.com/ec2/instance-types/c5/>`_ AWS instance (CPU Xeon Platinum 8275L).

Since you may be looking for specific use cases, we also performed this benchmark on private datasets with various document types below. Unfortunately, we are not able to share those at the moment since they contain sensitive information.


+----------------------------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
|                                              |          Receipts          |            Invoices        |            IDs             |        US Tax Forms        |         Resumes            |         Road Fines         |
+==============================================+============+===============+============+===============+============+===============+============+===============+============+===============+============+===============+
| **Architecture**                             | **Recall** | **Precision** | **Recall** | **Precision** | **Recall** | **Precision** | **Recall** | **Precision** | **Recall** | **Precision** | **Recall** | **Precision** |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+
| db_resnet50 + crnn_vgg16_bn (ours)           |   78.70    |   81.12       | 65.80      |   70.70       |   50.25    |   51.78       |   79.08    |   92.83       |            |               |            |               |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+
| db_resnet50 + master (ours)                  | **79.00**  | **81.42**     | 65.57      |   69.86       |   51.34    |   52.90       |   78.86    |   92.57       |            |               |            |               |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+
| db_resnet50 + sar_resnet31 (ours)            |   78.94    |   81.37       | 65.89      | **70.79**     | **51.78**  | **53.35**     |   79.04    |   92.78       |            |               |            |               |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+
| db_resnet50 + crnn_mobilenet_v3_small (ours) |   76.81    |     79.15     |    64.89   |    69.61      |  45.03     | 46.38         |  78.96     |   92.11       |    85.91   |     87.20     |   84.85    |     85.86     |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+
| db_resnet50 + crnn_mobilenet_v3_large (ours) |   78.01    |     80.39     |    65.36   |    70.11      |  48.00     | 49.43         |  79.39     |   92.62       |    87.68   |     89.00     |   85.65    |     86.67     |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+
| db_mobilenet_v3_large + crnn_vgg16_bn (ours) |   78.36    |   74.93       | 63.04      | 68.41         | 39.36      | 41.75         |   72.14    |   89.97       |            |               |            |               |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+
| Gvision doc. text detection                  | 68.91      | 59.89         | 63.20      | 52.85         | 43.70      | 29.21         |   69.79    |   65.68       |            |               |            |               |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+
| AWS textract                                 | 75.77      | 77.70         | **70.47**  | 69.13         | 46.39      | 43.32         | **84.31**  | **98.11**     |            |               |            |               |
+----------------------------------------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+------------+---------------+


Two-stage approaches
^^^^^^^^^^^^^^^^^^^^
Those architectures involve one stage of text detection, and one stage of text recognition. The text detection will be used to produces cropped images that will be passed into the text recognition block. Everything is wrapped up with `ocr_predictor <models.html#doctr.models.ocr_predictor>`_.

    >>> import numpy as np
    >>> from doctr.models import ocr_predictor
    >>> model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    >>> input_page = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    >>> out = model([input_page])


You can pass specific boolean arguments to the predictor:

* `assume_straight_pages`
* `preserve_aspect_ratio`
* `symmetric_pad`

Those 3 are going straight to the detection predictor, as mentioned above (in the detection part).

* `export_as_straight_boxes`: If you work with rotated and skewed documents but you still want to export straight bounding boxes and not polygons, set it to True.

For instance, this snippet instantiates an end-to-end ocr_predictor working with rotated documents, which preserves the aspect ratio of the documents, and returns polygons:

    >>> from doctr.model import ocr_predictor
    >>> model = ocr_predictor('linknet_resnet18_rotation', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)


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

To export the outpout as XML (hocr-format) you can use the `export_as_xml` method::

  xml_output = result.export_as_xml()
  for output in xml_output:
    xml_bytes_string = output[0]
    xml_element = output[1]

For reference, here is a sample XML byte string output::

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
