.. _using_models:

Choosing the right model
========================

The full Optical Character Recognition task can be seen as two consecutive tasks: text detection and text recognition.
Either performed at once or separately, to each task corresponds a type of deep learning architecture.

For a given task, docTR provides a Predictor, which is composed of 2 components:

* PreProcessor: a module in charge of making inputs directly usable by the deep learning model.
* Model: a deep learning model, implemented with all supported deep learning backends (PyTorch) along with its specific post-processor to make outputs structured and reusable.


Which predictor should I use?
------------------------------

.. list-table::
   :widths: 60 40
   :header-rows: 1

   * - I want to…
     - Use
   * - Extract all text (words, lines, layout hierarchy) from a document
     - :py:meth:`ocr_predictor <doctr.models.ocr_predictor>`
   * - Detect document regions by type (tables, figures, headers, …)
     - :py:meth:`layout_predictor <doctr.models.layout_predictor>`
   * - Get word bounding-boxes only, without recognition
     - :py:meth:`detection_predictor <doctr.models.detection_predictor>`
   * - Transcribe pre-cropped word images to strings
     - :py:meth:`recognition_predictor <doctr.models.recognition_predictor>`
   * - Detect the structure of a table (cell bounding-boxes and logical coordinates)
     - :py:meth:`table_predictor <doctr.models.table_structure.table_predictor>`

For :doc:`custom model loading <custom_models_training>` or sharing models, see the dedicated pages.


Text Detection
--------------

The task consists of localizing textual elements in a given image.
While those text elements can represent many things, in docTR, we will consider uninterrupted character sequences (words). Additionally, the localization can take several forms: from straight bounding boxes (delimited by the 2D coordinates of the top-left and bottom-right corner), to polygons, or binary segmentation (flagging which pixels belong to this element, and which don't).
Our latest detection models works with rotated and skewed documents!

Available detection architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following architectures are currently supported:

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
+==================================================+=================+===============+============+===============+============+===============+====================+
| **Architecture**                                 | **Input shape** | **# params**  | **Recall** | **Precision** | **Recall** | **Precision** | **sec/it (B: 1)**  |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| db_resnet34                                      | (1024, 1024, 3) | 22.4 M        | 82.76      | 76.75         | 89.20      | 71.74         | 0.8                |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| db_resnet50                                      | (1024, 1024, 3) | 25.4 M        | 83.56      | 86.68         | 92.61      | 86.39         | 1.1                |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| db_mobilenet_v3_large                            | (1024, 1024, 3) | 4.2 M         | 82.69      | 84.63         | 94.51      | 70.28         | 0.5                |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| linknet_resnet18                                 | (1024, 1024, 3) | 11.5 M        | 81.64      | 85.52         | 88.92      | 82.74         | 0.6                |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| linknet_resnet34                                 | (1024, 1024, 3) | 21.6 M        | 81.62      | 82.95         | 86.26      | 81.06         | 0.7                |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| linknet_resnet50                                 | (1024, 1024, 3) | 28.8 M        | 81.78      | 82.47         | 87.29      | 85.54         | 1.0                |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| fast_tiny                                        | (1024, 1024, 3) | 13.5 M (8.5M) | 84.90      | 85.04         | 93.73      | 76.26         | 0.7 (0.4)          |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| fast_small                                       | (1024, 1024, 3) | 14.7 M (9.7M) | 85.36      | 86.68         | 94.09      | 78.53         | 0.7 (0.5)          |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+
| fast_base                                        | (1024, 1024, 3) | 16.3 M (10.6M)| 84.95      | 86.73         | 94.39      | 85.36         | 0.8 (0.5)          |
+--------------------------------------------------+-----------------+---------------+------------+---------------+------------+---------------+--------------------+


All text detection models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combined have 199 pages which might not be representative enough of the model capabilities*

Seconds per iteration (with a batch size of 1) is computed after a warmup phase of 100 tensors, by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz`.


Detection predictors
^^^^^^^^^^^^^^^^^^^^

:py:meth:`detection_predictor <doctr.models.detection.detection_predictor>` wraps your detection model to make it easily usable with your favorite deep learning framework seamlessly.

.. code:: python3

    import numpy as np
    from doctr.models import detection_predictor
    model = detection_predictor('db_resnet50')
    dummy_img = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    out = model([dummy_img])

You can pass specific boolean arguments to the predictor:

* ``pretrained``: if you want to use a model that has been pretrained on a specific dataset, setting ``pretrained=True`` will load the corresponding weights. If ``pretrained=False`` (the default), the model is randomly initialized and will produce no useful results.
* ``assume_straight_pages``: if you work with straight documents only, it will fit straight bounding boxes to the text areas.
* ``preserve_aspect_ratio``: if you want to preserve the aspect ratio of your documents while resizing before sending them to the model.
* ``symmetric_pad``: if you choose to preserve the aspect ratio, it will pad the image symmetrically and not from the bottom-right.

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
* :py:meth:`viptr_tiny <doctr.models.recognition.viptr_tiny>`


For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:


+-----------------------------------------------------------------------------------+----------------------------+----------------------------+--------------------+
|                                                                                   |        FUNSD               |        CORD                |                    |
+==================================================+=================+==============+============+===============+============+===============+====================+
| **Architecture**                                 | **Input shape** | **# params** | **Exact**  | **Partial**   | **Exact**  | **Partial**   | **sec/it (B: 64)** |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| crnn_vgg16_bn                                    | (32, 128, 3)    | 15.8 M       | 88.21      | 88.95         | 95.47      | 95.91         | 0.6                |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| crnn_mobilenet_v3_small                          | (32, 128, 3)    | 2.1 M        | 87.25      | 87.99         | 93.91      | 94.34         | 0.05               |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| crnn_mobilenet_v3_large                          | (32, 128, 3)    | 4.5 M        | 87.38      | 88.09         | 94.46      | 94.92         | 0.08               |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| master                                           | (32, 128, 3)    | 58.7 M       | 88.57      | 89.39         | 95.73      | 96.21         | 17.6               |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| sar_resnet31                                     | (32, 128, 3)    | 55.4 M       | 88.10      | 88.88         | 94.83      | 95.29         | 4.9                |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| vitstr_small                                     | (32, 128, 3)    | 21.4 M       | 88.00      | 88.82         | 95.40      | 95.78         | 1.5                |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| vitstr_base                                      | (32, 128, 3)    | 85.2 M       | 88.33      | 89.09         | 95.32      | 95.71         | 4.1                |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| parseq                                           | (32, 128, 3)    | 23.8 M       | 88.53      | 89.24         | 95.56      | 95.91         | 2.2                |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+
| viptr_tiny                                       | (32, 128, 3)    | 3.2 M        | 86.03      | 86.71         | 93.08      | 93.47         | 0.08               |
+--------------------------------------------------+-----------------+--------------+------------+---------------+------------+---------------+--------------------+


All text recognition models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metric being used (exact match) are available in :ref:`metrics`.

While most of our recognition models were trained on our french vocab (cf. :ref:`vocabs`), you can easily access the vocab of any model as follows:

.. code:: python3

    from doctr.models import recognition_predictor
    predictor = recognition_predictor('crnn_vgg16_bn')
    print(predictor.model.cfg['vocab'])


*Disclaimer: both FUNSD subsets combined have 30595 word-level crops which might not be representative enough of the model capabilities*

Seconds per iteration (with a batch size of 64) is computed after a warmup phase of 100 tensors, by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz`.


Recognition predictors
^^^^^^^^^^^^^^^^^^^^^^
:py:meth:`recognition_predictor <doctr.models.recognition.recognition_predictor>` wraps your recognition model to make it easily usable with your favorite deep learning framework seamlessly.

.. code:: python3

    import numpy as np
    from doctr.models import recognition_predictor
    model = recognition_predictor('crnn_vgg16_bn')
    dummy_img = (255 * np.random.rand(50, 150, 3)).astype(np.uint8)
    out = model([dummy_img])


Layout Analysis
---------------

The task consists of localizing and classifying visual elements in a given image.
This is a more general task than text detection, as it can be used to detect and classify any type of visual element in a document, such as tables, figures, headers, footers, etc.
Our latest layout models works with rotated and skewed documents!

Available layout architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following architectures are currently supported:

* :py:meth:`lw_detr_s <doctr.models.layout.lw_detr_s>`
* :py:meth:`lw_detr_m <doctr.models.layout.lw_detr_m>`

For a comprehensive comparison, we have compiled a detailed benchmark:

+--------------------------------------------------+-----------------+---------------+------------------+-------------+--------------+--------------------+
|                                                  |                 |               |                  |             |              |                    |
+==================================================+=================+===============+==================+=============+==============+====================+
| **Architecture**                                 | **Input shape** | **# params**  | **mAP@[.5:.95]** | **AP@[.5]** | **AP@[.75]** | **sec/it (B: 1)**  |
+--------------------------------------------------+-----------------+---------------+------------------+-------------+--------------+--------------------+
| lw_detr_s                                        | (1024, 1024, 3) | 15.1 M        |                  |             |              | 0.5                |
+--------------------------------------------------+-----------------+---------------+------------------+-------------+--------------+--------------------+
| lw_detr_m                                        | (1024, 1024, 3) | 29.5 M        |                  |             |              | 0.7                |
+--------------------------------------------------+-----------------+---------------+------------------+-------------+--------------+--------------------+


Explanations about the metrics being used are available in :ref:`metrics`.

Seconds per iteration (with a batch size of 1) is computed after a warmup phase of 100 tensors, by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz`.


Layout predictors
^^^^^^^^^^^^^^^^^

:py:meth:`layout_predictor <doctr.models.layout.layout_predictor>` wraps your layout model to make it easily usable with your favorite deep learning framework seamlessly.

.. code:: python3

    import numpy as np
    from doctr.models import layout_predictor
    model = layout_predictor('lw_detr_s')
    dummy_img = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    out = model([dummy_img])

You can pass specific boolean arguments to the predictor:

* ``pretrained``: if you want to use a model that has been pretrained on a specific dataset, setting ``pretrained=True`` will load the corresponding weights. If ``pretrained=False`` (the default), the model is randomly initialized and will produce no useful results.
* ``assume_straight_pages``: if you work with straight documents only, it will fit straight bounding boxes to the text areas.
* ``preserve_aspect_ratio``: if you want to preserve the aspect ratio of your documents while resizing before sending them to the model.
* ``symmetric_pad``: if you choose to preserve the aspect ratio, it will pad the image symmetrically and not from the bottom-right.

For instance, this snippet instantiates a layout predictor able to detect text on rotated documents while preserving the aspect ratio:

.. code:: python3

    from doctr.models import layout_predictor
    predictor = layout_predictor('lw_detr_s', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)


Table Structure Recognition
---------------------------

The task consists of parsing the structure of a table into a machine-understandable representation: localizing every
cell (its spatial structure) and recovering the row and column it spans (its logical structure).

Available table architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following architectures are currently supported:

* :py:meth:`tablecenternet <doctr.models.table_structure.tablecenternet>`

For a comprehensive comparison, we have compiled a detailed benchmark on a publicly available dataset:

+--------------------------------------------------+-----------------+---------------+--------------+---------------+------------+-------------------+--------------------+
| **Architecture**                                 | **Input shape** | **# params**  | **Recall**   | **Precision** | **F1**     | **Structure acc** | **sec/it (B: 1)**  |
+==================================================+=================+===============+==============+===============+============+===================+====================+
| tablecenternet                                   | (1024, 1024, 3) | 7.1 M         |              |               |            |                   | 0.7                |
+--------------------------------------------------+-----------------+---------------+--------------+---------------+------------+-------------------+--------------------+

.. note::

    The reported metrics are produced by ``references/table/evaluate.py`` using the
    :py:class:`TableCellMetric <doctr.utils.metrics.TableCellMetric>`: cell-detection **Recall**, **Precision** and
    **F1** (cells matched above an IoU threshold of 0.5), and **Structure acc**, the share of matched cells whose
    logical (row/column) coordinates are correctly predicted.

Table structure predictors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:meth:`table_predictor <doctr.models.table_structure.table_predictor>` wraps your table model so it can be used directly on
document images. For each page it returns the list of detected cells, each with its geometry, its confidence score and its logical coordinates, together with the inferred number of rows
and columns.

.. code:: python3

    import numpy as np
    from doctr.models import table_predictor
    model = table_predictor('tablecenternet', pretrained=True)
    table_crop = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    out = model([table_crop])
    # out[0] -> {"cells": [{"geometry": ..., "score": ..., "row_start": 0, "row_end": 0,
    #            "col_start": 0, "col_end": 0}, ...], "num_rows": ..., "num_cols": ...}


End-to-End OCR
--------------

The task consists of both localizing and transcribing textual elements in a given image.

Available OCR architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use any combination of detection and recognition models supported by docTR.

For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:

+---------------------------------------------------------------------------+----------------------------+----------------------------+
|                                                                           |        FUNSD               |        CORD                |
+===========================================================================+============================+============+===============+
| **Architecture**                                                          | **Recall** | **Precision** | **Recall** | **Precision** |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| db_resnet50 + crnn_vgg16_bn                                               | 73.37      | 76.11         | 84.80      | 79.09         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| db_resnet50 + crnn_mobilenet_v3_small                                     | 73.06      | 75.79         | 84.64      | 78.94         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| db_resnet50 + crnn_mobilenet_v3_large                                     | 73.17      | 75.90         | 84.96      | 79.25         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| db_resnet50 + master                                                      | 73.90      | 76.66         | 85.84      | 80.07         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| db_resnet50 + sar_resnet31                                                | 73.58      | 76.33         | 85.64      | 79.88         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| db_resnet50 + vitstr_small                                                | 73.06      | 75.79         | 85.95      | 80.17         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| db_resnet50 + vitstr_base                                                 | 73.70      | 76.46         | 85.76      | 79.99         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| db_resnet50 + parseq                                                      | 73.52      | 76.27         | 85.91      | 80.13         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| Gvision text detection                                                    | 59.50      | 62.50         | 75.30      | 59.03         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| Gvision doc. text detection                                               | 64.00      | 53.30         | 68.90      | 61.10         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| AWS textract                                                              | 78.10      | 83.00         | 87.50      | 66.00         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+
| Azure Form Recognizer (v3.2)                                              | 79.42      | 85.89         | 89.62      | 88.93         |
+---------------------------------------------------------------------------+------------+---------------+------------+---------------+


All OCR models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combined have 199 pages which might not be representative enough of the model capabilities*


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
* `detect_orientation`: If you want to estimate the general page orientation and add it to each page, set it to True.
* `detect_language`: If you want to predict the language of the text on each page, set it to True.
* `detect_layout`: If you want to run a layout detection model on each page and attach the detected regions to each page, set it to True (default: False).
* `layout_arch`: The layout architecture name (e.g. ``'lw_detr_s'``, ``'lw_detr_m'``) or your own (fine-tuned) layout model instance to use when ``detect_layout=True``.

For instance, this snippet instantiates an end-to-end ocr_predictor working with rotated documents, which preserves the aspect ratio of the documents, and returns polygons:

.. code:: python3

    from doctr.models import ocr_predictor
    model = ocr_predictor('linknet_resnet18', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)


Additionally, you can change the batch size of the underlying detection and recognition predictors to optimize the performance depending on your hardware:

* `det_bs`: batch size for the detection model (default: 2) - will also be used for the layout model if ``detect_layout=True``
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


Detecting the document layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to running the :py:meth:`layout_predictor <doctr.models.layout.layout_predictor>` standalone, you can plug a layout detection model directly into the end-to-end pipeline by setting ``detect_layout=True``. The detected regions (e.g. Title, Text, Table, Page-header, Page-footer) are attached to every :class:`Page <doctr.io.Page>` and can be accessed through ``page.layout``, exported alongside the rest of the page, and rendered with :py:meth:`show <doctr.io.Page.show>`.

.. code:: python3

    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    model = ocr_predictor(pretrained=True, detect_layout=True)
    doc = DocumentFile.from_images("path/to/your/doc.jpg")
    result = model(doc)

    # Access the detected layout regions of the first page
    for region in result.pages[0].layout:
        print(region.type, region.confidence, region.geometry)

    # The layout is part of the exported representation
    export = result.pages[0].export()
    print(export["layout"])

    # Overlay both text and layout regions (use display_layout=False to hide the regions)
    result.pages[0].show()

The same ``detect_layout`` / ``layout_arch`` arguments are available for the :py:meth:`kie_predictor <doctr.models.kie_predictor>`.


Running the predictors on GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can run the predictors on GPU by specifying the appropriate device.

Here's how to do it for both **NVIDIA** and **Apple Silicon (MPS)** GPUs:

.. code:: python3

    import torch
    from doctr.models import ocr_predictor

    # For NVIDIA GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = ocr_predictor(pretrained=True).to(device)
    # Alternatively: predictor = ocr_predictor(pretrained=True).cuda()

    # For Apple Silicon (MPS)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    predictor = ocr_predictor(pretrained=True).to(device)


The same approach applies to all standalone predictors:

* `recognition_predictor`
* `detection_predictor`
* `crop_orientation_predictor`
* `page_orientation_predictor`
* `layout_predictor`

Just create the predictor instance and move it to the appropriate device.
To enable **half-precision inference**, you can append `.half()` after moving the predictor to the device.


What should I do with the output?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ocr_predictor returns a `Document` object with a nested structure (with `Page`, `Block`, `Line`, `Word`, `Artefact`).
When ``detect_layout=True`` was passed, each `Page` additionally carries a list of `LayoutElement` regions under ``page.layout``.
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

To export the output as XML (hocr-format) you can use the `export_as_xml` method:

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
      <meta name="ocr-system" content="doctr 0.11.0" />
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
            # 1. The output structure needs to be the same as the input location predictions
            # 2. Be aware that the coordinates are relative and needs to be between 0 and 1
            return loc_preds

    my_hook = CustomHook()

    predictor = ocr_predictor(pretrained=True)
    # Add a hook in the middle of the pipeline
    predictor.add_hook(my_hook)
    # You can also add multiple hooks which will be executed sequentially
    for hook in [my_hook, my_hook, my_hook]:
        predictor.add_hook(hook)


* Restrict the recognition model to a subset of its vocabulary.

If you only expect text from one or more known languages, you can whitelist the corresponding vocabs so the
recognition model can no longer predict any character outside of them. This works with every recognition
architecture and with any predictor wrapping one (`ocr_predictor`, `kie_predictor`, `recognition_predictor`).
A whitelist can only restrict a model to characters it already knows: characters that are not part of the
model's own vocabulary are silently ignored, so make sure the model was trained on a vocab that covers the
languages you need (e.g. a multilingual model).

.. code:: python3

    from doctr.datasets import VOCABS
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    from doctr.models.utils import add_whitelist

    predictor = ocr_predictor(pretrained=True)

    # The recognition model can now only predict Polish/German characters
    handle = add_whitelist(predictor, [VOCABS["polish"], VOCABS["german"]])

    input_page = DocumentFile.from_images("path/to/your/image.png")
    out = predictor(input_page)

    # Restore the original, unconstrained decoding
    handle.remove()

The returned handle can also be used as a context manager, in which case the whitelist is removed on exit:

.. code:: python3

    with add_whitelist(predictor, VOCABS["german"]):
        out = predictor(input_page)  # only German characters can be predicted here
    # the whitelist is automatically removed outside of the ``with`` block

By default forbidden characters are dropped (``strategy="mask"``), so decoding falls back to the highest-scoring
allowed character. Alternatively, ``strategy="nearest"`` folds each forbidden character onto the closest allowed
one (e.g. ``ä`` -> ``a``, ``ł`` -> ``l``), which is useful to normalize accents/diacritics onto a base alphabet.
The mapping is built by transliteration by default; pass ``mapping="weights"`` to derive it from the model's own
learned confusions, or a ``{forbidden_char: allowed_char}`` dict to override specific characters.

.. code:: python3

    from doctr.datasets import VOCABS
    from doctr.models import ocr_predictor
    from doctr.models.utils import add_whitelist

    predictor = ocr_predictor(pretrained=True)

    # Fold any non-ASCII character onto its closest ASCII letter (e.g. é -> e, ł -> l)
    handle = add_whitelist(predictor, VOCABS["latin"], strategy="nearest")
    out = predictor(input_page)
    handle.remove()
