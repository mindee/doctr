Using the CLI for Optical Character Recognition
===============================================

The full Optical Character Recognition (OCR) task can be executed by using the Command Line Interface (CLI) implemented in docTR. This tool allows you to process both images and PDF files without writing a single line of Python code, providing a streamlined way to export OCR results directly to JSON.

Basic Usage
-----------

To run the OCR engine on a file, use the following command structure:

.. code-block:: bash

    doctr-cli --input_path path/to/your/document.pdf --output results.json

Arguments
---------

The CLI supports a variety of arguments to fine-tune the detection and recognition process:

**Mandatory Arguments:**

* ``--input_path``: Path to the input image or PDF file you wish to process.

**Architecture Selection:**

* ``--det_arch``: The detection architecture / model to use (e.g., ``db_resnet50``). *Default: db_resnet50*
* ``--reco_arch``: The recognition architecture / model to use (e.g., ``crnn_vgg16_bn``). *Default: crnn_vgg16_bn*

**Processing Options:**

* ``--assume_straight_pages``, ``--no-assume_straight_pages``: Determine whether pages should be handled as straight or skewed pages. *Default: True*
* ``--straighten_pages``: If flagged, the tool will attempt to straighten skewed pages before analysis. *Default: True*
* ``--preserve_aspect_ratio``, ``--no-preserve_aspect_ratio``: Ensures that the aspect ratio is maintained during resizing. *Default: True*
* ``--symmetric_pad``: Applies symmetric padding to the input images. *Default: True*
* ``--det_bs``: Batch size used for the detection model. *Default: 2*
* ``--reco_bs``: Batch size used for the recognition model. *Default: 128*
* ``--detect_orientation``: Enables automatic detection of page orientation. *Default: False*
* ``--detect_language``: Enables language detection for the extracted text. *Default: False*

**Output Options:**

* ``--output``: The destination path where the JSON results will be saved. *Default: results.json*

Examples
--------

**Running OCR on an image:**

.. code-block:: bash

    doctr-cli --input_path image.jpg --output ocr_res.json

**Running OCR on a PDF:**

.. code-block:: bash

    doctr-cli --input_path image.pdf --output ocr_res.json

**Using a specific detection architecture and straightening pages:**

.. code-block:: bash

    doctr-cli --input_path doc.pdf --det_arch db_mobilenet_v3_large --straighten_pages

Output Format
-------------

The results are exported in a structured JSON format containing:

* **Pages**: Dimensions and orientation.
* **Blocks**: Grouping of lines.
* **Lines**: Grouping of words.
* **Words**: The actual text content with confidence scores and bounding box coordinates.
