Using the CLI for Optical Character Recognition
===============================================

The full Optical Character Recognition (OCR) task can be executed by using the Command Line Interface (CLI) implemented in docTR. This tool allows you to process both images and PDF files without writing a single line of Python code, and to export the results as JSON, plain text, Markdown, AsciiDoc, HTML or hOCR.

Basic Usage
-----------

To run the OCR engine on a file, use the following command structure:

.. code-block:: bash

    doctr-cli --input_path path/to/your/document.pdf --output results.json

Arguments
---------

The CLI supports a variety of arguments to fine-tune the detection, recognition and export process:

**Mandatory Arguments:**

* ``--input_path``: Path to one or more input images / PDF files. Every page is appended to a single document, so several files can be processed in one go. Values starting with ``http://`` or ``https://`` are interpreted as web pages and rendered as PDF (this requires ``weasyprint``, cf. :meth:`doctr.io.DocumentFile.from_url`).

**General Options:**

* ``--version``: Prints the installed docTR version and exits.

**Input Options:**

* ``--pdf_password``: Password used to unlock encrypted PDF files. *Default: None*
* ``--pdf_scale``: PDF rendering scale, where 1 corresponds to 72dpi. *Default: 2*

**Architecture Selection:**

* ``--det_arch``: The detection architecture to use (e.g., ``fast_base``). *Default: fast_base*
* ``--reco_arch``: The recognition architecture to use (e.g., ``crnn_vgg16_bn``). *Default: crnn_vgg16_bn*
* ``--layout_arch``: The layout architecture to use. *Default: lw_detr_s*
* ``--device``: Device the models are loaded on (``auto``, ``cpu``, ``cuda:0``, ...). On a CUDA device of compute capability 8.0 or above, the models are cast to bfloat16. *Default: auto*

**Processing Options:**

* ``--assume_straight_pages``, ``--no-assume_straight_pages``: Determine whether pages should be handled as straight or skewed pages. *Default: True*
* ``--straighten_pages``: If flagged, the tool will attempt to straighten skewed pages before analysis. *Default: False*
* ``--preserve_aspect_ratio``, ``--no-preserve_aspect_ratio``: Ensures that the aspect ratio is maintained during resizing. *Default: True*
* ``--symmetric_pad``, ``--no-symmetric_pad``: Applies symmetric padding to the input images. *Default: True*
* ``--det_bs``: Batch size used for the detection model. *Default: 2*
* ``--reco_bs``: Batch size used for the recognition model. *Default: 128*
* ``--bin_thresh``: Binarization threshold of the detection post-processing. *Default: the value of the architecture*
* ``--box_thresh``: Minimum confidence of a detected box. *Default: the value of the architecture*
* ``--detect_orientation``: Enables automatic detection of page orientation. *Default: False*
* ``--detect_language``: Enables language detection for the extracted text. *Default: False*
* ``--detect_layout``: Attaches the detected layout regions to each page. *Default: False*
* ``--detect_tables``: Regroups the words of the detected tables into structured tables. This enables the layout model. *Default: False*
* ``--ignore_regions``: Layout class names to mask out before detection & recognition (e.g. ``--ignore_regions Picture Table``). This enables the layout model. *Default: None*
* ``--export_as_straight_boxes``: Exports (potentially rotated) predictions as straight boxes. *Default: False*
* ``--preserve_original_coords``: Maps the boxes back to the original page coordinates when ``--straighten_pages`` is used. *Default: False*
* ``--disable_page_orientation``: Disables the page orientation classifier. *Default: False*
* ``--disable_crop_orientation``: Disables the crop orientation classifier. *Default: False*

**Document Assembling Options:**

* ``--resolve_lines``, ``--no-resolve_lines``: Groups the words into lines. *Default: True*
* ``--resolve_blocks``: Groups the lines into blocks. *Default: False*
* ``--paragraph_break``: Relative length of the minimum space separating two paragraphs. *Default: 0.035*
* ``--keep_reading_order``: Arranges the content of every page in reading order. *Default: False*

**Output Options:**

* ``--output``: The destination path where the results will be saved. *Default: results.json*
* ``--format``: The export format, one of ``json``/``dict``, ``txt``/``text``, ``md``/``markdown``, ``adoc``/``asciidoc``, ``html``, ``xml``/``hocr``. When it is not set, the format is inferred from the extension of ``--output``, and falls back to JSON. *Default: None*
* ``--direction``: Reading direction of the document, one of ``auto``, ``ltr``, ``rtl``, ``ttb-rtl``, ``ttb-ltr``. *Default: auto*
* ``--reading_order``, ``--no-reading_order``: Linearizes the content in reading order (JSON & hOCR exports). *Default: True*
* ``--escape``, ``--no-escape``: Escapes the characters carrying a structural meaning (Markdown & AsciiDoc exports). *Default: True*
* ``--include_furniture``, ``--no-include_furniture``: Includes page headers, page footers and footnotes (text-like exports). *Default: True*
* ``--file_title``: Title of the exported hOCR files. *Default: docTR - XML export (hOCR)*
* ``--indent``: Indentation of the JSON export. *Default: 4*
* ``--quiet``: Only logs errors. *Default: False*

Examples
--------

**Running OCR on an image:**

.. code-block:: bash

    doctr-cli --input_path image.jpg --output ocr_res.json

**Running OCR on several files at once:**

.. code-block:: bash

    doctr-cli --input_path page1.jpg page2.jpg scan.pdf --output ocr_res.json

**Using a specific detection architecture and straightening pages:**

.. code-block:: bash

    doctr-cli --input_path doc.pdf --det_arch db_mobilenet_v3_large --straighten_pages

**Exporting the text as Markdown, with tables and layout:**

.. code-block:: bash

    doctr-cli --input_path doc.pdf --detect_tables --output doc.md

**Exporting as hOCR on the GPU, ignoring the pictures of the document:**

.. code-block:: bash

    doctr-cli --input_path doc.pdf --device cuda:0 --ignore_regions Picture --output doc.xml

Output Format
-------------

By default, the results are exported in a structured JSON format containing:

* **Pages**: Dimensions, orientation, language and layout regions.
* **Blocks**: Grouping of lines.
* **Lines**: Grouping of words.
* **Words**: The actual text content with confidence scores and bounding box coordinates.
* **Tables**: The structured tables, when ``--detect_tables`` is used.

The text-like formats (``txt``, ``md``, ``adoc``, ``html``) are written to a single file. Since hOCR describes a single page per file, the ``xml`` export writes one file per page: an ``--output`` of ``doc.xml`` on a 3-page document produces ``doc_page_1.xml``, ``doc_page_2.xml`` and ``doc_page_3.xml``.
