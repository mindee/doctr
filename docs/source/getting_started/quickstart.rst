
**********
Quickstart
**********

This page shows you how to get OCR results from a document in just a few lines of code.
For more details see :doc:`../using_doctr/using_models`.


Load a document
===============

docTR can read PDFs, images, and web pages:

.. code:: python3

    from doctr.io import DocumentFile

    # From a PDF
    doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
    # From one or more images
    doc = DocumentFile.from_images("path/to/your/img.jpg")
    doc = DocumentFile.from_images(["path/to/page1.jpg", "path/to/page2.jpg"])
    # From a URL (requires the ``html`` extra: pip install "python-doctr[html]")
    doc = DocumentFile.from_url("https://www.example.com")


Run OCR
=======

.. code:: python3

    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
    model = ocr_predictor(pretrained=True)
    result = model(doc)

The predictor uses ``db_resnet50`` for text detection and ``crnn_vgg16_bn`` for text recognition by default.
You can choose any combination of :ref:`supported architectures <using_models>`.


Inspect the output
==================

The result is a :class:`~doctr.io.Document` object.

Render as plain text::

    print(result.render())

Export as a nested dictionary (JSON-serialisable)::

    import json
    print(json.dumps(result.export(), indent=2))

Visualise on screen (requires the ``viz`` extra: ``pip install "python-doctr[viz]"``)::

    result.pages[0].show()

