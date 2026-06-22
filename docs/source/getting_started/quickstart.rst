
**********
Quickstart
**********

This page shows you how to get OCR results from a document in just a few lines of code.
For more details see :ref:`using_models`.


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


Multi-page PDF end-to-end example
==================================

The following snippet processes every page of a PDF and collects the plain-text output:

.. code:: python3

    import json
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_pdf("path/to/multi_page.pdf")
    result = model(doc)

    # Plain-text — one string per page
    for page_idx, page in enumerate(result.pages):
        print(f"--- Page {page_idx + 1} ---")
        print(page.render())

    # Structured output — JSON-serialisable dict
    output = result.export()
    with open("ocr_output.json", "w") as f:
        json.dump(output, f, indent=2)


Common pitfalls
===============

.. note::

   * **Visualization** requires the ``viz`` extra (installs ``matplotlib`` and ``mplcursors``):
     ``pip install "python-doctr[viz]"``.  Calls to ``result.show()`` or
     ``result.pages[0].show()`` raise a ``ModuleNotFoundError`` without it.
   * **HTML input** requires the ``html`` extra: ``pip install "python-doctr[html]"``.
   * **Image format**: pass file paths or NumPy ``uint8`` arrays shaped ``(H, W, C)`` in
     RGB order.  Grayscale arrays must be converted to 3-channel before use.
   * **Pretrained weights** are downloaded on first use and cached locally by
     Hugging Face Hub.  Subsequent calls are instantaneous.
   * **PDF pages are returned as images**: ``DocumentFile.from_pdf`` returns one
     NumPy array per page, so ``result.pages[i]`` corresponds to the *i*-th PDF page.


Next steps
==========

* :doc:`../using_doctr/using_models` — full predictor guide, architecture benchmarks, GPU usage.
* :doc:`../using_doctr/custom_models_training` — train and load your own models.
* :doc:`../using_doctr/sharing_models` — share your trained models on Hugging Face Hub.
