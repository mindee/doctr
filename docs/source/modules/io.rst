doctr.io
========


.. currentmodule:: doctr.io

The io module enables users to easily access content from documents and export analysis
results to structured formats.

.. _document_structure:

Document structure
------------------

Structural organization of the documents.

Word
^^^^
A Word is an uninterrupted sequence of characters.

.. autoclass:: Word

Prediction
^^^^^^^^^^
A Prediction is a Word with an additional crop orientation field indicating the detected text rotation angle.

.. autoclass:: Prediction

Line
^^^^
A Line is a collection of Words aligned spatially and meant to be read together (on a two-column page, on the same horizontal, we will consider that there are two Lines).

.. autoclass:: Line

Artefact
^^^^^^^^

An Artefact is a non-textual element (e.g. QR code, picture, chart, signature, logo, etc.).

.. autoclass:: Artefact

LayoutElement
^^^^^^^^^^^^^

A LayoutElement is a region predicted by a layout detection model (e.g. Title, Text, Table, Page-header, Page-footer). Layout regions are attached to a :class:`Page` when the ``ocr_predictor`` / ``kie_predictor`` is run with ``detect_layout=True``.

.. autoclass:: LayoutElement

Block
^^^^^
A Block is a collection of Lines (e.g. an address written on several lines) and Artefacts (e.g. a graph with its title underneath).

.. autoclass:: Block

Page
^^^^

A Page is a collection of Blocks that were on the same physical page.

.. autoclass:: Page

   .. automethod:: show
   .. automethod:: items_in_reading_order
   .. automethod:: export_as_markdown
   .. automethod:: export_as_asciidoc
   .. automethod:: export_as_html
   .. automethod:: export_as_xml
   .. automethod:: export
   .. automethod:: render
   .. automethod:: export_as


KIEPage
^^^^^^^

A KIEPage is returned by the :py:meth:`kie_predictor <doctr.models.kie_predictor>`. It groups predictions by
semantic class rather than by spatial layout.

.. autoclass:: KIEPage

   .. automethod:: show
   .. automethod:: export_as_markdown
   .. automethod:: export_as_asciidoc
   .. automethod:: export_as_html
   .. automethod:: export_as_xml
   .. automethod:: export
   .. automethod:: render
   .. automethod:: export_as


Document
^^^^^^^^

A Document is a collection of Pages.

.. autoclass:: Document

   .. automethod:: show
   .. automethod:: export_as_markdown
   .. automethod:: export_as_asciidoc
   .. automethod:: export_as_xml
   .. automethod:: export_as_html
   .. automethod:: export
   .. automethod:: render
   .. automethod:: export_as


KIEDocument
^^^^^^^^^^^

A KIEDocument is a collection of :class:`KIEPage` elements, returned by the
:py:meth:`kie_predictor <doctr.models.kie_predictor>`.

.. autoclass:: KIEDocument

   .. automethod:: show


File reading
------------

High-performance file reading and conversion to processable structured data.

.. autofunction:: read_pdf

.. autofunction:: read_img_as_numpy

.. autofunction:: read_img_as_tensor

.. autofunction:: decode_img_as_tensor

.. autofunction:: read_html


.. autoclass:: DocumentFile

   .. automethod:: from_pdf

   .. automethod:: from_url

   .. automethod:: from_images


.. _reading_order:

Reading order
-------------

The reading-order-aware export of a :class:`Document` / :class:`Page` to Markdown, AsciiDoc, HTML, or XML is available
through the ``export_as_markdown`` / ``export_as_asciidoc`` / ``export_as_html`` / ``export_as_xml`` / ``export_as`` / ``export`` / ``render`` methods documented above, which
delegate to the exporters of :mod:`doctr.io.exporters`.
The underlying ordering primitives live in :mod:`doctr.models.reading_order`.

Every export path shares the same linearization, so ``render()``, ``export()``, ``export_as_xml()`` and the
Markdown / AsciiDoc / HTML exports all present the content in the same order. The result is memoized on the
page, so exporting one page to several formats orders it only once.

.. currentmodule:: doctr.io

.. autoclass:: TextExporter
    :members: export_page, export_kie_page, export_document

.. autoclass:: MarkdownExporter
    :members: export_page, export_kie_page, export_document

.. autoclass:: AsciiDocExporter
    :members: export_page, export_kie_page, export_document

.. autoclass:: HTMLExporter
    :members: export_page, export_kie_page, export_document

.. autoclass:: XMLExporter
    :members: export_page, export_kie_page, export_document

.. autofunction:: doctr.io.exporters.page_reading_order

.. autofunction:: doctr.io.exporters.predictions_in_reading_order

.. autofunction:: doctr.io.exporters.to_json_safe

Figures
-------

When the predictor runs with ``detect_layout=True``, the figures found by the layout model take part in the
reading order and are materialized by the Markdown, AsciiDoc and HTML exports. How they are materialized is
controlled by the ``images`` argument, which accepts either an image mode or a configured
:class:`FigureEncoder`:

* ``'placeholder'`` (the default): a comment marks where a figure was detected, without touching the pixels
* ``'none'``: the figures are dropped entirely
* ``'embedded'``: each figure is cropped out of the page and inlined as a base64 data URI
* ``'referenced'``: each crop is written next to the export and referenced by a relative path

.. code:: python

    from doctr.io import DocumentFile, FigureEncoder
    from doctr.models import ocr_predictor

    predictor = ocr_predictor(pretrained=True, detect_layout=True)
    doc = predictor(DocumentFile.from_pdf("report.pdf"))

    # A self-contained Markdown file
    markdown = doc.export_as_markdown(images="embedded")
    # ... or one that points at the crops on disk
    markdown = doc.export_as_markdown(images=FigureEncoder("referenced", image_dir="assets", path_prefix="assets/"))

A caption detected next to a figure becomes its alternative text (and its ``<figcaption>`` in HTML) as soon as
the export carries the pixels. Plain text and the hOCR export never inline an image: hOCR positions each figure
as an ``ocr_photo`` area instead. Pages restored from a JSON export carry no pixels, so their figures fall back
to a placeholder.

.. autoclass:: FigureEncoder
    :members: resolve, source, enabled, materializes, materializes_on

.. autofunction:: crop_layout_region

.. autofunction:: encode_crop

.. autofunction:: picture_regions

.. autofunction:: is_picture_region

.. autofunction:: is_picture_label
