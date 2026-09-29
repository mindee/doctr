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

When the predictor runs the layout model (``detect_layout=True`` or ``detect_tables=True``), the figures it finds
take part in the reading order and are materialized by the Markdown, AsciiDoc and HTML exports. How they are
materialized is controlled by the ``images`` argument, which accepts either an image mode or a configured
:class:`FigureEncoder`:

* ``'placeholder'`` (the default): a comment marks where a figure was detected, without touching the pixels
  (``<!-- image -->`` in Markdown and HTML, ``// image`` in AsciiDoc). The text recognized inside the figure is
  kept. Pass ``images="none"`` to get the exports without any figure marker.
* ``'none'``: the figures are left out of the export (they still take part in the reading order). The text
  recognized inside the figure is kept.
* ``'embedded'``: each figure is cropped out of the page and inlined as a base64 data URI. The text recognized
  inside the figure is **dropped** from the export.
* ``'referenced'``: each crop is written next to the export and referenced by a relative path. This mode needs a
  directory to write to, so it is only available through a configured encoder:
  ``images=FigureEncoder("referenced", image_dir=...)`` (the ``images="referenced"`` string raises). The file names
  carry the position of the figure and a hash of its content (e.g. ``page1_figure2-3fa2b1c9.png``), so several
  documents can share the same image directory. The text recognized inside the figure is **dropped** from the
  export. ``path_prefix`` accepts a string or a path, with or without a trailing separator, and Windows
  backslashes are written as forward slashes, so the links work on Linux, macOS and Windows alike.

.. note::
    Once a figure carries its pixels (``'embedded'`` and ``'referenced'``), the text the OCR recognized inside it
    (axis labels, legends, the words of a screenshot, often fragmented OCR noise) is already visible in the
    image, so it is left out of the export rather than scattered through the body text. Only the lines labeled
    as part of the picture by the layout model are dropped: the caption becomes the image's alternative text,
    and the text of a figure which could not be cropped (a degenerate region, or a page restored from a JSON
    export) is kept. Use ``'placeholder'`` or ``'none'`` to keep all the recognized text, and
    ``page.items_in_reading_order(include_figures=True)`` to access it along with the figures.

.. code:: python

    from doctr.io import DocumentFile, FigureEncoder
    from doctr.models import ocr_predictor

    predictor = ocr_predictor(pretrained=True, detect_layout=True)
    doc = predictor(DocumentFile.from_pdf("report.pdf"))

    # A self-contained Markdown file
    markdown = doc.export_as_markdown(images="embedded")
    # ... or one that points at the crops on disk
    markdown = doc.export_as_markdown(images=FigureEncoder("referenced", image_dir="assets", path_prefix="assets"))

A caption detected next to a figure becomes its alternative text as soon as the export carries the pixels, and
stays visible: an italic paragraph right under the image in Markdown, a ``<figcaption>`` in HTML and a block title
in AsciiDoc. Plain text and the hOCR export never inline an image: hOCR positions each figure
as an ``ocr_photo`` area instead. Pages restored from a JSON export carry no pixels, so their figures fall back
to a placeholder. With ``include_furniture=False``, the figures lying in a page header or footer (e.g. a logo)
are left out along with the rest of the page furniture.

The command line exposes the same options: ``--images`` picks the mode, and in ``referenced`` mode the crops are
written to ``--image_dir`` (by default a ``<output name>_images`` directory next to the output file), in the
``--image_format`` of your choice:

.. code:: bash

    doctr-cli --input_path report.pdf --detect_layout --output report.md --images referenced

.. autoclass:: FigureEncoder
    :members: resolve, source, enabled, materializes, materializes_on

.. autofunction:: crop_layout_region

.. autofunction:: encode_crop

.. autofunction:: picture_regions

.. autofunction:: is_picture_region

.. autofunction:: is_picture_label
