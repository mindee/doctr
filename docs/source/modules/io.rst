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


KIEPage
^^^^^^^

A KIEPage is returned by the :py:meth:`kie_predictor <doctr.models.kie_predictor>`. It groups predictions by
semantic class rather than by spatial layout.

.. autoclass:: KIEPage

   .. automethod:: show


Document
^^^^^^^^

A Document is a collection of Pages.

.. autoclass:: Document

   .. automethod:: show


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
