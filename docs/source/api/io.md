# doctr.io

```{eval-rst}
.. module:: doctr.io
```

Read documents into pages, represent the predictions as a tree of elements, and export them.

## Reading files

```{eval-rst}
.. autoclass:: DocumentFile

   .. automethod:: from_pdf
   .. automethod:: from_images
   .. automethod:: from_url
   .. automethod:: from_files

.. autoclass:: PageSource

   .. automethod:: iter_batches
   .. automethod:: close

.. autoclass:: PDFSource

.. autoclass:: ImageSource

.. autoclass:: ConcatSource

   .. automethod:: locate

.. autoclass:: PageDecodeError

.. autofunction:: read_pdf

.. autofunction:: read_img_as_numpy

.. autofunction:: read_img_as_tensor

.. autofunction:: decode_img_as_tensor

.. autofunction:: read_html
```

## Document structure

The elements returned by the predictors, from the largest to the smallest. See
[](../getting_started/concepts.md#the-document-tree) for an overview.

### Document

```{eval-rst}
.. autoclass:: Document

   .. automethod:: render
   .. automethod:: export
   .. automethod:: export_as
   .. automethod:: export_as_markdown
   .. automethod:: export_as_asciidoc
   .. automethod:: export_as_html
   .. automethod:: export_as_xml
   .. automethod:: show
   .. automethod:: synthesize
   .. automethod:: from_dict
```

### Page

```{eval-rst}
.. autoclass:: Page

   .. automethod:: render
   .. automethod:: export
   .. automethod:: export_as
   .. automethod:: export_as_markdown
   .. automethod:: export_as_asciidoc
   .. automethod:: export_as_html
   .. automethod:: export_as_xml
   .. automethod:: items_in_reading_order
   .. automethod:: show
   .. automethod:: synthesize
```

### Block, Line and Word

```{eval-rst}
.. autoclass:: Block

.. autoclass:: Line

.. autoclass:: Word

.. autoclass:: Artefact
```

### Layout and tables

```{eval-rst}
.. autoclass:: LayoutElement

.. autoclass:: Table

   .. automethod:: to_grid
   .. automethod:: render

.. autoclass:: TableCell
   :members: row_span, col_span
```

### Key information extraction

Returned by {func}`~doctr.models.kie_predictor`: predictions are grouped by class instead of by spatial layout.

```{eval-rst}
.. autoclass:: KIEDocument

   .. automethod:: show

.. autoclass:: KIEPage

   .. automethod:: render
   .. automethod:: export
   .. automethod:: export_as
   .. automethod:: export_as_markdown
   .. automethod:: export_as_asciidoc
   .. automethod:: export_as_html
   .. automethod:: export_as_xml
   .. automethod:: show

.. autoclass:: Prediction
```

## Exporters

The `export_as_*` methods of the elements delegate to these exporters, which share the same reading-order
linearization. See [](../user_guide/exporting.md).

```{eval-rst}
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

.. autofunction:: page_reading_order
```

## Figures

Rendering of the figures found by the layout model in the Markdown, AsciiDoc and HTML exports. See
[](../user_guide/exporting.md#figures).

```{eval-rst}
.. autoclass:: FigureEncoder
   :members: resolve, source, enabled, materializes

.. autofunction:: crop_layout_region

.. autofunction:: encode_crop

.. autofunction:: picture_regions

.. autofunction:: is_picture_region

.. autofunction:: is_picture_label
```
