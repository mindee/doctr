# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Collection
from typing import Any

from doctr.io.elements import Document, KIEDocument
from doctr.models.builder import KIEDocumentBuilder

from ..classification.predictor import OrientationPredictor
from ..predictor.base import _OCRPredictor

__all__ = ["_KIEPredictor"]


class _KIEPredictor(_OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        preserve_aspect_ratio: if True, resize preserving the aspect ratio (with padding)
        symmetric_pad: if True and preserve_aspect_ratio is True, pas the image symmetrically.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        ignore_regions: optional list of layout class names to ignore during detection/recognition. If provided, the
            layout model will be used to locate the regions of the specified classes, and these regions will
            be masked out (filled with black) before passing the pages to the detection/recognition modules.
        preserve_original_coords: if True and straighten_pages is True, bounding boxes are mapped back to the
            original page coordinates. Useful for redaction and annotation.
        page_chunk_size: the maximum number of pages processed at once. Longer inputs, as well as lazy page sources
            and iterators, are processed chunk by chunk so that the memory footprint does not grow with the number
            of pages.
        kwargs: keyword args of `DocumentBuilder`
    """

    crop_orientation_predictor: OrientationPredictor | None
    page_orientation_predictor: OrientationPredictor | None
    _document_cls: type[Document] = KIEDocument

    def __init__(
        self,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        detect_orientation: bool = False,
        ignore_regions: Collection[str] | None = None,
        preserve_original_coords: bool = False,
        page_chunk_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            assume_straight_pages,
            straighten_pages,
            preserve_aspect_ratio,
            symmetric_pad,
            detect_orientation,
            ignore_regions,
            preserve_original_coords=preserve_original_coords,
            page_chunk_size=page_chunk_size,
            **kwargs,
        )

        # Remove the following arguments from kwargs after initialization of the parent class
        kwargs.pop("disable_page_orientation", None)
        kwargs.pop("disable_crop_orientation", None)

        self.doc_builder: KIEDocumentBuilder = KIEDocumentBuilder(**kwargs)
