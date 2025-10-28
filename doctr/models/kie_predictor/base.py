# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

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
        kwargs: keyword args of `DocumentBuilder`
    """

    crop_orientation_predictor: OrientationPredictor | None
    page_orientation_predictor: OrientationPredictor | None

    def __init__(
        self,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        detect_orientation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            assume_straight_pages, straighten_pages, preserve_aspect_ratio, symmetric_pad, detect_orientation, **kwargs
        )

        # Remove the following arguments from kwargs after initialization of the parent class
        kwargs.pop("disable_page_orientation", None)
        kwargs.pop("disable_crop_orientation", None)

        self.doc_builder: KIEDocumentBuilder = KIEDocumentBuilder(**kwargs)
