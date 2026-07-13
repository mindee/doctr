# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Callable, Collection
from typing import Any

import numpy as np

from doctr.io.elements import Document, KIEPage
from doctr.models.builder import DocumentBuilder
from doctr.utils.geometry import (
    extract_crops,
    extract_rcrops,
    remove_image_padding,
    rotate_image,
    straighten_page,
)

from .._utils import estimate_orientation, mask_boxes, rectify_crops, rectify_loc_preds
from ..classification import crop_orientation_predictor, page_orientation_predictor
from ..classification.predictor import OrientationPredictor

__all__ = ["_OCRPredictor"]


class _OCRPredictor:
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
        **kwargs: keyword args of `DocumentBuilder`
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
        ignore_regions: Collection[str] | None = None,
        preserve_original_coords: bool = False,
        **kwargs: Any,
    ) -> None:
        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages
        self.preserve_original_coords = preserve_original_coords
        self._page_orientation_disabled = kwargs.pop("disable_page_orientation", False)
        self._crop_orientation_disabled = kwargs.pop("disable_crop_orientation", False)
        self.crop_orientation_predictor = (
            None
            if assume_straight_pages
            else crop_orientation_predictor(pretrained=True, disabled=self._crop_orientation_disabled)
        )
        self.page_orientation_predictor = (
            page_orientation_predictor(pretrained=True, disabled=self._page_orientation_disabled)
            if detect_orientation or straighten_pages or not assume_straight_pages
            else None
        )
        self.doc_builder = DocumentBuilder(**kwargs)

        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

        self.ignore_regions = ignore_regions

        self.hooks: list[Callable] = []

    def _general_page_orientations(
        self,
        pages: list[np.ndarray],
    ) -> list[tuple[int, float]]:
        _, classes, probs = zip(self.page_orientation_predictor(pages))  # type: ignore[misc]
        # Flatten to list of tuples with (value, confidence)
        page_orientations = [
            (orientation, prob)
            for page_classes, page_probs in zip(classes, probs)
            for orientation, prob in zip(page_classes, page_probs)
        ]
        return page_orientations

    def _get_orientations(
        self, pages: list[np.ndarray], seg_maps: list[np.ndarray]
    ) -> tuple[list[tuple[int, float]], list[int]]:
        general_pages_orientations = self._general_page_orientations(pages)
        origin_page_orientations = [
            estimate_orientation(seq_map, general_orientation)
            for seq_map, general_orientation in zip(seg_maps, general_pages_orientations)
        ]
        return general_pages_orientations, origin_page_orientations

    def _mask_regions(
        self,
        pages: list[np.ndarray],
        regions: list[dict[str, Any]] | None,
        fill_value: int = 0,
    ) -> list[np.ndarray]:
        if regions is None or not self.ignore_regions:
            return pages

        out_pages: list[np.ndarray] = []
        for page, region in zip(pages, regions):
            if region is None:
                out_pages.append(page)
                continue
            boxes = np.asarray(region["boxes"], dtype=np.float32)
            keep = np.array([name in self.ignore_regions for name in region["class_names"]], dtype=bool)
            out_pages.append(mask_boxes(page, boxes[keep], fill_value=fill_value) if keep.any() else page)

        return out_pages

    def _straighten_pages(
        self,
        pages: list[np.ndarray],
        seg_maps: list[np.ndarray],
        general_pages_orientations: list[tuple[int, float]] | None = None,
        origin_pages_orientations: list[int] | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        general_pages_orientations = (
            general_pages_orientations if general_pages_orientations else self._general_page_orientations(pages)
        )
        origin_pages_orientations = (
            origin_pages_orientations
            if origin_pages_orientations
            else [
                estimate_orientation(seq_map, general_orientation)
                for seq_map, general_orientation in zip(seg_maps, general_pages_orientations)
            ]
        )
        if not self.preserve_original_coords:
            # expand if height and width are not equal, then remove the padding
            return [
                remove_image_padding(rotate_image(page, angle, expand=page.shape[0] != page.shape[1]))
                for page, angle in zip(pages, origin_pages_orientations)
            ], []

        m_invs = []
        out = []
        for page, angle in zip(pages, origin_pages_orientations):
            straightened, m_inv = straighten_page(page, angle)
            m_invs.append(m_inv)
            out.append(straightened)
        return out, m_invs

    @staticmethod
    def _generate_crops(
        pages: list[np.ndarray],
        loc_preds: list[np.ndarray],
        assume_straight_pages: bool = False,
        assume_horizontal: bool = False,
    ) -> list[list[np.ndarray]]:
        if assume_straight_pages:
            crops = [extract_crops(page, _boxes[:, :4]) for page, _boxes in zip(pages, loc_preds)]
        else:
            crops = [
                extract_rcrops(page, _boxes[:, :4], assume_horizontal=assume_horizontal)
                for page, _boxes in zip(pages, loc_preds)
            ]
        return crops

    @staticmethod
    def _prepare_crops(
        pages: list[np.ndarray],
        loc_preds: list[np.ndarray],
        assume_straight_pages: bool = False,
        assume_horizontal: bool = False,
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        crops = _OCRPredictor._generate_crops(pages, loc_preds, assume_straight_pages, assume_horizontal)

        # Avoid sending zero-sized crops
        is_kept = [[all(s > 0 for s in crop.shape) for crop in page_crops] for page_crops in crops]
        crops = [
            [crop for crop, _kept in zip(page_crops, page_kept) if _kept]
            for page_crops, page_kept in zip(crops, is_kept)
        ]
        loc_preds = [_boxes[_kept] for _boxes, _kept in zip(loc_preds, is_kept)]

        return crops, loc_preds

    def _rectify_crops(
        self,
        crops: list[list[np.ndarray]],
        loc_preds: list[np.ndarray],
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[tuple[int, float]]]:
        page_counts = [len(page_crops) for page_crops in crops]
        flat_crops = [crop for page_crops in crops for crop in page_crops]
        flat_orientations, flat_classes, flat_probs = self.crop_orientation_predictor(flat_crops)  # type: ignore[misc]
        _bounds = np.cumsum([0, *page_counts])
        orientations = [flat_orientations[start:end] for start, end in zip(_bounds[:-1], _bounds[1:])]
        rect_crops = [rectify_crops(page_crops, orientation) for page_crops, orientation in zip(crops, orientations)]
        rect_loc_preds = [
            rectify_loc_preds(page_loc_preds, orientation) if len(page_loc_preds) > 0 else page_loc_preds
            for page_loc_preds, orientation in zip(loc_preds, orientations)
        ]
        # Flatten to list of tuples with (value, confidence)
        crop_orientations = [(orientation, prob) for orientation, prob in zip(flat_classes, flat_probs)]
        return rect_crops, rect_loc_preds, crop_orientations  # type: ignore[return-value]

    @staticmethod
    def _process_predictions(
        loc_preds: list[np.ndarray],
        word_preds: list[tuple[str, float]],
        crop_orientations: list[dict[str, Any]],
    ) -> tuple[list[np.ndarray], list[list[tuple[str, float]]], list[list[dict[str, Any]]]]:
        text_preds = []
        crop_orientation_preds = []
        if len(loc_preds) > 0:
            # Text & crop orientation predictions at page level
            _idx = 0
            for page_boxes in loc_preds:
                text_preds.append(word_preds[_idx : _idx + page_boxes.shape[0]])
                crop_orientation_preds.append(crop_orientations[_idx : _idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]

        return loc_preds, text_preds, crop_orientation_preds

    @staticmethod
    def _remap_geometry(obj: Any, m_inv: np.ndarray, sw: int, sh: int, oh: int, ow: int) -> None:
        """Remap a single geometry-bearing object from straightened to original coords."""
        pts = np.asarray(obj.geometry, dtype=np.float64).reshape(-1, 2)
        was_straight = pts.shape[0] == 2
        if was_straight:
            (x0, y0), (x1, y1) = pts
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        pts[:, 0] *= sw
        pts[:, 1] *= sh
        homo = np.column_stack([pts, np.ones(len(pts))])
        orig = (homo @ m_inv.T)[:, :2]
        orig[:, 0] = orig[:, 0].clip(0, ow - 1) / ow
        orig[:, 1] = orig[:, 1].clip(0, oh - 1) / oh
        if was_straight:
            obj.geometry = (
                (float(orig[:, 0].min()), float(orig[:, 1].min())),
                (float(orig[:, 0].max()), float(orig[:, 1].max())),
            )
        else:
            obj.geometry = tuple(tuple(r) for r in orig.tolist())

    def _remap_to_original_coords(
        self,
        out: Document,
        orig_shapes: list[tuple[int, int]],
        straight_shapes: list[tuple[int, int]],
        m_invs: list[np.ndarray],
        orig_pages: list[np.ndarray] | None = None,
    ) -> Document:
        """Remap word geometries from straightened-page coordinates back to original page coordinates.

        Args:
            out: the document returned by the builder
            orig_shapes: original (pre-straightening) page shapes (H, W)
            straight_shapes: straightened page shapes (H, W)
            m_invs: inverse affine matrices from the straightening pipeline, one per page
            orig_pages: optional list of original page images to restore on `page.page`

        Returns:
            the document with remapped word geometries

        """
        for pidx, (page, m_inv) in enumerate(zip(out.pages, m_invs)):
            sh, sw = straight_shapes[pidx]  # straightened_height, straightened_width
            oh, ow = orig_shapes[pidx]  # original_height, original_width

            if isinstance(page, KIEPage):
                for preds in page.predictions.values():
                    for pred in preds:
                        self._remap_geometry(pred, m_inv, sw, sh, oh, ow)
                for region in page.layout:
                    self._remap_geometry(region, m_inv, sw, sh, oh, ow)
            else:
                for block in page.blocks:
                    self._remap_geometry(block, m_inv, sw, sh, oh, ow)
                    for artefact in block.artefacts:
                        self._remap_geometry(artefact, m_inv, sw, sh, oh, ow)
                    for line in block.lines:
                        self._remap_geometry(line, m_inv, sw, sh, oh, ow)
                        for word in line.words:
                            self._remap_geometry(word, m_inv, sw, sh, oh, ow)
                for region in page.layout:
                    self._remap_geometry(region, m_inv, sw, sh, oh, ow)
                for table in page.tables:
                    self._remap_geometry(table, m_inv, sw, sh, oh, ow)
                    for cell in table.cells:
                        self._remap_geometry(cell, m_inv, sw, sh, oh, ow)

            if orig_pages is not None:
                out.pages[pidx].page = orig_pages[pidx]
                out.pages[pidx].dimensions = orig_shapes[pidx]

        return out

    def add_hook(self, hook: Callable) -> None:
        """Add a hook to the predictor

        Args:
            hook: a callable that takes as input the `loc_preds` and returns the modified `loc_preds`
        """
        self.hooks.append(hook)
