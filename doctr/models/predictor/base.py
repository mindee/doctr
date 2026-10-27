# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from doctr.io.elements import Document, KIEPage, Page
from doctr.io.sources import PageSource, _EncodedImage, _SourcePage
from doctr.models.builder import DocumentBuilder
from doctr.utils.geometry import (
    extract_crops,
    extract_rcrops,
    remove_image_padding,
    rotate_image,
    straighten_page,
)
from doctr.utils.multithreading import chunked, prefetch

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
        page_chunk_size: the maximum number of pages processed at once. Longer inputs, as well as lazy page sources
            (cf. `DocumentFile.from_pdf(..., lazy=True)`) and iterators, are processed chunk by chunk so that the
            memory footprint does not grow with the number of pages.
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
        page_chunk_size: int = 32,
        **kwargs: Any,
    ) -> None:
        if page_chunk_size < 1:
            raise ValueError("`page_chunk_size` must be a positive integer")
        self.page_chunk_size = page_chunk_size
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

    # Type of the documents returned by the predictor
    _document_cls: type[Document] = Document

    def _predict_pages(self, pages: list[np.ndarray], **kwargs: Any) -> Document:
        """Run the full pipeline on a list of pages held in memory"""
        raise NotImplementedError

    def _run(self, pages: Sequence[np.ndarray] | Iterable[np.ndarray], **kwargs: Any) -> Document:
        """Predict on any input: short lists are processed at once, everything else chunk by chunk"""
        if isinstance(pages, (list, tuple)) and len(pages) <= self.page_chunk_size:
            return self._predict_pages(list(pages), **kwargs)
        return self._document_cls(pages=list(self.stream(pages, **kwargs)))  # type: ignore[arg-type]

    def stream(
        self,
        pages: Sequence[np.ndarray] | Iterable[np.ndarray],
        page_chunk_size: int | None = None,
        prefetch_pages: bool = True,
        **kwargs: Any,
    ) -> Iterator[Page | KIEPage]:
        """Predict on a document of any length, yielding the pages one by one as soon as they are processed

        The pages are processed by chunks of `page_chunk_size`, and the next chunk is decoded in a background thread
        while the current one goes through the models. Only the pages of the chunks in flight are held in memory,
        so that a 1000+ page PDF or a folder of 100k images can be processed with a constant memory footprint.

        >>> from doctr.io import DocumentFile
        >>> from doctr.models import ocr_predictor
        >>> model = ocr_predictor(pretrained=True)
        >>> doc = DocumentFile.from_pdf("path/to/your/long_doc.pdf", lazy=True)
        >>> for page in model.stream(doc):
        ...     print(page.page_idx, page.render())

        Args:
            pages: a list of pages, a lazy page source (cf. `DocumentFile.from_pdf(..., lazy=True)`) or any
                iterable of pages (e.g. a generator)
            page_chunk_size: the number of pages processed at once (defaults to the `page_chunk_size` of the predictor)
            prefetch_pages: whether the next chunk of pages is decoded in a background thread
            **kwargs: keyword arguments forwarded to the sub-predictors (as for a regular call)

        Returns:
            an iterator over the predicted pages, in the input order. `page_idx` is the index of the page in the
            whole input. For lazy page sources, the predicted pages hold a reference to their image instead of the
            image itself: `page.page` decodes it again on access.
        """
        chunk_size = self.page_chunk_size if page_chunk_size is None else page_chunk_size
        source = pages if isinstance(pages, PageSource) else None
        offset = 0
        for chunk in prefetch(chunked(pages, chunk_size), enabled=prefetch_pages):
            result = self._predict_pages(chunk, **kwargs)
            for local_idx, (page, image) in enumerate(zip(result.pages, chunk)):
                page.page_idx = offset + local_idx
                if source is not None:
                    # Unchanged pages are decoded again from the source on access, transformed ones (e.g.
                    # straightened) are kept losslessly compressed
                    page.page = _SourcePage(source, page.page_idx) if page.page is image else _EncodedImage(page.page)
                yield page
            offset += len(chunk)

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
        objectness_scores: list[np.ndarray],
        assume_straight_pages: bool = False,
        assume_horizontal: bool = False,
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[np.ndarray]]:
        crops = _OCRPredictor._generate_crops(pages, loc_preds, assume_straight_pages, assume_horizontal)

        # Avoid sending zero-sized crops
        is_kept = [[all(s > 0 for s in crop.shape) for crop in page_crops] for page_crops in crops]
        crops = [
            [crop for crop, _kept in zip(page_crops, page_kept) if _kept]
            for page_crops, page_kept in zip(crops, is_kept)
        ]
        loc_preds = [_boxes[_kept] for _boxes, _kept in zip(loc_preds, is_kept)]
        # Drop the scores of the discarded boxes too, so that each kept box keeps its own score
        objectness_scores = [_scores[_kept] for _scores, _kept in zip(objectness_scores, is_kept)]

        return crops, loc_preds, objectness_scores

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

        Hooks run after text detection and before the word crops are extracted, in the order they were added.
        Each hook receives the location predictions, one array of boxes per page in relative coordinates
        (`(N, 4)` straight boxes or `(N, 4, 2)` polygons), and returns them in the same format. It can move,
        resize, add or remove boxes: only the returned boxes are cropped and read by the recognition model.

        Args:
            hook: a callable that takes as input the `loc_preds` and returns the modified `loc_preds`
        """
        self.hooks.append(hook)

    def _apply_hooks(
        self, loc_preds: list[np.ndarray], objectness_scores: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run the hooks on the location predictions, before the word crops are extracted

        Args:
            loc_preds: one array of boxes per page, in relative coordinates
            objectness_scores: one array of objectness scores per page

        Returns:
            the location predictions returned by the hooks, and the matching objectness scores
        """
        if not self.hooks:
            return loc_preds, objectness_scores
        # Pass a copy of the list, so that a hook replacing the arrays in place does not alter the reference
        hooked_preds: Any = list(loc_preds)
        for hook in self.hooks:
            hooked_preds = hook(hooked_preds)
        return self._align_hook_output(loc_preds, hooked_preds, objectness_scores)

    @staticmethod
    def _align_hook_output(
        loc_preds: list[np.ndarray], hooked_preds: Any, objectness_scores: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Check the location predictions returned by the hooks, and give each box an objectness score

        A box keeps the score of the closest predicted box of its page, so that moved, resized, reordered or
        filtered boxes keep their own score. A box added on a page without any prediction gets a score of 1.

        Args:
            loc_preds: the location predictions passed to the hooks, one array per page
            hooked_preds: the location predictions returned by the hooks
            objectness_scores: the objectness scores of `loc_preds`, one array per page

        Returns:
            the location predictions returned by the hooks as arrays, and their objectness scores
        """
        if not isinstance(hooked_preds, (list, tuple)) or len(hooked_preds) != len(loc_preds):
            received = type(hooked_preds).__name__
            if isinstance(hooked_preds, (list, tuple)):
                received += f" of length {len(hooked_preds)}"
            raise ValueError(
                f"hooks must return one array of boxes per page: expected a list of {len(loc_preds)} arrays, "
                f"got a {received}"
            )

        out_preds: list[np.ndarray] = []
        out_scores: list[np.ndarray] = []
        for page_idx, (boxes, new_boxes, scores) in enumerate(zip(loc_preds, hooked_preds, objectness_scores)):
            new_boxes = np.asarray(new_boxes, dtype=boxes.dtype)
            if new_boxes.size == 0:
                new_boxes = new_boxes.reshape((0, *boxes.shape[1:]))
            if new_boxes.shape[1:] != boxes.shape[1:]:
                raise ValueError(
                    "hooks must keep the format of the boxes: expected an array of shape "
                    f"(N, {', '.join(map(str, boxes.shape[1:]))}) for page {page_idx}, got {new_boxes.shape}"
                )

            if new_boxes.shape == boxes.shape and np.array_equal(new_boxes, boxes):
                new_scores = scores
            elif len(new_boxes) == 0:
                new_scores = scores[:0]
            elif len(boxes) == 0:
                new_scores = np.ones(len(new_boxes), dtype=scores.dtype)
            else:
                # Match each returned box with the closest predicted box (L1 distance between the corners),
                # by chunks to bound the memory used on dense pages
                ref = boxes.reshape(len(boxes), -1)
                flat = new_boxes.reshape(len(new_boxes), -1)
                idcs = np.concatenate([
                    np.abs(flat[start : start + 256, None, :] - ref[None, :, :]).sum(-1).argmin(axis=1)
                    for start in range(0, len(flat), 256)
                ])
                new_scores = scores[idcs]

            out_preds.append(new_boxes)
            out_scores.append(new_scores)

        return out_preds, out_scores
