# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Tuple

import numpy as np
import math

from numpy.lib.financial import nper

from doctr.models.builder import DocumentBuilder
from .._utils import extract_crops, extract_rcrops
from ..classification import mobilenet_v3_small

__all__ = ['_OCRPredictor']


class _OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    doc_builder: DocumentBuilder
    def __init__(self) -> None:
        self.classifier = mobilenet_v3_small(pretrained=True)
    
    def _rectify_crops(
        self,
        crops: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:

        # Batch
        batch_size = 64
        # flatten crops
        num_batches = int(math.ceil(len(samples) / batch_size))
        batches = [
            np.stack(samples[idx * self.batch_size: min((idx + 1) * self.batch_size, len(samples))], axis=0)
            for idx in range(int(num_batches))
        ]

        batches = batch(padded)

        # Pad
        padded = []
        target_size = 128
        for batch in batches:
            batch = []
            for crop in batch:
                h, w, _ = crop.shape
                if h > w:
                    resize_target = [target_size, round(target_size * w / h), 3]
                    pad = round(target_size * (1 - w / h) / 2)
                    padding_target = [[0, 0], [pad, target_size - pad - round(target_size * w / h)], [0, 0]]
                else:
                    resize_target = [round(target_size * h / w), target_size, 3]
                    pad = round(target_size * (1 - h / w) / 2)
                    padding_target = [[pad, target_size - pad - round(target_size * h / w)], [0, 0], [0, 0]]
                crop = np.resize(crop, resize_target)
                crop = np.pad(crop, padding_target)
                batch.append(crop)
            padded.append(batch)

        # Send to classifier
        rectified_batches = []
        for i, padded_batch in padded:
            directions = self.classifier(padded_batch)
            rect_batch = [np.rot90(batch[i][j], direction) for j, direction in enumerate(directions)]
            rectified_batches.append(rect_batch)

        #Rema batched to pages

        return crops

    @staticmethod
    def _generate_crops(
        pages: List[np.ndarray],
        loc_preds: List[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
    ) -> List[List[np.ndarray]]:

        extraction_fn = extract_crops if assume_straight_pages else extract_rcrops

        crops = [
            extraction_fn(page, _boxes[:, :-1], channels_last=channels_last)  # type: ignore[operator]
            for page, _boxes in zip(pages, loc_preds)
        ]
        return crops

    @staticmethod
    def _prepare_crops(
        pages: List[np.ndarray],
        loc_preds: List[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:

        crops = _OCRPredictor._generate_crops(pages, loc_preds, channels_last, assume_straight_pages)

        # Avoid sending zero-sized crops
        is_kept = [[all(s > 0 for s in crop.shape) for crop in page_crops] for page_crops in crops]
        crops = [
            [crop for crop, _kept in zip(page_crops, page_kept) if _kept]
            for page_crops, page_kept in zip(crops, is_kept)
        ]

        crops = _OCRPredictor._rectify_crops(crops)

        loc_preds = [_boxes[_kept] for _boxes, _kept in zip(loc_preds, is_kept)]

        return crops, loc_preds

    @staticmethod
    def _process_predictions(
        loc_preds: List[np.ndarray],
        word_preds: List[Tuple[str, float]],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[str, float]]]]:

        text_preds = []
        if len(loc_preds) > 0:
            # Text
            _idx = 0
            for page_boxes in loc_preds:
                text_preds.append(word_preds[_idx: _idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]

        return loc_preds, text_preds
