# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict
from .preprocessor import PreProcessor
from .detection import DetectionPredictor
from .recognition import RecognitionPredictor
from ._utils import extract_crops

__all__ = ['OCRPredictor']


class OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
    ) -> None:

        self.det_predictor = det_predictor
        self.reco_predictor = reco_predictor

    def __call__(
        self,
        documents: List[List[np.ndarray]],
    ) -> List[List[List[Dict[str, Any]]]]:

        pages = [page for doc in documents for page in doc]

        # Localize text elements
        boxes = self.det_predictor(pages)
        # Crop images
        crops = [crop for page, _boxes in zip(pages, boxes) for crop in extract_crops(page, _boxes)]
        # Identify character sequences
        char_sequences = self.reco_predictor(crops)

        # Reorganize
        num_pages = [len(doc) for doc in documents]
        num_crops = [_boxes.shape[0] for _boxes in boxes]
        page_idx, crop_idx = 0, 0
        results = []
        for doc_idx, nb_pages in enumerate(num_pages):
            doc = []
            for page_boxes in boxes[page_idx: page_idx + nb_pages]:
                page = []
                for _idx in range(num_crops[page_idx]):
                    page.append(dict(box=page_boxes[_idx], text=char_sequences[crop_idx + _idx]))

                crop_idx += num_crops[page_idx]
                page_idx += 1
                doc.append(page)
            results.append(doc)

        return results
