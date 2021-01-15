# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import cv2
import json
import os
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict

class Preprocessor():

    def __init__(self, out_size: Tuple[int, int],
        normalization: bool = True,
        batch_size: int = 1):
        
        self.out_size = out_size
        self.normalization = normalization
        self.batch_size = batch_size

    def __call__(
        documents: Tuple[List[List[np.ndarray]], List[List[str]], List[List[Tuple[int, int]]]]
    ) -> List[Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]]:
        """
        perform resizing, normalization and batching on documents
        """
        images, names, shapes = documents
        images = resize_documents_imgs(images, out_size=self.out_size)
        if self.normalization:
            images = normalize_documents_imgs(images)
        norm_and_sized_docs = images, names, shapes
        b_docs, docs_indexes, pages_indexes = batch_documents(
            norm_and_sized_docs, batch_size=self.batch_size)

        return b_docs, docs_indexes, pages_indexes

    
    def normalize_documents_imgs(
        documents_imgs: List[List[np.ndarray]],
        mode: str = 'symmetric'
    ) -> List[List[np.ndarray]]:
        """
        normalize documents imgs according to mode
        """
        
        if mode == 'symmetric':
            return [[(img - 128) / 128 for img in doc] for doc in documents_imgs]


    def resize_documents_imgs(
        documents_imgs: List[List[np.ndarray]],
        out_size: Tuple[int, int]
    ) -> List[List[np.ndarray]]:
        """
        Resize documents img to the out_size : size for the model inputs
        The nested structure documents/pages is preserved
        returns resized documents img
        """
        return [[cv2.resize(img, out_size, cv2.INTER_LINEAR) for img in doc]
            for doc in documents_imgs]


    def batch_documents(
        documents: Tuple[List[List[np.ndarray]], List[List[str]], List[List[Tuple[int, int]]]],
        batch_size: int = 1
    ) -> List[Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]]:
        """
        function to batch a list of read documents
        :param documents: documents read by documents.reader.read_documents
        :param batch_size: batch_size to use during inference, default goes to 1
        """

        images, names, shapes = documents

        # keep track of both documents and pages indexes
        docs_indexes = [images.index(doc) for doc in images for _ in doc]
        pages_indexes = [doc.index(page) for doc in images for page in doc]

        # flatten structure
        flat_images = [image for doc in images for raw in doc]
        flat_names = [name for doc in names for name in doc]
        flat_shapes = [shape for doc in shapes for shape in doc]

        range_batch = range((len(flat_shapes) + batch_size - 1) // batch_size)

        b_images = [flat_images[i * batch_size:(i + 1) * batch_size] for i in range_batch]
        b_names = [flat_names[i * batch_size:(i + 1) * batch_size] for i in range_batch]
        b_shapes = [flat_shapes[i * batch_size:(i + 1) * batch_size] for i in range_batch]

        b_docs = [[b_i, b_n, b_s] for b_i, b_n, b_s in zip(b_images, b_names, b_shapes)]

        return b_docs, docs_indexes, pages_indexes
