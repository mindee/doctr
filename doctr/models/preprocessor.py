# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import cv2
import json
import os
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict


__all__ = ['Preprocessor']


class Preprocessor:
    """
    class to preprocess documents
    a processor can perform noramization, resizing and batching
    a processor is called on a document
    """

    def __init__(
        self,
        out_size: Tuple[int, int],
        normalization: bool = True,
        mode: str = 'symmetric',
        batch_size: int = 1
    ) -> None:

        self.out_size = out_size
        self.normalization = normalization
        self.mode = mode
        self.batch_size = batch_size

    def normalize_documents_imgs(
        self,
        documents_imgs: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """
        normalize documents imgs according to mode
        """

        if self.mode == 'symmetric':
            normalized = [[(img - 128) / 128 for img in doc] for doc in documents_imgs]
        else:
            normalized = documents_imgs
        return normalized

    def resize_documents_imgs(
        self,
        documents_imgs: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """
        Resize documents img to the out_size : size for the model inputs
        The nested structure documents/pages is preserved
        returns resized documents img
        """
        return [[cv2.resize(img, self.out_size, cv2.INTER_LINEAR) for img in doc] for doc in documents_imgs]

    def batch_documents(
        self,
        documents: Tuple[List[List[np.ndarray]], List[List[str]], List[List[Tuple[int, int]]]]
    ) -> Tuple[List[Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]], List[int], List[int]]:
        """
        function to batch a list of read documents
        :param documents: documents read by documents.reader.read_documents
        :param batch_size: batch_size to use during inference, default goes to 1
        """

        images, names, shapes = documents

        # keep track of both documents and pages indexes
        docs_indexes = [i for i, doc in enumerate(images) for _ in doc]
        pages_indexes = [i for doc in images for i, page in enumerate(doc)]

        # flatten structure
        flat_images = [image for doc in images for image in doc]
        flat_names = [name for doc in names for name in doc]
        flat_shapes = [shape for doc in shapes for shape in doc]

        range_batch = range((len(flat_shapes) + self.batch_size - 1) // self.batch_size)

        b_images = [flat_images[i * self.batch_size:(i + 1) * self.batch_size] for i in range_batch]
        b_names = [flat_names[i * self.batch_size:(i + 1) * self.batch_size] for i in range_batch]
        b_shapes = [flat_shapes[i * self.batch_size:(i + 1) * self.batch_size] for i in range_batch]

        b_docs = [(b_i, b_n, b_s) for b_i, b_n, b_s in zip(b_images, b_names, b_shapes)]

        return b_docs, docs_indexes, pages_indexes

    def __call__(
        self,
        documents: Tuple[List[List[np.ndarray]], List[List[str]], List[List[Tuple[int, int]]]]
    ) -> Tuple[List[Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]], List[int], List[int]]:
        """
        perform resizing, normalization and batching on documents
        """
        images, names, shapes = documents
        images = self.resize_documents_imgs(images)
        if self.normalization:
            images = self.normalize_documents_imgs(images)
        norm_and_sized_docs = images, names, shapes
        b_docs, docs_indexes, pages_indexes = self.batch_documents(norm_and_sized_docs)

        return b_docs, docs_indexes, pages_indexes
