# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

from .datasets import AbstractDataset
from doctr.utils.geometry import fit_rbbox


__all__ = ['OCRDataset', 'DocDataset']


class OCRDataset(AbstractDataset):
    """Implements an OCR dataset

    Args:
        img_folder: local path to image folder (all jpg at the root)
        label_file: local path to the label file
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_file: str,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)
        self.sample_transforms = sample_transforms

        # List images
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float16 if self.fp16 else np.float32
        with open(label_file, 'rb') as f:
            data = json.load(f)

        for file_dic in data:
            # Get image path
            img_name = Path(os.path.basename(file_dic["raw-archive-filepath"])).stem + '.jpg'
            # File existence check
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")

            # handle empty images
            if (len(file_dic["coordinates"]) == 0 or
               (len(file_dic["coordinates"]) == 1 and file_dic["coordinates"][0] == "N/A")):
                self.data.append((img_name, dict(boxes=np.zeros((0, 4), dtype=np_dtype), labels=[])))
                continue
            is_valid: List[bool] = []
            box_targets: List[List[float]] = []
            for box in file_dic["coordinates"]:
                if rotated_bbox:
                    x, y, w, h, alpha = fit_rbbox(np.asarray(box, dtype=np_dtype))
                    box = [x, y, w, h, alpha]
                    is_valid.append(w > 0 and h > 0)
                else:
                    xs, ys = zip(*box)
                    box = [min(xs), min(ys), max(xs), max(ys)]
                    is_valid.append(box[0] < box[2] and box[1] < box[3])
                if is_valid[-1]:
                    box_targets.append(box)

            text_targets = [word for word, _valid in zip(file_dic["string"], is_valid) if _valid]
            self.data.append((img_name, dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=text_targets)))


class DocDataset(AbstractDataset):
    """Implements an OCR dataset

    Args:
        img_folder: local path to image folder (all jpg at the root)
        label_file: local path to the label file
        sample_transforms: composable transformations that will be applied to each image
        **kwargs: keyword arguments from `VisionDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_file: str,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)
        self.sample_transforms = sample_transforms

        # List images
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float16 if self.fp16 else np.float32
        with open(label_file, 'rb') as f:
            data = json.load(f)

        for img_name, annotations in data.items():
            # Get image path
            img_name = Path(img_name)
            # File existence check
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")

            # handle empty images
            if len(annotations["typed_words"]) == 0:
                self.data.append((img_name, dict(boxes=np.zeros((0, 4), dtype=np_dtype), labels=[])))
                continue
            # Unpack
            box_targets = [tuple(map(float, obj['geometry'])) for obj in annotations['typed_words']]
            text_targets = [obj['value'] for obj in annotations['typed_words']]

            self.data.append((img_name, dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=text_targets)))
