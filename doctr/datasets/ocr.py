# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import tensorflow as tf

from .core import AbstractDataset


__all__ = ['OCRDataset']


class OCRDataset(AbstractDataset):
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
        sample_transforms: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        **kwargs: Any,
    ) -> None:

        self.sample_transforms = sample_transforms
        self.root = img_folder

        # List images
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        with open(label_file, 'rb') as f:
            data = json.load(f)

        for file_dic in data:
            # Get image path
            img_name = Path(os.path.basename(file_dic["raw-archive-filepath"])).stem + '.jpg'
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")

            # handle empty images
            if (len(file_dic["coordinates"]) == 0 or
               (len(file_dic["coordinates"]) == 1 and file_dic["coordinates"][0] == "N/A")):
                self.data.append((img_name, dict(boxes=np.zeros((0, 4), dtype=np.float32), labels=[])))
                continue
            is_valid: List[bool] = []
            box_targets: List[List[float]] = []
            for box in file_dic["coordinates"]:
                xs, ys = zip(*box)
                box = [min(xs), min(ys), max(xs), max(ys)]
                is_valid.append(box[0] < box[2] and box[1] < box[3])
                if is_valid[-1]:
                    box_targets.append(box)

            text_targets = [word for word, _valid in zip(file_dic["string"], is_valid) if _valid]
            self.data.append((img_name, dict(boxes=np.asarray(box_targets, dtype=np.float32), labels=text_targets)))
