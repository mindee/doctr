# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .datasets import AbstractDataset

__all__ = ["OCRDataset"]


class OCRDataset(AbstractDataset):
    """Implements an OCR dataset

    >>> from doctr.datasets import OCRDataset
    >>> train_set = OCRDataset(img_folder="/path/to/images",
    >>>                        label_file="/path/to/labels.json")
    >>> img, target = train_set[0]

    Args:
    ----
        img_folder: local path to image folder (all jpg at the root)
        label_file: local path to the label file
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_file: str,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)

        # List images
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float32
        with open(label_file, "rb") as f:
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
            # Unpack the straight boxes (xmin, ymin, xmax, ymax)
            geoms = [list(map(float, obj["geometry"][:4])) for obj in annotations["typed_words"]]
            if use_polygons:
                # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                geoms = [
                    [geom[:2], [geom[2], geom[1]], geom[2:], [geom[0], geom[3]]]  # type: ignore[list-item]
                    for geom in geoms
                ]

            text_targets = [obj["value"] for obj in annotations["typed_words"]]

            self.data.append((img_name, dict(boxes=np.asarray(geoms, dtype=np_dtype), labels=text_targets)))
