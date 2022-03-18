# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Dict, List, Tuple

import defusedxml.ElementTree as ET
import numpy as np

from .datasets import VisionDataset

__all__ = ['SVT']


class SVT(VisionDataset):
    """SVT dataset from `"The Street View Text Dataset - UCSD Computer Vision"
    <http://vision.ucsd.edu/~kai/svt/>`_.

    .. image:: https://github.com/mindee/doctr/releases/download/v0.5.0/svt-grid.png
        :align: center

    >>> from doctr.datasets import SVT
    >>> train_set = SVT(train=True, download=True)
    >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'http://vision.ucsd.edu/~kai/svt/svt.zip'
    SHA256 = '63b3d55e6b6d1e036e2a844a20c034fe3af3c32e4d914d6e0c4a3cd43df3bebf'

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(self.URL, None, self.SHA256, True, **kwargs)
        self.train = train
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float32

        # Load xml data
        tmp_root = os.path.join(self.root, 'svt1') if self.SHA256 else self.root
        xml_tree = ET.parse(os.path.join(tmp_root, 'train.xml')) if self.train else ET.parse(
            os.path.join(tmp_root, 'test.xml'))
        xml_root = xml_tree.getroot()

        for image in xml_root:
            name, _, _, resolution, rectangles = image

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, name.text)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, name.text)}")

            if use_polygons:
                # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                _boxes = [
                    [
                        [float(rect.attrib['x']), float(rect.attrib['y'])],
                        [float(rect.attrib['x']) + float(rect.attrib['width']), float(rect.attrib['y'])],
                        [
                            float(rect.attrib['x']) + float(rect.attrib['width']),
                            float(rect.attrib['y']) + float(rect.attrib['height'])
                        ],
                        [float(rect.attrib['x']), float(rect.attrib['y']) + float(rect.attrib['height'])],
                    ]
                    for rect in rectangles
                ]
            else:
                # x_min, y_min, x_max, y_max
                _boxes = [
                    [float(rect.attrib['x']), float(rect.attrib['y']),  # type: ignore[list-item]
                     float(rect.attrib['x']) + float(rect.attrib['width']),  # type: ignore[list-item]
                     float(rect.attrib['y']) + float(rect.attrib['height'])]  # type: ignore[list-item]
                    for rect in rectangles
                ]
            # Convert them to relative
            w, h = int(resolution.attrib['x']), int(resolution.attrib['y'])
            boxes = np.asarray(_boxes, dtype=np_dtype)
            if use_polygons:
                boxes[:, :, 0] /= w
                boxes[:, :, 1] /= h
            else:
                boxes[:, [0, 2]] /= w
                boxes[:, [1, 3]] /= h

            # Get the labels
            labels = [lab.text for rect in rectangles for lab in rect]

            self.data.append((name.text, dict(boxes=boxes, labels=labels)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
