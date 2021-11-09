# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .datasets import VisionDataset

__all__ = ['SVT']


class SVT(VisionDataset):
    """SVT dataset from `"The Street View Text Dataset - UCSD Computer Vision"
    <http://vision.ucsd.edu/~kai/svt/>`_.

    Example::
        >>> from doctr.datasets import SVT
        >>> train_set = SVT(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'http://vision.ucsd.edu/~kai/svt/svt.zip'
    SHA256 = '63b3d55e6b6d1e036e2a844a20c034fe3af3c32e4d914d6e0c4a3cd43df3bebf'

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(self.URL, None, self.SHA256, True, **kwargs)
        self.sample_transforms = sample_transforms
        self.train = train
        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        np_dtype = np.float16 if self.fp16 else np.float32

        # Load xml data
        tmp_root = os.path.join(self.root, 'svt1')
        xml_tree = ET.parse(os.path.join(tmp_root, 'train.xml')) if self.train else ET.parse(
            os.path.join(tmp_root, 'test.xml'))
        xml_root = xml_tree.getroot()

        for child in xml_root:
            _tmp_box_targets = list()
            _tmp_labels = list()
            for image_tag in child:
                if image_tag.tag == 'imageName':
                    _raw_path = str(image_tag.text)
                    # File existence check
                    if not os.path.exists(os.path.join(tmp_root, _raw_path)):
                        raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, _raw_path)}")

                for rect_tag in image_tag:
                    xmin, ymin, xmax, ymax = (int(rect_tag.attrib['x']), int(rect_tag.attrib['y']),
                                              int(rect_tag.attrib['x']) + int(rect_tag.attrib['width']),
                                              int(rect_tag.attrib['y']) + int(rect_tag.attrib['height']))
                    if rotated_bbox:
                        # box_targets: xmin, ymin, xmax, ymax -> x, y, w, h, alpha = 0
                        x, y, w, h, alpha = ((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin, 0)
                        _tmp_box_targets.append([x, y, w, h, alpha])
                    else:
                        _tmp_box_targets.append([xmin, ymin, xmax, ymax])
                    for label in rect_tag:
                        _tmp_labels.append(label.text)

            if len(_tmp_labels) != len(_tmp_box_targets):
                raise ValueError(f"{_tmp_labels} and {_tmp_box_targets} are not same length")

            self.data.append((Path(_raw_path), dict(boxes=np.asarray(
                _tmp_box_targets, dtype=np_dtype), labels=_tmp_labels)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
