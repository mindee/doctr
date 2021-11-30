# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import defusedxml.ElementTree as ET
import numpy as np

from .datasets import VisionDataset

__all__ = ['IC03']


class IC03(VisionDataset):
    """IC03 dataset from `"ICDAR 2003 Robust Reading Competitions: Entries, Results and Future Directions"
    <http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions>`_.

    Example::
        >>> from doctr.datasets import IC03
        >>> train_set = IC03(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    TRAIN = ('http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/scene.zip',
             '9d86df514eb09dd693fb0b8c671ef54a0cfe02e803b1bbef9fc676061502eb94',
             'ic03_train.zip')
    TEST = ('http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTest/scene.zip',
            'dbc4b5fd5d04616b8464a1b42ea22db351ee22c2546dd15ac35611857ea111f8',
            'ic03_test.zip')

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        url, sha256, file_name = self.TRAIN if train else self.TEST
        super().__init__(url, file_name, sha256, True, **kwargs)
        self.sample_transforms = sample_transforms
        self.train = train
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float32

        # Load xml data
        tmp_root = os.path.join(self.root, 'SceneTrialTrain' if self.train else 'SceneTrialTest')
        xml_tree = ET.parse(os.path.join(tmp_root, 'words.xml'))
        xml_root = xml_tree.getroot()

        for image in xml_root:
            name, resolution, rectangles = image

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, name.text)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, name.text)}")

            if rotated_bbox:
                # x_center, y_center, width, height, 0
                _boxes = [
                    [float(rect.attrib['x']) + float(rect.attrib['width']) / 2,
                     float(rect.attrib['y']) + float(rect.attrib['height']) / 2,
                     float(rect.attrib['width']), float(rect.attrib['height']), float(rect.attrib['rotation'])]
                    for rect in rectangles
                ]
            else:
                # x_min, y_min, x_max, y_max
                _boxes = [
                    [float(rect.attrib['x']), float(rect.attrib['y']),
                     float(rect.attrib['x']) + float(rect.attrib['width']),
                     float(rect.attrib['y']) + float(rect.attrib['height'])]
                    for rect in rectangles
                ]

            # filter images without boxes
            if len(_boxes) > 0:
                # Convert them to relative
                w, h = int(resolution.attrib['x']), int(resolution.attrib['y'])
                boxes = np.asarray(_boxes, dtype=np_dtype)
                boxes[:, [0, 2]] /= w
                boxes[:, [1, 3]] /= h

                # Get the labels
                labels = [lab.text for rect in rectangles for lab in rect if lab.text]

                self.data.append((name.text, dict(boxes=boxes, labels=labels)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
