# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

from .datasets import VisionDataset
from doctr.utils.geometry import fit_rbbox

__all__ = ['CORD']


class CORD(VisionDataset):
    """CORD dataset from `"CORD: A Consolidated Receipt Dataset forPost-OCR Parsing"
    <https://openreview.net/pdf?id=SJl3z659UH>`_.

    Example::
        >>> from doctr.datasets import CORD
        >>> train_set = CORD(train=True, download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """
    TRAIN = ('https://github.com/mindee/doctr/releases/download/v0.1.1/cord_train.zip',
             '45f9dc77f126490f3e52d7cb4f70ef3c57e649ea86d19d862a2757c9c455d7f8')

    TEST = ('https://github.com/mindee/doctr/releases/download/v0.1.1/cord_test.zip',
            '8c895e3d6f7e1161c5b7245e3723ce15c04d84be89eaa6093949b75a66fb3c58')

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        url, sha256 = self.TRAIN if train else self.TEST
        super().__init__(url, None, sha256, True, **kwargs)

        # # List images
        tmp_root = os.path.join(self.root, 'image')
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        np_dtype = np.float16 if self.fp16 else np.float32
        self.train = train
        self.sample_transforms = sample_transforms
        for img_path in os.listdir(tmp_root):
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path)}")
            stem = Path(img_path).stem
            _targets = []
            with open(os.path.join(self.root, 'json', f"{stem}.json"), 'rb') as f:
                label = json.load(f)
                for line in label["valid_line"]:
                    for word in line["words"]:
                        if len(word["text"]) > 0:
                            x = word["quad"]["x1"], word["quad"]["x2"], word["quad"]["x3"], word["quad"]["x4"]
                            y = word["quad"]["y1"], word["quad"]["y2"], word["quad"]["y3"], word["quad"]["y4"]
                            if rotated_bbox:
                                box = list(fit_rbbox(np.array([
                                    [x[0], y[0]],
                                    [x[1], y[1]],
                                    [x[2], y[2]],
                                    [x[3], y[3]],
                                ], dtype=np_dtype)))
                            else:
                                # Reduce 8 coords to 4
                                box = [min(x), min(y), max(x), max(y)]
                            _targets.append((word['text'], box))

            text_targets, box_targets = zip(*_targets)

            self.data.append((
                img_path,
                dict(boxes=np.asarray(box_targets, dtype=int).clip(min=0), labels=text_targets)
            ))
        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
