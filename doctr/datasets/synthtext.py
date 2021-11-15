# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .datasets import VisionDataset

__all__ = ['SynthText']


class SynthText(VisionDataset):
    """SynthText dataset from `"Synthetic Data for Text Localisation in Natural Images"
    <https://arxiv.org/abs/1604.06646>`_.

    Example::
        >>> # NOTE: This dataset has currently no train/test split
        >>> from doctr.datasets import SynthText
        >>> data_set = SynthText(download=True)
        >>> img, target = data_set[0]

    Args:
        sample_transforms: composable transformations that will be applied to each image
        rotated_bbox: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip'
    SHA256 = '28ab030485ec8df3ed612c568dd71fb2793b9afbfa3a9d9c6e792aef33265bf1'

    def __init__(
        self,
        train: bool = True,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        rotated_bbox: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(url=self.URL, file_name=None, file_hash=self.SHA256, extract_archive=True, **kwargs)
        self.sample_transforms = sample_transforms

        # Load mat data
        tmp_root = os.path.join(self.root, 'SynthText')
        mat_data = sio.loadmat(os.path.join(tmp_root, 'gt.mat'))

        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        np_dtype = np.float16 if self.fp16 else np.float32

        for img_path, word_boxes, txt in tqdm(iterable=zip(
                mat_data['imnames'][0],
                mat_data['wordBB'][0],
                mat_data['txt'][0]
        ), desc='Load SynthText', total=len(mat_data['imnames'][0])):

            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path[0])):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path[0])}")

            labels = _text_to_words(txt)
            word_boxes = word_boxes.transpose(2, 1, 0) if word_boxes.ndim == 3 else np.expand_dims(word_boxes, axis=0)

            if rotated_bbox:
                # x_center, y_center, w, h, alpha = 0
                box_targets = [_compute_rotated_box(pts) for pts in word_boxes]
            else:
                # xmin, ymin, xmax, ymax
                box_targets = [_compute_straight_box(pts) for pts in word_boxes]  # type: ignore[misc]

            self.data.append((img_path[0], dict(boxes=np.asarray(box_targets, dtype=np_dtype), labels=labels)))

        self.root = tmp_root


def _text_to_words(txt: np.ndarray) -> List[str]:
    """Convert np.str-Array to list of str."""
    line = '\n'.join(txt)
    return line.split()


def _compute_straight_box(pts: np.ndarray) -> Tuple[float, float, float, float]:
    # pts: Nx2
    xmin = np.min(pts[:, 0])
    xmax = np.max(pts[:, 0])
    ymin = np.min(pts[:, 1])
    ymax = np.max(pts[:, 1])
    return xmin, ymin, xmax, ymax


def _compute_rotated_box(pts: np.ndarray) -> Tuple[float, float, float, float, int]:
    # pts: Nx2
    x = np.min(pts[:, 0])
    y = np.min(pts[:, 1])
    width = np.max(pts[:, 0]) - x
    height = np.max(pts[:, 1]) - y
    # x_center, y_center, w, h, alpha = 0
    return x + width / 2, y + height / 2, width, height, 0
