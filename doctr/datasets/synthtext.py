# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import glob
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image
from scipy import io as sio
from tqdm import tqdm

from .datasets import VisionDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["SynthText"]


class SynthText(VisionDataset):
    """SynthText dataset from `"Synthetic Data for Text Localisation in Natural Images"
    <https://arxiv.org/abs/1604.06646>`_ | `"repository" <https://github.com/ankush-me/SynthText>`_ |
    `"website" <https://www.robots.ox.ac.uk/~vgg/data/scenetext/>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/svt-grid.png&src=0
        :align: center

    >>> from doctr.datasets import SynthText
    >>> train_set = SynthText(train=True, download=True)
    >>> img, target = train_set[0]

    Args:
    ----
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = "https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip"
    SHA256 = "28ab030485ec8df3ed612c568dd71fb2793b9afbfa3a9d9c6e792aef33265bf1"

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        recognition_task: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            self.URL,
            None,
            file_hash=None,
            extract_archive=True,
            pre_transforms=convert_target_to_relative if not recognition_task else None,
            **kwargs,
        )
        self.train = train
        self.data: List[Tuple[Union[str, np.ndarray], Union[str, Dict[str, Any]]]] = []
        np_dtype = np.float32

        # Load mat data
        tmp_root = os.path.join(self.root, "SynthText") if self.SHA256 else self.root
        # define folder to write SynthText recognition dataset
        reco_folder_name = "SynthText_recognition_train" if self.train else "SynthText_recognition_test"
        reco_folder_name = "Poly_" + reco_folder_name if use_polygons else reco_folder_name
        reco_folder_path = os.path.join(tmp_root, reco_folder_name)
        reco_images_counter = 0

        if recognition_task and os.path.isdir(reco_folder_path):
            self._read_from_folder(reco_folder_path)
            return
        elif recognition_task and not os.path.isdir(reco_folder_path):
            os.makedirs(reco_folder_path, exist_ok=False)

        mat_data = sio.loadmat(os.path.join(tmp_root, "gt.mat"))
        train_samples = int(len(mat_data["imnames"][0]) * 0.9)
        set_slice = slice(train_samples) if self.train else slice(train_samples, None)
        paths = mat_data["imnames"][0][set_slice]
        boxes = mat_data["wordBB"][0][set_slice]
        labels = mat_data["txt"][0][set_slice]
        del mat_data

        for img_path, word_boxes, txt in tqdm(
            iterable=zip(paths, boxes, labels), desc="Unpacking SynthText", total=len(paths)
        ):
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_path[0])):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_path[0])}")

            labels = [elt for word in txt.tolist() for elt in word.split()]
            # (x, y) coordinates of top left, top right, bottom right, bottom left corners
            word_boxes = (
                word_boxes.transpose(2, 1, 0)
                if word_boxes.ndim == 3
                else np.expand_dims(word_boxes.transpose(1, 0), axis=0)
            )

            if not use_polygons:
                # xmin, ymin, xmax, ymax
                word_boxes = np.concatenate((word_boxes.min(axis=1), word_boxes.max(axis=1)), axis=1)

            if recognition_task:
                crops = crop_bboxes_from_image(img_path=os.path.join(tmp_root, img_path[0]), geoms=word_boxes)
                for crop, label in zip(crops, labels):
                    if crop.shape[0] > 0 and crop.shape[1] > 0 and len(label) > 0:
                        # write data to disk
                        with open(os.path.join(reco_folder_path, f"{reco_images_counter}.txt"), "w") as f:
                            f.write(label)
                            tmp_img = Image.fromarray(crop)
                            tmp_img.save(os.path.join(reco_folder_path, f"{reco_images_counter}.png"))
                            reco_images_counter += 1
            else:
                self.data.append((img_path[0], dict(boxes=np.asarray(word_boxes, dtype=np_dtype), labels=labels)))

        if recognition_task:
            self._read_from_folder(reco_folder_path)

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"

    def _read_from_folder(self, path: str) -> None:
        for img_path in glob.glob(os.path.join(path, "*.png")):
            with open(os.path.join(path, f"{os.path.basename(img_path)[:-4]}.txt"), "r") as f:
                self.data.append((img_path, f.read()))
