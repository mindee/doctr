# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from typing import Any, Dict, List, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

from .datasets import VisionDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["SVHN"]


class SVHN(VisionDataset):
    """SVHN dataset from `"The Street View House Numbers (SVHN) Dataset"
    <http://ufldl.stanford.edu/housenumbers/>`_.

    .. image:: https://doctr-static.mindee.com/models?id=v0.5.0/svhn-grid.png&src=0
        :align: center

    >>> from doctr.datasets import SVHN
    >>> train_set = SVHN(train=True, download=True)
    >>> img, target = train_set[0]

    Args:
    ----
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        recognition_task: whether the dataset should be used for recognition task
        **kwargs: keyword arguments from `VisionDataset`.
    """

    TRAIN = (
        "http://ufldl.stanford.edu/housenumbers/train.tar.gz",
        "4b17bb33b6cd8f963493168f80143da956f28ec406cc12f8e5745a9f91a51898",
        "svhn_train.tar",
    )

    TEST = (
        "http://ufldl.stanford.edu/housenumbers/test.tar.gz",
        "57ac9ceb530e4aa85b55d991be8fc49c695b3d71c6f6a88afea86549efde7fb5",
        "svhn_test.tar",
    )

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        recognition_task: bool = False,
        **kwargs: Any,
    ) -> None:
        url, sha256, name = self.TRAIN if train else self.TEST
        super().__init__(
            url,
            file_name=name,
            file_hash=sha256,
            extract_archive=True,
            pre_transforms=convert_target_to_relative if not recognition_task else None,
            **kwargs,
        )
        self.train = train
        self.data: List[Tuple[Union[str, np.ndarray], Union[str, Dict[str, Any]]]] = []
        np_dtype = np.float32

        tmp_root = os.path.join(self.root, "train" if train else "test")

        # Load mat data (matlab v7.3 - can not be loaded with scipy)
        with h5py.File(os.path.join(tmp_root, "digitStruct.mat"), "r") as f:
            img_refs = f["digitStruct/name"]
            box_refs = f["digitStruct/bbox"]
            for img_ref, box_ref in tqdm(iterable=zip(img_refs, box_refs), desc="Unpacking SVHN", total=len(img_refs)):
                # convert ascii matrix to string
                img_name = "".join(map(chr, f[img_ref[0]][()].flatten()))

                # File existence check
                if not os.path.exists(os.path.join(tmp_root, img_name)):
                    raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_name)}")

                # Unpack the information
                box = f[box_ref[0]]
                if box["left"].shape[0] == 1:
                    box_dict = {k: [int(vals[0][0])] for k, vals in box.items()}
                else:
                    box_dict = {k: [int(f[v[0]][()].item()) for v in vals] for k, vals in box.items()}

                # Convert it to the right format
                coords: np.ndarray = np.array(
                    [box_dict["left"], box_dict["top"], box_dict["width"], box_dict["height"]], dtype=np_dtype
                ).transpose()
                label_targets = list(map(str, box_dict["label"]))

                if use_polygons:
                    # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                    box_targets: np.ndarray = np.stack(
                        [
                            np.stack([coords[:, 0], coords[:, 1]], axis=-1),
                            np.stack([coords[:, 0] + coords[:, 2], coords[:, 1]], axis=-1),
                            np.stack([coords[:, 0] + coords[:, 2], coords[:, 1] + coords[:, 3]], axis=-1),
                            np.stack([coords[:, 0], coords[:, 1] + coords[:, 3]], axis=-1),
                        ],
                        axis=1,
                    )
                else:
                    # x, y, width, height -> xmin, ymin, xmax, ymax
                    box_targets = np.stack(
                        [
                            coords[:, 0],
                            coords[:, 1],
                            coords[:, 0] + coords[:, 2],
                            coords[:, 1] + coords[:, 3],
                        ],
                        axis=-1,
                    )

                if recognition_task:
                    crops = crop_bboxes_from_image(img_path=os.path.join(tmp_root, img_name), geoms=box_targets)
                    for crop, label in zip(crops, label_targets):
                        if crop.shape[0] > 0 and crop.shape[1] > 0 and len(label) > 0:
                            self.data.append((crop, label))
                else:
                    self.data.append((img_name, dict(boxes=box_targets, labels=label_targets)))

        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
