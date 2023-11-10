# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from typing import Any, List, Tuple

from tqdm import tqdm

from .datasets import AbstractDataset

__all__ = ["MJSynth"]


class MJSynth(AbstractDataset):
    """MJSynth dataset from `"Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition"
    <https://www.robots.ox.ac.uk/~vgg/data/text/>`_.

    >>> # NOTE: This is a pure recognition dataset without bounding box labels.
    >>> # NOTE: You need to download the dataset.
    >>> from doctr.datasets import MJSynth
    >>> train_set = MJSynth(img_folder="/path/to/mjsynth/mnt/ramdisk/max/90kDICT32px",
    >>>                     label_path="/path/to/mjsynth/mnt/ramdisk/max/90kDICT32px/imlist.txt",
    >>>                     train=True)
    >>> img, target = train_set[0]
    >>> test_set = MJSynth(img_folder="/path/to/mjsynth/mnt/ramdisk/max/90kDICT32px",
    >>>                    label_path="/path/to/mjsynth/mnt/ramdisk/max/90kDICT32px/imlist.txt")
    >>>                    train=False)
    >>> img, target = test_set[0]

    Args:
    ----
        img_folder: folder with all the images of the dataset
        label_path: path to the file with the labels
        train: whether the subset should be the training one
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    # filter corrupted or missing images
    BLACKLIST = [
        "./1881/4/225_Marbling_46673.jpg\n",
        "./2069/4/192_whittier_86389.jpg\n",
        "./869/4/234_TRIASSIC_80582.jpg\n",
        "./173/2/358_BURROWING_10395.jpg\n",
        "./913/4/231_randoms_62372.jpg\n",
        "./596/2/372_Ump_81662.jpg\n",
        "./936/2/375_LOCALITIES_44992.jpg\n",
        "./2540/4/246_SQUAMOUS_73902.jpg\n",
        "./1332/4/224_TETHERED_78397.jpg\n",
        "./627/6/83_PATRIARCHATE_55931.jpg\n",
        "./2013/2/370_refract_63890.jpg\n",
        "./2911/6/77_heretical_35885.jpg\n",
        "./1730/2/361_HEREON_35880.jpg\n",
        "./2194/2/334_EFFLORESCENT_24742.jpg\n",
        "./2025/2/364_SNORTERS_72304.jpg\n",
        "./368/4/232_friar_30876.jpg\n",
        "./275/6/96_hackle_34465.jpg\n",
        "./384/4/220_bolts_8596.jpg\n",
        "./905/4/234_Postscripts_59142.jpg\n",
        "./2749/6/101_Chided_13155.jpg\n",
        "./495/6/81_MIDYEAR_48332.jpg\n",
        "./2852/6/60_TOILSOME_79481.jpg\n",
        "./554/2/366_Teleconferences_77948.jpg\n",
        "./1696/4/211_Queened_61779.jpg\n",
        "./2128/2/369_REDACTED_63458.jpg\n",
        "./2557/2/351_DOWN_23492.jpg\n",
        "./2489/4/221_snored_72290.jpg\n",
        "./1650/2/355_stony_74902.jpg\n",
        "./1863/4/223_Diligently_21672.jpg\n",
        "./264/2/362_FORETASTE_30276.jpg\n",
        "./429/4/208_Mainmasts_46140.jpg\n",
        "./1817/2/363_actuating_904.jpg\n",
    ]

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        train: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)

        # File existence check
        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(f"unable to locate {label_path if not os.path.exists(label_path) else img_folder}")

        self.data: List[Tuple[str, str]] = []
        self.train = train

        with open(label_path) as f:
            img_paths = f.readlines()

        train_samples = int(len(img_paths) * 0.9)
        set_slice = slice(train_samples) if self.train else slice(train_samples, None)

        for path in tqdm(iterable=img_paths[set_slice], desc="Unpacking MJSynth", total=len(img_paths[set_slice])):
            if path not in self.BLACKLIST:
                label = path.split("_")[1]
                img_path = os.path.join(img_folder, path[2:]).strip()

                self.data.append((img_path, label))

    def extra_repr(self) -> str:
        return f"train={self.train}"
