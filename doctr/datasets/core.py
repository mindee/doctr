# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from pathlib import Path
from zipfile import ZipFile
from typing import List, Any, Optional, Tuple
import tensorflow as tf

from doctr.models.utils import download_from_url


__all__ = ['AbstractDataset', 'VisionDataset']


class AbstractDataset:

    data: List[Any] = []

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[tf.Tensor, Any]:

        img_name, target = self.data[index]
        # Read image
        img = tf.io.read_file(os.path.join(self.root, img_name))
        img = tf.image.decode_jpeg(img, channels=3)
        if self.sample_transforms is not None:
            img = self.sample_transforms(img)

        return img, target

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"

    @staticmethod
    def collate_fn(samples: List[Tuple[tf.Tensor, Any]]) -> Tuple[tf.Tensor, List[Any]]:

        images, targets = zip(*samples)
        images = tf.stack(images, axis=0)

        return images, list(targets)


class VisionDataset(AbstractDataset):
    """Implements an abstract dataset

    Args:
        url: URL of the dataset
        file_name: name of the file once downloaded
        file_hash: expected SHA256 of the file
        extract_archive: whether the downloaded file is an archive to be extracted
        download: whether the dataset should be downloaded if not present on disk
        overwrite: whether the archive should be re-extracted
    """

    def __init__(
        self,
        url: str,
        file_name: Optional[str] = None,
        file_hash: Optional[str] = None,
        extract_archive: bool = False,
        download: bool = False,
        overwrite: bool = False,
    ) -> None:

        dataset_cache = os.path.join(os.path.expanduser('~'), '.cache', 'doctr', 'datasets')

        file_name = file_name if isinstance(file_name, str) else os.path.basename(url)
        # Download the file if not present
        archive_path = os.path.join(dataset_cache, file_name)

        if not os.path.exists(archive_path) and not download:
            raise ValueError("the dataset needs to be downloaded first with download=True")

        archive_path = download_from_url(url, file_name, file_hash, cache_subdir='datasets')

        # Extract the archive
        if extract_archive:
            archive_path = Path(archive_path)
            dataset_path = archive_path.parent.joinpath(archive_path.stem)
            if not dataset_path.is_dir() or overwrite:
                with ZipFile(archive_path, 'r') as f:
                    f.extractall(path=dataset_path)

        # List images
        self._root = dataset_path if extract_archive else archive_path
        self.data: List[Any] = []
