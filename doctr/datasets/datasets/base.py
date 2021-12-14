# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from doctr.utils.data import download_from_url

__all__ = ['_AbstractDataset', '_VisionDataset']


class _AbstractDataset:

    data: List[Any] = []

    def __init__(
        self,
        root: Union[str, Path],
    ) -> None:

        if not Path(root).is_dir():
            raise ValueError(f'expected a path to a reachable folder: {root}')

        self.root = root

    def __len__(self) -> int:
        return len(self.data)

    def _read_sample(self, index: int) -> Tuple[Any, Any]:
        raise NotImplementedError

    @staticmethod
    def _get_img_shape(img: Any) -> Tuple[int, int]:
        raise NotImplementedError

    def __getitem__(
        self,
        index: int
    ) -> Tuple[Any, Any]:

        # Read image
        img, target = self._read_sample(index)
        self.sample_transforms: Optional[Callable[[Any], Any]]
        if self.sample_transforms is not None:
            # typing issue cf. https://github.com/python/mypy/issues/5485
            img = self.sample_transforms(img)  # type: ignore[call-arg]

        return img, target

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class _VisionDataset(_AbstractDataset):
    """Implements an abstract dataset

    Args:
        url: URL of the dataset
        file_name: name of the file once downloaded
        file_hash: expected SHA256 of the file
        extract_archive: whether the downloaded file is an archive to be extracted
        download: whether the dataset should be downloaded if not present on disk
        overwrite: whether the archive should be re-extracted
        cache_dir: cache directory
        cache_subdir: subfolder to use in the cache
    """

    def __init__(
        self,
        url: str,
        file_name: Optional[str] = None,
        file_hash: Optional[str] = None,
        extract_archive: bool = False,
        download: bool = False,
        overwrite: bool = False,
        cache_dir: Optional[str] = None,
        cache_subdir: Optional[str] = None,
    ) -> None:

        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'doctr') if cache_dir is None else cache_dir
        cache_subdir = 'datasets' if cache_subdir is None else cache_subdir

        file_name = file_name if isinstance(file_name, str) else os.path.basename(url)
        # Download the file if not present
        archive_path: Union[str, Path] = os.path.join(cache_dir, cache_subdir, file_name)

        if not os.path.exists(archive_path) and not download:
            raise ValueError("the dataset needs to be downloaded first with download=True")

        archive_path = download_from_url(url, file_name, file_hash, cache_dir=cache_dir, cache_subdir=cache_subdir)

        # Extract the archive
        if extract_archive:
            archive_path = Path(archive_path)
            dataset_path = archive_path.parent.joinpath(archive_path.stem)
            if not dataset_path.is_dir() or overwrite:
                shutil.unpack_archive(archive_path, dataset_path)

        super().__init__(dataset_path if extract_archive else archive_path)
