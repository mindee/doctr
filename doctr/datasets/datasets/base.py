# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from doctr.io.image import get_img_shape
from doctr.utils.data import download_from_url

from ...models.utils import _copy_tensor

__all__ = ["_AbstractDataset", "_VisionDataset"]


class _AbstractDataset:
    data: List[Any] = []
    _pre_transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None

    def __init__(
        self,
        root: Union[str, Path],
        img_transforms: Optional[Callable[[Any], Any]] = None,
        sample_transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
        pre_transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    ) -> None:
        if not Path(root).is_dir():
            raise ValueError(f"expected a path to a reachable folder: {root}")

        self.root = root
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms
        self._pre_transforms = pre_transforms
        self._get_img_shape = get_img_shape

    def __len__(self) -> int:
        return len(self.data)

    def _read_sample(self, index: int) -> Tuple[Any, Any]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Read image
        img, target = self._read_sample(index)
        # Pre-transforms (format conversion at run-time etc.)
        if self._pre_transforms is not None:
            img, target = self._pre_transforms(img, target)

        if self.img_transforms is not None:
            # typing issue cf. https://github.com/python/mypy/issues/5485
            img = self.img_transforms(img)

        if self.sample_transforms is not None:
            # Conditions to assess it is detection model with multiple classes and avoid confusion with other tasks.
            if (
                isinstance(target, dict)
                and all(isinstance(item, np.ndarray) for item in target.values())
                and set(target.keys()) != {"boxes", "labels"}  # avoid confusion with obj detection target
            ):
                img_transformed = _copy_tensor(img)
                for class_name, bboxes in target.items():
                    img_transformed, target[class_name] = self.sample_transforms(img, bboxes)
                img = img_transformed
            else:
                img, target = self.sample_transforms(img, target)

        return img, target

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class _VisionDataset(_AbstractDataset):
    """Implements an abstract dataset

    Args:
    ----
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
        **kwargs: Any,
    ) -> None:
        cache_dir = (
            str(os.environ.get("DOCTR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "doctr")))
            if cache_dir is None
            else cache_dir
        )

        cache_subdir = "datasets" if cache_subdir is None else cache_subdir

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

        super().__init__(dataset_path if extract_archive else archive_path, **kwargs)
