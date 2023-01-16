# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py

import hashlib
import logging
import os
import re
import urllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Union

from tqdm.auto import tqdm

__all__ = ["download_from_url"]


# matches bfd8deac from resnet18-bfd8deac.ckpt
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")
USER_AGENT = "mindee/doctr"


def _urlretrieve(url: str, filename: Union[Path, str], chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def _check_integrity(file_path: Union[str, Path], hash_prefix: str) -> bool:
    with open(file_path, "rb") as f:
        sha_hash = hashlib.sha256(f.read()).hexdigest()

    return sha_hash[: len(hash_prefix)] == hash_prefix


def download_from_url(
    url: str,
    file_name: Optional[str] = None,
    hash_prefix: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_subdir: Optional[str] = None,
) -> Path:
    """Download a file using its URL

    >>> from doctr.models import download_from_url
    >>> download_from_url("https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        url: the URL of the file to download
        file_name: optional name of the file once downloaded
        hash_prefix: optional expected SHA256 hash of the file
        cache_dir: cache directory
        cache_subdir: subfolder to use in the cache

    Returns:
        the location of the downloaded file

    Note:
        You can change cache directory location by using `DOCTR_CACHE_DIR` environment variable.
    """

    if not isinstance(file_name, str):
        file_name = url.rpartition("/")[-1].split("&")[0]

    cache_dir = (
        str(os.environ.get("DOCTR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "doctr")))
        if cache_dir is None
        else cache_dir
    )

    # Check hash in file name
    if hash_prefix is None:
        r = HASH_REGEX.search(file_name)
        hash_prefix = r.group(1) if r else None

    folder_path = Path(cache_dir) if cache_subdir is None else Path(cache_dir, cache_subdir)
    file_path = folder_path.joinpath(file_name)
    # Check file existence
    if file_path.is_file() and (hash_prefix is None or _check_integrity(file_path, hash_prefix)):
        logging.info(f"Using downloaded & verified file: {file_path}")
        return file_path

    try:
        # Create folder hierarchy
        folder_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        error_message = f"Failed creating cache direcotry at {folder_path}"
        if os.environ.get("DOCTR_CACHE_DIR", ""):
            error_message += " using path from 'DOCTR_CACHE_DIR' environment variable."
        else:
            error_message += (
                ". You can change default cache directory using 'DOCTR_CACHE_DIR' environment variable if needed."
            )
        logging.error(error_message)
        raise
    # Download the file
    try:
        print(f"Downloading {url} to {file_path}")
        _urlretrieve(url, file_path)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead." f" Downloading {url} to {file_path}")
            _urlretrieve(url, file_path)
        else:
            raise e

    # Remove corrupted files
    if isinstance(hash_prefix, str) and not _check_integrity(file_path, hash_prefix):
        # Remove file
        os.remove(file_path)
        raise ValueError(f"corrupted download, the hash of {url} does not match its expected value")

    return file_path
