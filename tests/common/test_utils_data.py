import os
from pathlib import PosixPath
from unittest.mock import patch

import pytest

from doctr.utils.data import download_from_url


@patch("doctr.utils.data._urlretrieve")
@patch("pathlib.Path.mkdir")
@patch.dict(os.environ, {"HOME": "/"}, clear=True)
def test_download_from_url(mkdir_mock, urlretrieve_mock):
    download_from_url("test_url")
    urlretrieve_mock.assert_called_with("test_url", PosixPath("/.cache/doctr/test_url"))


@patch.dict(os.environ, {"DOCTR_CACHE_DIR": "/test"}, clear=True)
@patch("doctr.utils.data._urlretrieve")
@patch("pathlib.Path.mkdir")
def test_download_from_url_customizing_cache_dir(mkdir_mock, urlretrieve_mock):
    download_from_url("test_url")
    urlretrieve_mock.assert_called_with("test_url", PosixPath("/test/test_url"))


@patch.dict(os.environ, {"HOME": "/"}, clear=True)
@patch("pathlib.Path.mkdir", side_effect=OSError)
@patch("logging.error")
def test_download_from_url_error_creating_directory(logging_mock, mkdir_mock):
    with pytest.raises(OSError):
        download_from_url("test_url")
    logging_mock.assert_called_with(
        "Failed creating cache direcotry at /.cache/doctr."
        " You can change default cache directory using 'DOCTR_CACHE_DIR' environment variable if needed."
    )


@patch.dict(os.environ, {"HOME": "/", "DOCTR_CACHE_DIR": "/test"}, clear=True)
@patch("pathlib.Path.mkdir", side_effect=OSError)
@patch("logging.error")
def test_download_from_url_error_creating_directory_with_env_var(logging_mock, mkdir_mock):
    with pytest.raises(OSError):
        download_from_url("test_url")
    logging_mock.assert_called_with(
        "Failed creating cache direcotry at /test using path from 'DOCTR_CACHE_DIR' environment variable."
    )
