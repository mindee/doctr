import logging
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
def test_download_from_url_error_creating_directory(mkdir_mock, caplog):
    with caplog.at_level(logging.ERROR, logger="doctr.utils.data"):
        with pytest.raises(OSError):
            download_from_url("test_url")
    assert (
        "Failed creating cache directory at /.cache/doctr."
        " You can change default cache directory using 'DOCTR_CACHE_DIR' environment variable if needed."
    ) in caplog.text


@patch.dict(os.environ, {"HOME": "/", "DOCTR_CACHE_DIR": "/test"}, clear=True)
@patch("pathlib.Path.mkdir", side_effect=OSError)
def test_download_from_url_error_creating_directory_with_env_var(mkdir_mock, caplog):
    with caplog.at_level(logging.ERROR, logger="doctr.utils.data"):
        with pytest.raises(OSError):
            download_from_url("test_url")
    assert (
        "Failed creating cache directory at /test using path from 'DOCTR_CACHE_DIR' environment variable."
    ) in caplog.text
