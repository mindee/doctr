import io
import logging
import os
from pathlib import PosixPath
from unittest.mock import patch

import pytest

from doctr.utils.data import _urlretrieve, download_from_url


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


@pytest.mark.parametrize("file_name", ["checkpoint-10.pt", "model-face.pt", "my-.zip"])
@patch("doctr.utils.data._urlretrieve")
def test_download_from_url_without_hash_in_file_name(urlretrieve_mock, file_name, tmp_path):
    # a short run of hex digits after a dash is part of the name, not the hash of the file
    file_path = tmp_path / file_name
    file_path.write_bytes(b"weights")
    assert download_from_url(f"https://example.com/{file_name}", cache_dir=str(tmp_path)) == file_path
    urlretrieve_mock.assert_not_called()
    assert file_path.read_bytes() == b"weights"


@patch("doctr.utils.data._urlretrieve")
def test_download_from_url_with_hash_in_file_name(urlretrieve_mock, tmp_path):
    file_path = tmp_path / "db_resnet50-79bd7d70.pt"
    file_path.write_bytes(b"weights")
    with pytest.raises(ValueError, match="corrupted download"):
        download_from_url("https://example.com/db_resnet50-79bd7d70.pt", cache_dir=str(tmp_path))
    urlretrieve_mock.assert_called_once()
    assert not file_path.exists()


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


class _Response(io.BytesIO):
    """A response that drops the connection after `fail_after` bytes, if set"""

    def __init__(self, content, fail_after=None):
        super().__init__(content)
        self.length = len(content)
        self.fail_after = fail_after

    def read(self, size=-1):
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise ConnectionResetError("connection dropped")
        return super().read(size)


def test_urlretrieve_interrupted_leaves_no_file(tmp_path):
    file_path = tmp_path / "model.pt"
    with patch("urllib.request.urlopen", return_value=_Response(b"x" * 4096, fail_after=2048)):
        with pytest.raises(ConnectionResetError):
            _urlretrieve("https://example.com/model.pt", file_path)
    # No truncated file is left behind to be reused by the next download
    assert list(tmp_path.iterdir()) == []

    with patch("urllib.request.urlopen", return_value=_Response(b"x" * 4096)):
        _urlretrieve("https://example.com/model.pt", file_path)
    assert file_path.read_bytes() == b"x" * 4096
    assert list(tmp_path.iterdir()) == [file_path]


def test_urlretrieve_concurrent_downloads(tmp_path):
    file_path = tmp_path / "model.pt"

    class _InterleavedResponse(_Response):
        """A response that runs a whole second download of the same file while the first one is in progress"""

        def read(self, size=-1):
            if self.tell() == 2048:
                with patch("urllib.request.urlopen", return_value=_Response(b"y" * 4096)):
                    _urlretrieve("https://example.com/model.pt", file_path)
            return super().read(size)

    with patch("urllib.request.urlopen", return_value=_InterleavedResponse(b"x" * 4096)):
        _urlretrieve("https://example.com/model.pt", file_path)
    # Both downloads complete, the last one to finish wins and no temporary file is left behind
    assert file_path.read_bytes() == b"x" * 4096
    assert list(tmp_path.iterdir()) == [file_path]
